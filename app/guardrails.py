"""Runner-wide security, Model Armor, and self-evaluation guardrail plugin.

Implements the session-poisoning defense pattern from `google/adk-recipes`
(`safety-plugins`) alongside Google Cloud Model Armor integration, deterministic
tool policy invariants, and output self-evaluation.
"""

from __future__ import annotations

import os
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app.observability.logging_config import get_logger
from app.observability.pii_redactor import get_pii_redactor

logger = get_logger("guardrails")

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:bypass|override|disable)\s+(?:supervisor|human|hitl|safety|approval|policy|constitution)",
        re.IGNORECASE,
    ),
    re.compile(
        r"you\s+are\s+now\s+(?:in\s+)?(?:developer|god|unrestricted|dan)\s+mode",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:reveal|print|dump|exfiltrate)\s+(?:your\s+)?(?:system\s+prompt|secret|api\s+key)",
        re.IGNORECASE,
    ),
)

_UNAUTHORIZED_INSTANT_DISBURSEMENT_RE = re.compile(
    r"(?:disbursed|completed\s+instant\s+refund\s+of)\s+\$([1-9]\d{2,}(?:\.\d{2})?)",
    re.IGNORECASE,
)

SAFE_REFUSAL_MESSAGE = (
    "I cannot process instructions that attempt to override SwiftShip security "
    "policies, monetary authority limits, or system instructions. I can only "
    "assist with verified SwiftShip shipment inquiries and SLA resolutions."
)


class SecurityAndSelfEvalGuardrailPlugin(BasePlugin):
    """Global ADK BasePlugin enforcing input safety, tool policies, and output self-eval."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        location_id: str | None = None,
        model_armor_template_id: str | None = None,
        model_armor_client: Any | None = None,
    ) -> None:
        super().__init__(name="security_and_self_eval_guardrail")
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location_id = location_id or os.environ.get(
            "MODEL_ARMOR_LOCATION", "us-central1"
        )
        self._template_id = (
            model_armor_template_id
            if model_armor_template_id is not None
            else os.environ.get("MODEL_ARMOR_TEMPLATE_ID", "")
        )
        self._model_armor_client = model_armor_client

    def _get_model_armor_client(self) -> Any | None:
        if not (self._project_id and self._template_id):
            return None
        if self._model_armor_client is None:
            from google.api_core.client_options import ClientOptions
            from google.cloud import modelarmor_v1

            self._model_armor_client = modelarmor_v1.ModelArmorClient(
                client_options=ClientOptions(
                    api_endpoint=f"modelarmor.{self._location_id}.rep.googleapis.com"
                )
            )
        return self._model_armor_client

    def _check_model_armor_prompt(self, text: str) -> str | None:
        client = self._get_model_armor_client()
        if client is None:
            return None
        try:
            from google.cloud import modelarmor_v1

            template_name = (
                f"projects/{self._project_id}/locations/{self._location_id}"
                f"/templates/{self._template_id}"
            )
            resp = client.sanitize_user_prompt(
                request=modelarmor_v1.SanitizeUserPromptRequest(
                    name=template_name,
                    user_prompt_data=modelarmor_v1.DataItem(text=text),
                )
            )
            match_state = getattr(resp.sanitization_result, "filter_match_state", None)
            if match_state and "MATCH_FOUND" in str(match_state):
                return "model_armor_filter_match"
        except Exception as exc:
            logger.warning(
                "Model Armor check degraded cleanly to local policy rules",
                extra={
                    "event_type": "model_armor_fallback",
                    "error_type": type(exc).__name__,
                },
            )
        return None

    def inspect_user_text(self, text: str) -> str | None:
        """Return violation reason string if input violates security policy, else None."""
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                return f"prompt_injection_pattern:{pattern.pattern[:36]}"
        return self._check_model_armor_prompt(text)

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        if not user_message.parts:
            return None
        combined_text = "\n".join(p.text for p in user_message.parts if p.text)
        violation = self.inspect_user_text(combined_text)
        if violation is None:
            return None

        # Session-poisoning defense: scrub unsafe text from stored history and set halt flag
        invocation_context.session.state["is_user_prompt_safe"] = False
        invocation_context.session.state["security_block_reason"] = violation
        logger.warning(
            "Blocked adversarial user prompt before persistence",
            extra={
                "event_type": "guardrail_input_blocked",
                "violation": violation,
                "session_id": invocation_context.session.id,
            },
        )
        return types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="[User message redacted by SwiftShip Security Guardrail]"
                )
            ],
        )

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        if not invocation_context.session.state.get("is_user_prompt_safe", True):
            invocation_context.session.state["is_user_prompt_safe"] = True
            return types.Content(
                role="model",
                parts=[types.Part.from_text(text=SAFE_REFUSAL_MESSAGE)],
            )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        if tool.name == "execute_high_value_order_remediation":
            raw_amount = tool_args.get("remediation_amount_usd", 0.0)
            try:
                amount = float(raw_amount)
            except (TypeError, ValueError):
                amount = -1.0
            if amount <= 0.0 or amount > 10000.0:
                logger.warning(
                    "Guardrail blocked out-of-bounds remediation tool call",
                    extra={
                        "event_type": "guardrail_tool_blocked",
                        "tool_name": tool.name,
                        "requested_amount": raw_amount,
                    },
                )
                return {
                    "status": "error",
                    "error_code": "GUARDRAIL_AMOUNT_POLICY_VIOLATION",
                    "message": f"Guardrail blocked invalid remediation amount: {raw_amount!r}.",
                    "recovery_instructions": (
                        "Call `calculate_sla_breach_compensation` first and pass a "
                        "positive USD amount within policy limits."
                    ),
                }
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Self-evaluate outgoing LLM text for leaked PII or unconfirmed >$100 claims."""
        content = llm_response.content
        if not content or not content.parts:
            return None
        # Do not alter pure function-call turns or structured JSON planner outputs
        if any(p.function_call is not None for p in content.parts):
            return None
        if callback_context.agent_name == "remediation_strategist_agent":
            return None

        redactor = get_pii_redactor()
        modified = False
        new_parts: list[types.Part] = []

        for part in content.parts:
            if not part.text:
                new_parts.append(part)
                continue
            cleaned = redactor.scrub_text(part.text)
            # Self-eval check: ensure agent never claims an unapproved >$100 instant payout
            match = _UNAUTHORIZED_INSTANT_DISBURSEMENT_RE.search(cleaned)
            pending = callback_context.state.get("pending_supervisor_approval")
            if match and pending:
                cleaned += (
                    "\n\n[Policy Self-Check Note: Under SLA-GOV-09, disbursements "
                    "above $100.00 USD remain pending until supervisor confirmation.]"
                )
                modified = True
            if cleaned != part.text:
                modified = True
            new_parts.append(types.Part.from_text(text=cleaned))

        if not modified:
            return None

        logger.info(
            "Output self-evaluation guardrail sanitized model response",
            extra={
                "event_type": "guardrail_output_sanitized",
                "agent_name": callback_context.agent_name,
            },
        )
        return LlmResponse(
            content=types.Content(role=content.role or "model", parts=new_parts)
        )
