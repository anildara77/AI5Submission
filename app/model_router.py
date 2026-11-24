"""Strategic model routing across Gemini Flash (speed) and Gemini Pro (planning).

Assigns:
- `FLASH_MODEL` (`gemini-3.8-flash`) for low-latency triage, tool-driven order &
  policy retrieval, history compaction, and final execution.
- `PRO_PLANNING_MODEL` (`gemini-3.1-pro-preview`) for multi-clause SLA reasoning and
  structured `RemediationStrategyPlan` synthesis, plus dynamic escalation when
  complex multi-order or thermal-excursion disputes arise.
"""

from __future__ import annotations

import os

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from app.observability.logging_config import get_logger

logger = get_logger("model_router")

FLASH_MODEL = os.environ.get("SWIFTSHIP_FLASH_MODEL", "gemini-3.8-flash")
PRO_PLANNING_MODEL = os.environ.get("SWIFTSHIP_PRO_MODEL", "gemini-3.1-pro-preview")

_COMPLEX_DISPUTE_KEYWORDS: tuple[str, ...] = (
    "cold-chain",
    "cold chain",
    "biologics",
    "thermal",
    "spoilage",
    "arbitration",
    "multiple orders",
)


def build_flash_model() -> Gemini:
    """Instantiate fast, cost-efficient Gemini Flash model with retry policy."""
    return Gemini(
        model=FLASH_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    )


def build_pro_planning_model() -> Gemini:
    """Instantiate high-reasoning Gemini Pro model for SLA strategy planning."""
    return Gemini(
        model=PRO_PLANNING_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    )


def select_model_for_complexity(
    *, agent_name: str, prompt_text: str = ""
) -> tuple[str, str]:
    """Return `(selected_model_id, routing_reason)` based on agent role & query."""
    if agent_name == "remediation_strategist_agent":
        return PRO_PLANNING_MODEL, "dedicated_pro_sla_planning_tier"

    lowered = prompt_text.lower()
    if (
        agent_name == "swiftship_ops_agent"
        and sum(1 for kw in _COMPLEX_DISPUTE_KEYWORDS if kw in lowered) >= 3
    ):
        return PRO_PLANNING_MODEL, "dynamic_complex_dispute_escalation"

    return FLASH_MODEL, "low_latency_flash_operational_tier"


class StrategicModelRoutingPlugin(BasePlugin):
    """Observability & dynamic model routing plugin recording Flash vs. Pro usage."""

    def __init__(self) -> None:
        super().__init__(name="strategic_model_routing_plugin")

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        latest_text = ""
        if llm_request.contents:
            last = llm_request.contents[-1]
            if last.parts:
                latest_text = " ".join(p.text for p in last.parts if p.text)

        target_model, reason = select_model_for_complexity(
            agent_name=callback_context.agent_name,
            prompt_text=latest_text,
        )
        llm_request.model = target_model
        callback_context.state["last_routed_model"] = target_model

        logger.info(
            "Routed LLM request to strategic model tier",
            extra={
                "event_type": "strategic_model_routed",
                "agent_name": callback_context.agent_name,
                "selected_model": target_model,
                "routing_reason": reason,
                "invocation_id": callback_context.invocation_id,
            },
        )
        return None
