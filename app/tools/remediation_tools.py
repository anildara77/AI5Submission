"""Order remediation execution tool with explicit Human-in-the-Loop confirmation.

Enforces policy `SLA-GOV-09`:
- Remediations <= $100.00 USD execute autonomously and record a Secret Manager
  HMAC-signed disbursement entry in the persistent BigQuery ledger.
- Remediations > $100.00 USD or `full_replacement_reshipment` invoke ADK's
  `tool_context.request_confirmation(...)` stop hook, halting disbursement until
  a human supervisor explicitly approves the action.
"""

from __future__ import annotations

import uuid
from typing import Any

from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from app.observability.logging_config import get_logger
from app.observability.tracing import traced_operation
from app.persistence import get_logistics_store
from app.schemas import (
    GuidedToolError,
    OrderRemediationInput,
    RemediationExecutionResponse,
)
from app.secrets import sign_remediation_disbursement

logger = get_logger("remediation_tools")

SUPERVISOR_APPROVAL_THRESHOLD_USD = 100.0


def requires_supervisor_approval(
    remediation_amount_usd: float,
    action_type: str,
) -> bool:
    """Return True when policy SLA-GOV-09 mandates human supervisor confirmation."""
    return (
        remediation_amount_usd > SUPERVISOR_APPROVAL_THRESHOLD_USD
        or action_type == "full_replacement_reshipment"
    )


@traced_operation("swiftship.tool_impl.execute_high_value_order_remediation")
def execute_high_value_order_remediation(
    order_id: str,
    action_type: str,
    remediation_amount_usd: float,
    policy_clause_id: str,
    justification: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Execute an approved customer refund, SLA credit, or replacement reshipment.

    Validates the requested remediation against the verified order's maximum
    liability cap, enforces the mandatory $100.00 USD Human-in-the-Loop supervisor
    confirmation gate (`SLA-GOV-09`), signs the transaction via Google Cloud Secret
    Manager, and writes an immutable audit row to `swiftship_ops.remediation_ledger`.

    Args:
        order_id: Verified SwiftShip order identifier (`ORD-XXXXX`).
        action_type: One of `shipping_fee_credit`, `partial_service_credit`,
            `full_order_refund`, or `full_replacement_reshipment`.
        remediation_amount_usd: Positive USD value to disburse (must not exceed
            `merchandise_value_usd + shipping_fee_usd` for the verified order).
        policy_clause_id: Authoritative policy clause justifying disbursement
            (for example, `SLA-EXP-02` or `SLA-CLD-04`).
        justification: Factual explanation referencing verified shipment metrics
            (10 to 500 characters).

    Returns:
        A JSON-serializable dictionary matching `RemediationExecutionResponse`
        (`status="success"` when auto-approved or human-confirmed;
        `status="pending_human_approval"` when halted at the HITL confirmation gate),
        or a `GuidedToolError` with `recovery_instructions` if validation fails.
    """
    try:
        validated = OrderRemediationInput(
            order_id=order_id.strip().upper(),
            action_type=action_type,  # type: ignore[arg-type]
            remediation_amount_usd=round(remediation_amount_usd, 2),
            policy_clause_id=policy_clause_id.strip().upper(),
            justification=justification.strip(),
        )
    except ValidationError as exc:
        return GuidedToolError(
            error_code="INVALID_REMEDIATION_ARGUMENTS",
            message=f"Remediation input validation failed: {exc.errors()[0]['msg']}.",
            recovery_instructions=(
                "Verify `order_id` matches 'ORD-XXXXX', `action_type` is one of the "
                "four supported remediation types, `remediation_amount_usd` > 0, and "
                "`justification` is at least 10 characters."
            ),
            retryable=True,
        ).model_dump()

    store = get_logistics_store()
    order = store.get_order(validated.order_id)
    if order is None:
        return GuidedToolError(
            error_code="UNVERIFIED_ORDER_REMEDIATION_BLOCKED",
            message=f"Cannot disburse funds for unverified order {validated.order_id!r}.",
            recovery_instructions=(
                "Call `fetch_verified_shipment_order` first to confirm the order "
                "exists before attempting any remediation."
            ),
            retryable=False,
        ).model_dump()

    max_cap = round(order.merchandise_value_usd + order.shipping_fee_usd, 2)
    if validated.remediation_amount_usd > max_cap:
        return GuidedToolError(
            error_code="LIABILITY_CAP_EXCEEDED",
            message=(
                f"Requested amount ${validated.remediation_amount_usd:.2f} exceeds "
                f"maximum liability cap ${max_cap:.2f} for {validated.order_id}."
            ),
            recovery_instructions=(
                "Call `calculate_sla_breach_compensation` and retry with the exact "
                f"policy-authorized amount (<= ${max_cap:.2f})."
            ),
            retryable=True,
        ).model_dump()

    remediation_id = f"REM-{uuid.uuid4().hex[:8].upper()}"
    needs_approval = requires_supervisor_approval(
        validated.remediation_amount_usd, validated.action_type
    )

    if needs_approval:
        confirmation = getattr(tool_context, "tool_confirmation", None)
        if confirmation is None:
            approval_payload = {
                "remediation_id": remediation_id,
                "order_id": validated.order_id,
                "action_type": validated.action_type,
                "remediation_amount_usd": validated.remediation_amount_usd,
                "policy_clause_id": validated.policy_clause_id,
                "governance_rule": "SLA-GOV-09",
            }
            tool_context.request_confirmation(
                hint=(
                    f"Supervisor approval required under SLA-GOV-09: Approve "
                    f"{validated.action_type} of ${validated.remediation_amount_usd:.2f} "
                    f"for order {validated.order_id} ({validated.policy_clause_id})?"
                ),
                payload=approval_payload,
            )
            tool_context.state["pending_supervisor_approval"] = approval_payload
            logger.warning(
                "Halted high-value remediation pending human supervisor confirmation",
                extra={
                    "event_type": "hitl_confirmation_requested",
                    "remediation_id": remediation_id,
                    "order_id": validated.order_id,
                    "requested_amount_usd": validated.remediation_amount_usd,
                    "action_type": validated.action_type,
                },
            )
            return RemediationExecutionResponse(
                status="pending_human_approval",
                remediation_id=remediation_id,
                order_id=validated.order_id,
                action_type=validated.action_type,
                disbursed_amount_usd=0.0,
                policy_clause_id=validated.policy_clause_id,
                approval_mode="awaiting_human_supervisor",
                partner_gateway_reference="PENDING-SUPERVISOR-CONFIRMATION",
                customer_summary=(
                    f"Verified {validated.action_type} of "
                    f"${validated.remediation_amount_usd:.2f} for {validated.order_id} "
                    f"under {validated.policy_clause_id} exceeds the $100.00 autonomous "
                    "limit (SLA-GOV-09) and is paused awaiting human supervisor sign-off."
                ),
            ).model_dump()

        if not confirmation.confirmed:
            return GuidedToolError(
                error_code="SUPERVISOR_CONFIRMATION_REJECTED",
                message=(
                    f"Human supervisor declined {validated.action_type} of "
                    f"${validated.remediation_amount_usd:.2f} for {validated.order_id}."
                ),
                recovery_instructions=(
                    "Do NOT retry `execute_high_value_order_remediation`. Inform the "
                    "customer politely that the high-value claim requires manual "
                    "claims-specialist review."
                ),
                retryable=False,
            ).model_dump()

        approval_mode = "human_supervisor_confirmed"
    else:
        approval_mode = "auto_approved_under_limit"

    gateway_ref = sign_remediation_disbursement(
        validated.order_id, validated.remediation_amount_usd, remediation_id
    )
    store.record_remediation(
        {
            "remediation_id": remediation_id,
            "order_id": validated.order_id,
            "action_type": validated.action_type,
            "disbursed_amount_usd": validated.remediation_amount_usd,
            "policy_clause_id": validated.policy_clause_id,
            "approval_mode": approval_mode,
            "partner_gateway_reference": gateway_ref,
            "justification": validated.justification,
        }
    )

    return RemediationExecutionResponse(
        status="success",
        remediation_id=remediation_id,
        order_id=validated.order_id,
        action_type=validated.action_type,
        disbursed_amount_usd=validated.remediation_amount_usd,
        policy_clause_id=validated.policy_clause_id,
        approval_mode=approval_mode,  # type: ignore[arg-type]
        partner_gateway_reference=gateway_ref,
        customer_summary=(
            f"Issued {validated.action_type} of ${validated.remediation_amount_usd:.2f} "
            f"for {validated.order_id} ({validated.policy_clause_id}), reference {gateway_ref}."
        ),
    ).model_dump()
