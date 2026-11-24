"""SLA policy semantic search and deterministic compensation calculator tools."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.observability.tracing import traced_operation
from app.persistence import get_logistics_store
from app.schemas import (
    CompensationQuoteResponse,
    GuidedToolError,
    PolicySearchInput,
    PolicySearchResponse,
    SlaCompensationInput,
)

SUPERVISOR_LIMIT_USD = 100.0


@traced_operation("swiftship.tool_impl.search_sla_policy_knowledge_base")
def search_sla_policy_knowledge_base(query: str, top_k: int) -> dict[str, Any]:
    """Search SwiftShip SLA policy clauses using vector similarity retrieval.

    Queries the persistent `sla_policy_chunks` vector store (embedded with Vertex AI
    `text-embedding-005`) to retrieve authoritative SLA guarantee rules, thermal
    excursion liability clauses, and monetary approval governance thresholds.

    Args:
        query: Natural-language description of the shipping scenario, SLA tier,
            delay duration, or damage condition (4 to 400 characters).
        top_k: Maximum number of top-matching policy clauses to return (integer
            between 1 and 5, typically 3).

    Returns:
        A JSON-serializable dictionary matching `PolicySearchResponse` containing
        ranked `PolicyClauseHit` items with `policy_clause_id` citations, or a
        `GuidedToolError` with `recovery_instructions` if arguments are invalid.
    """
    try:
        validated = PolicySearchInput(query=query.strip(), top_k=top_k)
    except ValidationError as exc:
        return GuidedToolError(
            error_code="INVALID_POLICY_SEARCH_INPUT",
            message=f"Policy search validation failed: {exc.errors()[0]['msg']}.",
            recovery_instructions=(
                "Retry `search_sla_policy_knowledge_base` with a descriptive `query` "
                "of at least 4 characters (e.g., 'express_48h late delivery credit') "
                "and `top_k` between 1 and 5."
            ),
            retryable=True,
        ).model_dump()

    store = get_logistics_store()
    matches = store.search_policies(validated.query, top_k=validated.top_k)
    return PolicySearchResponse(
        retrieval_backend=store.backend_name,
        matches=matches,
    ).model_dump()


@traced_operation("swiftship.tool_impl.calculate_sla_breach_compensation")
def calculate_sla_breach_compensation(
    order_id: str,
    sla_tier: str,
    merchandise_value_usd: float,
    shipping_fee_usd: float,
    delay_hours: int,
    damage_or_spoilage_confirmed: bool,
) -> dict[str, Any]:
    """Calculate deterministic SLA breach compensation and supervisor gate status.

    Evaluates verified order telemetry against SwiftShip's four SLA service tiers
    to compute the exact USD refund or credit and determine whether Human-in-the-Loop
    supervisor confirmation (`SLA-GOV-09`) is mandatory.

    Args:
        order_id: Verified SwiftShip order identifier (`ORD-XXXXX`).
        sla_tier: Contracted SLA tier (`standard_ground`, `express_48h`,
            `priority_overnight`, or `cold_chain_biologics`).
        merchandise_value_usd: Verified merchandise value in USD (>= 0).
        shipping_fee_usd: Verified paid shipping fee in USD (>= 0).
        delay_hours: Verified delivery delay past SLA commitment in hours (>= 0).
        damage_or_spoilage_confirmed: True only if physical damage or a cold-chain
            thermal excursion (> 8.0C) is confirmed in the order record.

    Returns:
        A JSON-serializable dictionary matching `CompensationQuoteResponse` with
        the exact USD compensation and `requires_human_supervisor_approval` flag,
        or a `GuidedToolError` if input validation fails.
    """
    try:
        validated = SlaCompensationInput(
            order_id=order_id.strip().upper(),
            sla_tier=sla_tier,  # type: ignore[arg-type]
            merchandise_value_usd=merchandise_value_usd,
            shipping_fee_usd=shipping_fee_usd,
            delay_hours=delay_hours,
            damage_or_spoilage_confirmed=damage_or_spoilage_confirmed,
        )
    except ValidationError as exc:
        return GuidedToolError(
            error_code="INVALID_COMPENSATION_PARAMETERS",
            message=f"Compensation input validation failed: {exc.errors()[0]['msg']}.",
            recovery_instructions=(
                "First call `fetch_verified_shipment_order` and pass the exact "
                "`order_id`, `sla_tier`, `merchandise_value_usd`, `shipping_fee_usd`, "
                "and `delay_hours` values from that verified record."
            ),
            retryable=True,
        ).model_dump()

    tier = validated.sla_tier
    if tier == "cold_chain_biologics" and validated.damage_or_spoilage_confirmed:
        amount = round(validated.merchandise_value_usd + validated.shipping_fee_usd, 2)
        action = "full_order_refund"
        clause = "SLA-CLD-04"
        eligible = True
        rationale = (
            "Confirmed cold-chain thermal excursion (>8.0C) or spoilage entitles "
            "the customer to 100% merchandise + shipping refund under SLA-CLD-04."
        )
    elif tier == "express_48h" and validated.delay_hours >= 24:
        amount = round(validated.shipping_fee_usd, 2)
        action = "shipping_fee_credit"
        clause = "SLA-EXP-02"
        eligible = True
        rationale = (
            f"Express 48h delay of {validated.delay_hours}h (>=24h) qualifies for "
            "100% paid shipping fee credit under SLA-EXP-02."
        )
    elif tier == "priority_overnight" and validated.delay_hours >= 12:
        amount = round(
            min(
                validated.shipping_fee_usd + 25.0,
                validated.merchandise_value_usd + validated.shipping_fee_usd,
            ),
            2,
        )
        action = "partial_service_credit"
        clause = "SLA-PRI-01"
        eligible = True
        rationale = (
            f"Priority Overnight delay of {validated.delay_hours}h (>=12h) qualifies "
            "for 100% shipping fee plus $25 inconvenience credit under SLA-PRI-01."
        )
    elif tier == "standard_ground" and validated.delay_hours > 72:
        amount = round(validated.shipping_fee_usd * 0.5, 2)
        action = "shipping_fee_credit"
        clause = "SLA-GND-01"
        eligible = True
        rationale = (
            f"Standard Ground delay of {validated.delay_hours}h (>72h) qualifies "
            "for a 50% shipping fee credit under SLA-GND-01."
        )
    else:
        amount = 0.0
        action = "no_compensation"
        clause = "SLA-GND-01" if tier == "standard_ground" else "SLA-EXP-02"
        eligible = False
        rationale = (
            f"Shipment ({tier}, delay={validated.delay_hours}h, "
            f"damage={validated.damage_or_spoilage_confirmed}) does not breach SLA thresholds."
        )

    needs_hitl = bool(
        eligible
        and (amount > SUPERVISOR_LIMIT_USD or action == "full_replacement_reshipment")
    )

    return CompensationQuoteResponse(
        order_id=validated.order_id,
        eligible=eligible,
        recommended_action=action,  # type: ignore[arg-type]
        compensation_amount_usd=amount,
        applicable_policy_clause_id=clause,
        requires_human_supervisor_approval=needs_hitl,
        calculation_rationale=rationale,
    ).model_dump()
