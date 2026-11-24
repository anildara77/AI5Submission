"""Strict Pydantic v2 input/output JSON schemas and guided error contracts.

Constrains LLM tool arguments, validates return payloads, and defines the
structured `RemediationStrategyPlan` emitted by the Gemini Pro planner.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    """Base schema forbidding undeclared fields to prevent LLM argument drift."""

    model_config = ConfigDict(extra="forbid")


class GuidedToolError(StrictSchema):
    """Structured error payload guiding the LLM on how to recover without crashing."""

    status: Literal["error"] = Field(
        default="error",
        description="Discriminator indicating the tool call did not succeed.",
    )
    error_code: str = Field(
        description="Machine-readable error classification code.",
    )
    message: str = Field(
        description="Human-readable explanation of what failed.",
    )
    recovery_instructions: str = Field(
        description=(
            "Explicit next-step instructions telling the LLM how to recover, "
            "correct arguments, or ask the user for missing details."
        ),
    )
    retryable: bool = Field(
        default=False,
        description="Whether retrying with corrected inputs can succeed.",
    )


class ShipmentLookupInput(StrictSchema):
    """Validated input schema for `fetch_verified_shipment_order`."""

    order_id: str = Field(
        pattern=r"^ORD-\d{5}$",
        description="Canonical SwiftShip order identifier in ORD-XXXXX format (e.g., ORD-10042).",
    )


class ShipmentOrderRecord(StrictSchema):
    """Persistent order record stored in BigQuery `swiftship_ops.orders`."""

    order_id: str = Field(description="Canonical SwiftShip order ID (ORD-XXXXX).")
    tracking_number: str = Field(
        description="Carrier tracking identifier (TRK-XXXXXXXX)."
    )
    sla_tier: Literal[
        "standard_ground", "express_48h", "priority_overnight", "cold_chain_biologics"
    ] = Field(description="Contracted SwiftShip SLA service tier.")
    shipment_status: Literal[
        "in_transit",
        "delayed_at_hub",
        "delivered_late",
        "delivered_damaged",
        "delivered_on_time",
    ] = Field(description="Current carrier scan status.")
    merchandise_value_usd: float = Field(
        ge=0.0, description="Declared merchandise value in USD."
    )
    shipping_fee_usd: float = Field(ge=0.0, description="Paid shipping fee in USD.")
    delay_hours: int = Field(
        ge=0, description="Verified delivery delay beyond SLA commitment in hours."
    )
    max_recorded_temp_celsius: float | None = Field(
        default=None,
        description="Peak sensor temperature in Celsius for cold-chain shipments, or null for ambient freight.",
    )
    contents_summary: str = Field(description="Brief non-PII summary of shipped goods.")


class ShipmentOrderResponse(StrictSchema):
    """Validated output schema returned by `fetch_verified_shipment_order`."""

    status: Literal["success"] = "success"
    order: ShipmentOrderRecord


class PolicySearchInput(StrictSchema):
    """Validated input schema for `search_sla_policy_knowledge_base`."""

    query: str = Field(
        min_length=4,
        max_length=400,
        description="Natural-language policy search query describing the SLA tier, delay, or damage scenario.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of policy clauses to retrieve (1 to 5).",
    )


class PolicyClauseHit(StrictSchema):
    """Single policy clause retrieved from the knowledge base."""

    policy_clause_id: str = Field(
        description="Authoritative SwiftShip policy clause identifier (e.g., SLA-EXP-02)."
    )
    title: str = Field(description="Title of the policy section.")
    sla_tier: str = Field(description="Applicable SLA tier or 'all'.")
    clause_text: str = Field(description="Verbatim policy rule text.")
    similarity_score: float = Field(
        description="Semantic relevance score (higher is closer)."
    )


class PolicySearchResponse(StrictSchema):
    """Validated output schema returned by `search_sla_policy_knowledge_base`."""

    status: Literal["success"] = "success"
    retrieval_backend: str = Field(description="Underlying vector/search store used.")
    matches: list[PolicyClauseHit]


class SlaCompensationInput(StrictSchema):
    """Validated input schema for `calculate_sla_breach_compensation`."""

    order_id: str = Field(
        pattern=r"^ORD-\d{5}$",
        description="Verified SwiftShip order ID (ORD-XXXXX).",
    )
    sla_tier: Literal[
        "standard_ground", "express_48h", "priority_overnight", "cold_chain_biologics"
    ] = Field(description="Contracted SLA tier from the verified order record.")
    merchandise_value_usd: float = Field(
        ge=0.0, description="Verified merchandise value in USD."
    )
    shipping_fee_usd: float = Field(
        ge=0.0, description="Verified paid shipping fee in USD."
    )
    delay_hours: int = Field(ge=0, description="Verified SLA delay in hours.")
    damage_or_spoilage_confirmed: bool = Field(
        description="True only when physical damage or cold-chain thermal excursion (>8.0C) is confirmed in telemetry."
    )


class CompensationQuoteResponse(StrictSchema):
    """Validated output schema returned by `calculate_sla_breach_compensation`."""

    status: Literal["success"] = "success"
    order_id: str
    eligible: bool
    recommended_action: Literal[
        "no_compensation",
        "shipping_fee_credit",
        "partial_service_credit",
        "full_order_refund",
        "full_replacement_reshipment",
    ]
    compensation_amount_usd: float = Field(ge=0.0)
    applicable_policy_clause_id: str
    requires_human_supervisor_approval: bool = Field(
        description="True when compensation > $100.00 USD or action is full_replacement_reshipment."
    )
    calculation_rationale: str


class OrderRemediationInput(StrictSchema):
    """Validated input schema for `execute_high_value_order_remediation`."""

    order_id: str = Field(
        pattern=r"^ORD-\d{5}$",
        description="Verified SwiftShip order ID (ORD-XXXXX).",
    )
    action_type: Literal[
        "shipping_fee_credit",
        "partial_service_credit",
        "full_order_refund",
        "full_replacement_reshipment",
    ] = Field(description="Remediation action to execute.")
    remediation_amount_usd: float = Field(
        gt=0.0,
        le=10000.0,
        description="Exact USD value of the credit or refund to issue.",
    )
    policy_clause_id: str = Field(
        min_length=4,
        max_length=32,
        description="Authoritative SwiftShip policy clause ID justifying this action (e.g., SLA-EXP-02).",
    )
    justification: str = Field(
        min_length=10,
        max_length=500,
        description="Concise factual justification referencing verified order metrics.",
    )


class RemediationExecutionResponse(StrictSchema):
    """Validated output schema returned by `execute_high_value_order_remediation`."""

    status: Literal["success", "pending_human_approval"]
    remediation_id: str
    order_id: str
    action_type: str
    disbursed_amount_usd: float
    policy_clause_id: str
    approval_mode: Literal[
        "auto_approved_under_limit",
        "human_supervisor_confirmed",
        "awaiting_human_supervisor",
    ]
    partner_gateway_reference: str
    customer_summary: str


class RemediationStrategyPlan(StrictSchema):
    """Structured plan generated by the Gemini Pro `remediation_strategist_agent`."""

    order_id: str = Field(
        description="Verified order ID under review, or 'UNVERIFIED' if missing."
    )
    eligibility_verdict: Literal["eligible", "ineligible", "needs_valid_order_id"] = (
        Field(
            description="Whether the shipment qualifies for SLA compensation or remediation."
        )
    )
    cited_policy_clause_ids: list[str] = Field(
        description="Exact policy clause IDs retrieved from the knowledge base."
    )
    recommended_action: Literal[
        "none",
        "request_valid_order_id",
        "shipping_fee_credit",
        "partial_service_credit",
        "full_order_refund",
        "full_replacement_reshipment",
    ] = Field(description="Recommended remediation action.")
    estimated_compensation_usd: float = Field(
        ge=0.0,
        description="Estimated compensation in USD based on verified order figures.",
    )
    requires_human_approval: bool = Field(
        description="True if estimated_compensation_usd > 100.0 or action is full_replacement_reshipment."
    )
    planning_rationale: str = Field(
        description="Step-by-step SLA policy analysis connecting shipment telemetry to the recommended remedy."
    )
