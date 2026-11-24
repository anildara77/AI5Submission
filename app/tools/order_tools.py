"""Order verification tool backed by persistent BigQuery storage."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.observability.tracing import traced_operation
from app.persistence import get_logistics_store
from app.schemas import (
    GuidedToolError,
    ShipmentLookupInput,
    ShipmentOrderResponse,
)


@traced_operation("swiftship.tool_impl.fetch_verified_shipment_order")
def fetch_verified_shipment_order(order_id: str) -> dict[str, Any]:
    """Fetch a verified shipment order record and telemetry from persistent storage.

    Queries the authoritative SwiftShip `orders` table to retrieve contracted SLA
    tier, carrier tracking number, paid merchandise and shipping amounts, verified
    delivery delay hours, and cold-chain temperature sensor readings.

    Args:
        order_id: Canonical SwiftShip order identifier formatted as `ORD-XXXXX`
            (for example, `ORD-10042` or `ORD-10089`).

    Returns:
        A JSON-serializable dictionary matching `ShipmentOrderResponse` when the
        order exists, or `GuidedToolError` with explicit `recovery_instructions`
        when the format is invalid or the order cannot be located.
    """
    normalized_id = order_id.strip().upper()
    try:
        validated = ShipmentLookupInput(order_id=normalized_id)
    except ValidationError:
        return GuidedToolError(
            error_code="INVALID_ORDER_ID_FORMAT",
            message=(
                f"Provided identifier {order_id!r} does not match the canonical "
                "SwiftShip format 'ORD-XXXXX' (prefix 'ORD-' followed by 5 digits)."
            ),
            recovery_instructions=(
                "Do NOT invent or guess an order ID. Politely ask the customer to "
                "provide their 5-digit SwiftShip order identifier in 'ORD-XXXXX' "
                "format (for example, 'ORD-10042') found on their shipping invoice."
            ),
            retryable=True,
        ).model_dump()

    try:
        order = get_logistics_store().get_order(validated.order_id)
    except Exception as exc:
        return GuidedToolError(
            error_code="ORDER_STORE_TRANSIENT_FAILURE",
            message=f"Temporary error reading persistent order store: {type(exc).__name__}.",
            recovery_instructions=(
                "Apologize for the brief lookup interruption and retry "
                "`fetch_verified_shipment_order` once with the same `order_id`."
            ),
            retryable=True,
        ).model_dump()

    if order is None:
        return GuidedToolError(
            error_code="ORDER_NOT_FOUND",
            message=f"No shipment order exists for verified format ID {validated.order_id!r}.",
            recovery_instructions=(
                "Inform the customer that order ID "
                f"{validated.order_id} was not found in SwiftShip's active ledger "
                "and ask them to double-check the 5 digits on their confirmation email."
            ),
            retryable=False,
        ).model_dump()

    return ShipmentOrderResponse(order=order).model_dump()
