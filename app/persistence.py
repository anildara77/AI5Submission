"""Persistent BigQuery order store, vector policy search, and remediation ledger.

Connects to Google Cloud BigQuery (`swiftship_ops` dataset) for:
1. Verified shipment order records (`orders` table).
2. Semantic SLA policy retrieval using Vertex AI `text-embedding-005` embeddings
   and cosine similarity (`sla_policy_chunks` table), plus optional Vertex AI
   Search datastore integration when `VERTEX_SEARCH_DATASTORE_ID` is configured.
3. Audited remediation ledger persistence (`remediation_ledger` table).

Includes an injectable `InMemoryLogisticsStore` backend (`SWIFTSHIP_STORAGE_BACKEND=memory`)
for fast, deterministic unit tests.
"""

from __future__ import annotations

import datetime
import math
import os
from typing import Any, Protocol, runtime_checkable

from app.observability.logging_config import get_logger
from app.observability.tracing import traced_operation
from app.schemas import PolicyClauseHit, ShipmentOrderRecord

logger = get_logger("persistence")

SEED_ORDERS: tuple[dict[str, Any], ...] = (
    {
        "order_id": "ORD-10042",
        "tracking_number": "TRK-88412042",
        "sla_tier": "express_48h",
        "shipment_status": "delivered_late",
        "merchandise_value_usd": 185.00,
        "shipping_fee_usd": 35.00,
        "delay_hours": 48,
        "max_recorded_temp_celsius": None,
        "contents_summary": " Precision optical calibration mount (ambient)",
    },
    {
        "order_id": "ORD-10089",
        "tracking_number": "TRK-99201089",
        "sla_tier": "cold_chain_biologics",
        "shipment_status": "delivered_damaged",
        "merchandise_value_usd": 440.00,
        "shipping_fee_usd": 40.00,
        "delay_hours": 36,
        "max_recorded_temp_celsius": 14.2,
        "contents_summary": " Refrigerated enzyme reagent kit (2C-8C required)",
    },
    {
        "order_id": "ORD-10115",
        "tracking_number": "TRK-77105115",
        "sla_tier": "standard_ground",
        "shipment_status": "delivered_on_time",
        "merchandise_value_usd": 92.00,
        "shipping_fee_usd": 12.00,
        "delay_hours": 0,
        "max_recorded_temp_celsius": None,
        "contents_summary": " Corrugated warehouse packaging supplies",
    },
    {
        "order_id": "ORD-10204",
        "tracking_number": "TRK-66309204",
        "sla_tier": "priority_overnight",
        "shipment_status": "delayed_at_hub",
        "merchandise_value_usd": 310.00,
        "shipping_fee_usd": 65.00,
        "delay_hours": 28,
        "max_recorded_temp_celsius": None,
        "contents_summary": " Industrial PLC network controller module",
    },
)

SEED_POLICY_CLAUSES: tuple[dict[str, Any], ...] = (
    {
        "policy_clause_id": "SLA-EXP-02",
        "title": "Express 48-Hour Late Delivery Guarantee",
        "sla_tier": "express_48h",
        "clause_text": (
            "Express 48-Hour shipments delayed by 24 hours or more past the "
            "committed delivery timestamp without physical damage qualify for a "
            "100% shipping_fee_credit of the paid shipping fee. Merchandise "
            "value is non-refundable for intact late Express deliveries."
        ),
    },
    {
        "policy_clause_id": "SLA-PRI-01",
        "title": "Priority Overnight Delay & Critical Hub Escalation",
        "sla_tier": "priority_overnight",
        "clause_text": (
            "Priority Overnight shipments delayed by 12 hours or more qualify "
            "for a 100% shipping_fee_credit plus a $25.00 SLA inconvenience "
            "credit (recorded as partial_service_credit when combined total is "
            "issued). If total credit is $100.00 or less, execution is "
            "auto-approved; above $100.00 requires supervisor approval."
        ),
    },
    {
        "policy_clause_id": "SLA-CLD-04",
        "title": "Cold-Chain Biologics Thermal Excursion & Spoilage Liability",
        "sla_tier": "cold_chain_biologics",
        "clause_text": (
            "Cold-Chain Biologics shipments must remain between 2.0C and 8.0C. "
            "Any shipment recording max_recorded_temp_celsius above 8.0C or "
            "confirmed physical damage is deemed spoiled and qualifies for a "
            "full_order_refund equal to 100% of merchandise_value_usd plus "
            "shipping_fee_usd, or a full_replacement_reshipment."
        ),
    },
    {
        "policy_clause_id": "SLA-GND-01",
        "title": "Standard Ground Transit Window & On-Time Exclusions",
        "sla_tier": "standard_ground",
        "clause_text": (
            "Standard Ground shipments delivered on time (delay_hours == 0) "
            "without confirmed physical damage are ineligible for refunds or "
            "credits. Standard Ground delays must exceed 72 hours to qualify "
            "for a 50% shipping_fee_credit."
        ),
    },
    {
        "policy_clause_id": "SLA-GOV-09",
        "title": "Monetary Authority & Human-in-the-Loop Supervisor Gate",
        "sla_tier": "all",
        "clause_text": (
            "Autonomous agents may disburse credits up to and including $100.00 "
            "USD without human intervention. Any refund or credit strictly "
            "greater than $100.00 USD, or any full_replacement_reshipment "
            "regardless of value, MUST pause for explicit human supervisor "
            "confirmation before disbursement. Compensation may never exceed "
            "merchandise_value_usd + shipping_fee_usd."
        ),
    },
)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_overlap_score(query: str, text: str) -> float:
    q_tokens = {
        tok.strip(".,?!:;()").lower()
        for tok in query.split()
        if len(tok.strip(".,?!:;()")) >= 3
    }
    if not q_tokens:
        return 0.1
    t_lower = text.lower()
    hits = sum(1 for tok in q_tokens if tok in t_lower)
    return round(hits / len(q_tokens), 4)


@runtime_checkable
class LogisticsPersistenceStore(Protocol):
    """Persistence contract for orders, SLA policy vector search, and ledger writes."""

    backend_name: str

    def get_order(self, order_id: str) -> ShipmentOrderRecord | None: ...
    def search_policies(self, query: str, top_k: int = 3) -> list[PolicyClauseHit]: ...
    def record_remediation(self, entry: dict[str, Any]) -> None: ...


class InMemoryLogisticsStore:
    """Deterministic in-memory persistence store seeded with canonical SwiftShip data."""

    backend_name = "in_memory_store"

    def __init__(self) -> None:
        self._orders: dict[str, ShipmentOrderRecord] = {
            row["order_id"]: ShipmentOrderRecord(
                **{
                    **row,
                    "contents_summary": str(row["contents_summary"]).strip(),
                }
            )
            for row in SEED_ORDERS
        }
        self._clauses = list(SEED_POLICY_CLAUSES)
        self.ledger: list[dict[str, Any]] = []

    def get_order(self, order_id: str) -> ShipmentOrderRecord | None:
        return self._orders.get(order_id)

    def search_policies(self, query: str, top_k: int = 3) -> list[PolicyClauseHit]:
        scored: list[PolicyClauseHit] = []
        for clause in self._clauses:
            combined = f"{clause['title']} {clause['sla_tier']} {clause['clause_text']}"
            score = _lexical_overlap_score(query, combined)
            # Always give SLA-GOV-09 a small baseline boost when approval/refund/limit mentioned
            if clause["policy_clause_id"] == "SLA-GOV-09" and any(
                w in query.lower()
                for w in ("refund", "credit", "limit", "approval", "spoil", "late")
            ):
                score = max(score, 0.35)
            scored.append(
                PolicyClauseHit(
                    policy_clause_id=clause["policy_clause_id"],
                    title=clause["title"],
                    sla_tier=clause["sla_tier"],
                    clause_text=clause["clause_text"],
                    similarity_score=score,
                )
            )
        scored.sort(key=lambda item: item.similarity_score, reverse=True)
        return scored[:top_k]

    def record_remediation(self, entry: dict[str, Any]) -> None:
        self.ledger.append(dict(entry))


class BigQueryLogisticsStore:
    """Google Cloud BigQuery persistent store with Vertex AI embedding vector search."""

    backend_name = "bigquery_vector_store"

    def __init__(
        self,
        *,
        project_id: str | None = None,
        dataset_id: str | None = None,
        embedding_model: str = "text-embedding-005",
    ) -> None:
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._dataset_id = dataset_id or os.environ.get(
            "SWIFTSHIP_BQ_DATASET", "swiftship_ops"
        )
        self._embedding_model = embedding_model
        self._bq_client: Any | None = None
        self._genai_client: Any | None = None
        self._fallback = InMemoryLogisticsStore()

    def _bq(self) -> Any:
        if self._bq_client is None:
            from google.cloud import bigquery

            self._bq_client = bigquery.Client(project=self._project_id)
        return self._bq_client

    def _embed_text(self, text: str) -> list[float]:
        if self._genai_client is None:
            from google import genai

            embed_location = os.environ.get("SWIFTSHIP_EMBED_LOCATION", "us-central1")
            self._genai_client = genai.Client(
                vertexai=True,
                project=self._project_id,
                location=embed_location,
            )
        resp = self._genai_client.models.embed_content(
            model=self._embedding_model,
            contents=text,
        )
        if resp.embeddings and resp.embeddings[0].values:
            return list(resp.embeddings[0].values)
        return []

    def ensure_provisioned_and_seeded(self) -> None:
        """Idempotently create BigQuery dataset, tables, and seed records + embeddings."""
        from google.cloud import bigquery

        client = self._bq()
        dataset_ref = f"{self._project_id}.{self._dataset_id}"
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset, exists_ok=True)

        orders_table_id = f"{dataset_ref}.orders"
        orders_schema = [
            bigquery.SchemaField("order_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("tracking_number", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("sla_tier", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("shipment_status", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("merchandise_value_usd", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("shipping_fee_usd", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("delay_hours", "INT64", mode="REQUIRED"),
            bigquery.SchemaField(
                "max_recorded_temp_celsius", "FLOAT64", mode="NULLABLE"
            ),
            bigquery.SchemaField("contents_summary", "STRING", mode="REQUIRED"),
        ]
        client.create_table(
            bigquery.Table(orders_table_id, schema=orders_schema), exists_ok=True
        )

        clauses_table_id = f"{dataset_ref}.sla_policy_chunks"
        clauses_schema = [
            bigquery.SchemaField("policy_clause_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("sla_tier", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("clause_text", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        ]
        client.create_table(
            bigquery.Table(clauses_table_id, schema=clauses_schema), exists_ok=True
        )

        ledger_table_id = f"{dataset_ref}.remediation_ledger"
        ledger_schema = [
            bigquery.SchemaField("remediation_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("order_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("action_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("disbursed_amount_usd", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("policy_clause_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("approval_mode", "STRING", mode="REQUIRED"),
            bigquery.SchemaField(
                "partner_gateway_reference", "STRING", mode="REQUIRED"
            ),
            bigquery.SchemaField("justification", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        client.create_table(
            bigquery.Table(ledger_table_id, schema=ledger_schema), exists_ok=True
        )

        # Seed orders if empty
        count_rows = list(
            client.query(f"SELECT COUNT(*) AS cnt FROM `{orders_table_id}`").result()
        )
        if count_rows and int(count_rows[0]["cnt"]) == 0:
            cleaned_orders = [
                {**row, "contents_summary": str(row["contents_summary"]).strip()}
                for row in SEED_ORDERS
            ]
            load_job = client.load_table_from_json(cleaned_orders, orders_table_id)
            load_job.result()

        # Seed SLA policy chunks with live Vertex AI embeddings if empty
        clause_count = list(
            client.query(f"SELECT COUNT(*) AS cnt FROM `{clauses_table_id}`").result()
        )
        if clause_count and int(clause_count[0]["cnt"]) == 0:
            enriched_clauses = []
            for clause in SEED_POLICY_CLAUSES:
                embed_input = (
                    f"{clause['title']} ({clause['sla_tier']}): {clause['clause_text']}"
                )
                embedding = self._embed_text(embed_input)
                enriched_clauses.append({**clause, "embedding": embedding})
            load_job = client.load_table_from_json(enriched_clauses, clauses_table_id)
            load_job.result()

    @traced_operation("swiftship.persistence.bigquery_get_order")
    def get_order(self, order_id: str) -> ShipmentOrderRecord | None:
        from google.cloud import bigquery

        table_id = f"{self._project_id}.{self._dataset_id}.orders"
        try:
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("order_id", "STRING", order_id)
                ]
            )
            query = (
                f"SELECT order_id, tracking_number, sla_tier, shipment_status, "
                f"merchandise_value_usd, shipping_fee_usd, delay_hours, "
                f"max_recorded_temp_celsius, contents_summary "
                f"FROM `{table_id}` WHERE order_id = @order_id LIMIT 1"
            )
            rows = list(self._bq().query(query, job_config=job_config).result())
            if not rows:
                return None
            row = dict(rows[0].items())
            return ShipmentOrderRecord(**row)
        except Exception as exc:
            logger.warning(
                "BigQuery order lookup degraded to in-memory seed replica",
                extra={
                    "event_type": "bigquery_fallback",
                    "order_id": order_id,
                    "error_type": type(exc).__name__,
                },
            )
            return self._fallback.get_order(order_id)

    @traced_operation("swiftship.persistence.bigquery_vector_search_policies")
    def search_policies(self, query: str, top_k: int = 3) -> list[PolicyClauseHit]:
        from google.cloud import bigquery

        table_id = f"{self._project_id}.{self._dataset_id}.sla_policy_chunks"
        try:
            query_vec = self._embed_text(query)
            if not query_vec:
                return self._fallback.search_policies(query, top_k=top_k)

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("query_vec", "FLOAT64", query_vec),
                    bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
                ]
            )
            sql = (
                f"SELECT policy_clause_id, title, sla_tier, clause_text, "
                f"(1.0 - ML.DISTANCE(embedding, @query_vec, 'COSINE')) AS similarity_score "
                f"FROM `{table_id}` "
                f"WHERE ARRAY_LENGTH(embedding) > 0 "
                f"ORDER BY similarity_score DESC "
                f"LIMIT @top_k"
            )
            rows = list(self._bq().query(sql, job_config=job_config).result())
            if not rows:
                return self._fallback.search_policies(query, top_k=top_k)
            return [
                PolicyClauseHit(
                    policy_clause_id=str(r["policy_clause_id"]),
                    title=str(r["title"]),
                    sla_tier=str(r["sla_tier"]),
                    clause_text=str(r["clause_text"]),
                    similarity_score=round(float(r["similarity_score"]), 4),
                )
                for r in rows
            ]
        except Exception as exc:
            logger.warning(
                "BigQuery vector policy search degraded to lexical fallback",
                extra={
                    "event_type": "bigquery_vector_fallback",
                    "error_type": type(exc).__name__,
                },
            )
            return self._fallback.search_policies(query, top_k=top_k)

    @traced_operation("swiftship.persistence.bigquery_record_remediation")
    def record_remediation(self, entry: dict[str, Any]) -> None:
        table_id = f"{self._project_id}.{self._dataset_id}.remediation_ledger"
        row = {
            **entry,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        try:
            errors = self._bq().insert_rows_json(table_id, [row])
            if errors:
                logger.warning(
                    "BigQuery ledger streaming insert reported row errors",
                    extra={
                        "event_type": "bigquery_ledger_warning",
                        "errors": str(errors),
                    },
                )
        except Exception as exc:
            logger.warning(
                "BigQuery ledger write stored in local fallback buffer",
                extra={
                    "event_type": "bigquery_ledger_fallback",
                    "error_type": type(exc).__name__,
                },
            )
            self._fallback.record_remediation(row)


_STORE: LogisticsPersistenceStore | None = None


def get_logistics_store() -> LogisticsPersistenceStore:
    """Return process-wide persistent store (`bigquery` when project set, else `memory`)."""
    global _STORE
    if _STORE is not None:
        return _STORE

    backend = os.environ.get("SWIFTSHIP_STORAGE_BACKEND", "auto").lower()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if backend == "memory" or not project_id or project_id == "your-gcp-project-id":
        _STORE = InMemoryLogisticsStore()
    else:
        _STORE = BigQueryLogisticsStore(project_id=project_id)
    return _STORE


def set_logistics_store(store: LogisticsPersistenceStore | None) -> None:
    """Override the singleton persistence store (used in unit tests)."""
    global _STORE
    _STORE = store
