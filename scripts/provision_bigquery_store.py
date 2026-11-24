"""Idempotent bootstrap script to provision & seed BigQuery tables and embeddings.

Creates `swiftship_ops.orders`, `swiftship_ops.sla_policy_chunks` (populated with
live Vertex AI `text-embedding-005` vectors), and `swiftship_ops.remediation_ledger`.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from app.persistence import BigQueryLogisticsStore


def main() -> int:
    load_dotenv()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project_id or project_id == "your-gcp-project-id":
        print(
            "ERROR: Set GOOGLE_CLOUD_PROJECT before running provisioning.",
            file=sys.stderr,
        )
        return 1

    store = BigQueryLogisticsStore(project_id=project_id)
    store.ensure_provisioned_and_seeded()

    sample = store.get_order("ORD-10042")
    clauses = store.search_policies(
        "cold-chain biologics thermal excursion refund", top_k=2
    )
    print(f"BigQuery dataset provisioned in project={project_id}")
    print(f"Verified sample order: {sample.order_id if sample else 'NONE'}")
    print(
        f"Verified top vector clause: {clauses[0].policy_clause_id if clauses else 'NONE'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
