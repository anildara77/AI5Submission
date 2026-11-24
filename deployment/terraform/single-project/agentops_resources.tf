# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Persistent BigQuery dataset & tables for orders, SLA vector chunks, and ledger
resource "google_bigquery_dataset" "swiftship_ops" {
  dataset_id  = "swiftship_ops"
  project     = var.project_id
  location    = "US"
  description = "Persistent order records, Vertex AI vector SLA policy chunks, and remediation audit ledger"

  depends_on = [google_project_service.services]
}

resource "google_bigquery_table" "orders" {
  dataset_id          = google_bigquery_dataset.swiftship_ops.dataset_id
  table_id            = "orders"
  project             = var.project_id
  deletion_protection = false

  schema = jsonencode([
    { name = "order_id", type = "STRING", mode = "REQUIRED" },
    { name = "tracking_number", type = "STRING", mode = "REQUIRED" },
    { name = "sla_tier", type = "STRING", mode = "REQUIRED" },
    { name = "shipment_status", type = "STRING", mode = "REQUIRED" },
    { name = "merchandise_value_usd", type = "FLOAT64", mode = "REQUIRED" },
    { name = "shipping_fee_usd", type = "FLOAT64", mode = "REQUIRED" },
    { name = "delay_hours", type = "INT64", mode = "REQUIRED" },
    { name = "max_recorded_temp_celsius", type = "FLOAT64", mode = "NULLABLE" },
    { name = "contents_summary", type = "STRING", mode = "REQUIRED" },
  ])
}

resource "google_bigquery_table" "sla_policy_chunks" {
  dataset_id          = google_bigquery_dataset.swiftship_ops.dataset_id
  table_id            = "sla_policy_chunks"
  project             = var.project_id
  deletion_protection = false

  schema = jsonencode([
    { name = "policy_clause_id", type = "STRING", mode = "REQUIRED" },
    { name = "title", type = "STRING", mode = "REQUIRED" },
    { name = "sla_tier", type = "STRING", mode = "REQUIRED" },
    { name = "clause_text", type = "STRING", mode = "REQUIRED" },
    { name = "embedding", type = "FLOAT64", mode = "REPEATED" },
  ])
}

resource "google_bigquery_table" "remediation_ledger" {
  dataset_id          = google_bigquery_dataset.swiftship_ops.dataset_id
  table_id            = "remediation_ledger"
  project             = var.project_id
  deletion_protection = false

  schema = jsonencode([
    { name = "remediation_id", type = "STRING", mode = "REQUIRED" },
    { name = "order_id", type = "STRING", mode = "REQUIRED" },
    { name = "action_type", type = "STRING", mode = "REQUIRED" },
    { name = "disbursed_amount_usd", type = "FLOAT64", mode = "REQUIRED" },
    { name = "policy_clause_id", type = "STRING", mode = "REQUIRED" },
    { name = "approval_mode", type = "STRING", mode = "REQUIRED" },
    { name = "partner_gateway_reference", type = "STRING", mode = "REQUIRED" },
    { name = "justification", type = "STRING", mode = "REQUIRED" },
    { name = "created_at", type = "TIMESTAMP", mode = "REQUIRED" },
  ])
}

# Google Cloud Secret Manager secret for HMAC remediation disbursement signing
resource "google_secret_manager_secret" "remediation_signing_key" {
  secret_id = "swiftship-remediation-signing-key"
  project   = var.project_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

# Least-privilege Agent Identity IAM bindings for Secret Manager, DLP, and BigQuery
resource "google_secret_manager_secret_iam_member" "agent_secret_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.remediation_signing_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "principal://${google_vertex_ai_reasoning_engine.app.spec[0].effective_identity}"
}

resource "google_project_iam_member" "agent_dlp_user" {
  project = var.project_id
  role    = "roles/dlp.user"
  member  = "principal://${google_vertex_ai_reasoning_engine.app.spec[0].effective_identity}"
}

resource "google_bigquery_dataset_iam_member" "agent_swiftship_ops_editor" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.swiftship_ops.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "principal://${google_vertex_ai_reasoning_engine.app.spec[0].effective_identity}"
}
