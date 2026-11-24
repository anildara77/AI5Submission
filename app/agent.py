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

"""SwiftShip Operations & Order Resolution Multi-Agent Application.

Wires together:
- Coordinator (`swiftship_ops_agent`, Gemini Flash) + `SequentialAgent`
  (`order_resolution_pipeline`: Investigator -> Gemini Pro Strategist -> Executor)
- Token-based `EventsCompactionConfig` (`SwiftShipEventSummarizer`) + `ContextCacheConfig`
- `ResumabilityConfig(is_resumable=True)` for Human-in-the-Loop confirmation gates
- `PreloadMemoryTool` + non-blocking async Memory Bank consolidation
- Runner-wide plugins: `SecurityAndSelfEvalGuardrailPlugin`,
  `StrategicModelRoutingPlugin`, `IntentOutcomeAuditPlugin`,
  `ReflectAndRetryToolPlugin`, and `BigQueryAgentAnalyticsPlugin`.
"""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types

from app.guardrails import SecurityAndSelfEvalGuardrailPlugin
from app.memory import (
    SwiftShipEventSummarizer,
    schedule_async_memory_consolidation,
)
from app.model_router import (
    StrategicModelRoutingPlugin,
    build_flash_model,
    build_pro_planning_model,
)
from app.observability.audit_plugin import IntentOutcomeAuditPlugin
from app.observability.logging_config import configure_structured_logging, get_logger
from app.observability.tracing import configure_tracing
from app.prompts import (
    COORDINATOR_INSTRUCTION,
    EXECUTOR_INSTRUCTION,
    INVESTIGATOR_INSTRUCTION,
    STRATEGIST_INSTRUCTION,
)
from app.schemas import RemediationStrategyPlan
from app.tools import (
    calculate_sla_breach_compensation,
    execute_high_value_order_remediation,
    fetch_verified_shipment_order,
    search_sla_policy_knowledge_base,
)

configure_structured_logging()
configure_tracing()
logger = get_logger("agent")


async def initialize_session_state(callback_context: CallbackContext) -> None:
    """Initialize required template keys in session state on first turn."""
    state = callback_context.state
    if "case_dossier" not in state:
        state["case_dossier"] = "(not yet investigated)"
    if "remediation_plan" not in state:
        state["remediation_plan"] = "(not yet planned)"
    if "is_user_prompt_safe" not in state:
        state["is_user_prompt_safe"] = True


def create_case_investigator_agent() -> Agent:
    """Step 1 (Flash): Gather verified order telemetry and SLA policy clauses."""
    return Agent(
        name="case_investigator_agent",
        description=(
            "Fetches verified shipment order records and searches the SLA policy "
            "vector knowledge base to build a factual case dossier."
        ),
        model=build_flash_model(),
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        instruction=INVESTIGATOR_INSTRUCTION,
        tools=[
            fetch_verified_shipment_order,
            search_sla_policy_knowledge_base,
        ],
        output_key="case_dossier",
    )


def create_remediation_strategist_agent() -> Agent:
    """Step 2 (Pro): Synthesize multi-clause SLA rules into `RemediationStrategyPlan`."""
    return Agent(
        name="remediation_strategist_agent",
        description=(
            "Uses Gemini Pro reasoning to evaluate complex SLA disputes and "
            "produce a schema-validated RemediationStrategyPlan."
        ),
        model=build_pro_planning_model(),
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        instruction=STRATEGIST_INSTRUCTION,
        output_schema=RemediationStrategyPlan,
        output_key="remediation_plan",
    )


def create_remediation_execution_agent() -> Agent:
    """Step 3 (Flash + HITL): Calculate exact payout and execute or pause for HITL."""
    return Agent(
        name="remediation_execution_agent",
        description=(
            "Computes deterministic SLA compensation and executes approved "
            "remediations, enforcing the $100 Human-in-the-Loop supervisor gate."
        ),
        model=build_flash_model(),
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        instruction=EXECUTOR_INSTRUCTION,
        tools=[
            calculate_sla_breach_compensation,
            execute_high_value_order_remediation,
        ],
        after_agent_callback=schedule_async_memory_consolidation,
    )


def create_order_resolution_pipeline() -> SequentialAgent:
    """Build the 3-stage deterministic SequentialAgent resolution pipeline."""
    return SequentialAgent(
        name="order_resolution_pipeline",
        description=(
            "Handles all shipment order lookups, late-delivery claims, cold-chain "
            "spoilage disputes, refund/credit requests, and reshipment inquiries."
        ),
        sub_agents=[
            create_case_investigator_agent(),
            create_remediation_strategist_agent(),
            create_remediation_execution_agent(),
        ],
    )


def create_root_agent() -> Agent:
    """Build the front-door Operations Triage Coordinator (`swiftship_ops_agent`)."""
    return Agent(
        # Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml
        name="swiftship_ops_agent",
        description=(
            "Front-door SwiftShip Operations Coordinator routing order disputes "
            "to the SequentialAgent resolution pipeline and answering policy queries."
        ),
        model=build_flash_model(),
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        instruction=COORDINATOR_INSTRUCTION,
        tools=[
            search_sla_policy_knowledge_base,
            PreloadMemoryTool(),
        ],
        sub_agents=[create_order_resolution_pipeline()],
        before_agent_callback=initialize_session_state,
        after_agent_callback=schedule_async_memory_consolidation,
    )


def _build_plugins() -> list[BasePlugin]:
    """Assemble runner-wide security, routing, observability, and analytics plugins."""
    plugins: list[BasePlugin] = [
        SecurityAndSelfEvalGuardrailPlugin(),
        StrategicModelRoutingPlugin(),
        IntentOutcomeAuditPlugin(),
        ReflectAndRetryToolPlugin(
            max_retries=2,
            throw_exception_if_retry_exceeded=False,
        ),
    ]

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    enable_bq_analytics = os.environ.get(
        "ENABLE_BQ_AGENT_ANALYTICS", "false"
    ).lower() in ("1", "true") or bool(os.environ.get("BQ_ANALYTICS_GCS_BUCKET"))

    if project_id and project_id != "your-gcp-project-id" and enable_bq_analytics:
        try:
            from google.cloud import bigquery

            dataset_id = os.environ.get(
                "BQ_ANALYTICS_DATASET_ID", "adk_agent_analytics"
            )
            bq_location = os.environ.get("BQ_ANALYTICS_LOCATION", "US")
            bq_client: Any = bigquery.Client(project=project_id)
            bq_client.create_dataset(f"{project_id}.{dataset_id}", exists_ok=True)
            plugins.append(
                BigQueryAgentAnalyticsPlugin(
                    project_id=project_id,
                    dataset_id=dataset_id,
                    location=bq_location,
                    config=BigQueryLoggerConfig(
                        gcs_bucket_name=os.environ.get("BQ_ANALYTICS_GCS_BUCKET"),
                        connection_id=os.environ.get("BQ_ANALYTICS_CONNECTION_ID"),
                    ),
                )
            )
        except Exception as exc:
            logger.warning(
                "BigQuery Agent Analytics plugin skipped during initialization",
                extra={
                    "event_type": "bq_analytics_init_warning",
                    "error_type": type(exc).__name__,
                },
            )

    return plugins


root_agent = create_root_agent()

app = App(
    name="app",
    root_agent=root_agent,
    plugins=_build_plugins(),
    events_compaction_config=EventsCompactionConfig(
        token_threshold=24000,
        event_retention_size=6,
        compaction_interval=8,
        overlap_size=2,
        summarizer=SwiftShipEventSummarizer(llm=build_flash_model()),
    ),
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=1800,
        cache_intervals=10,
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
)
