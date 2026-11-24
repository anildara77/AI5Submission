"""ADK Plugin capturing Intent vs. Outcome audit events and OpenTelemetry spans.

Explicitly records:
1. `agent_action_intent` BEFORE every tool call (planned tool, scrubbed arguments,
   agent author, session/invocation IDs, correlation ID).
2. `agent_action_outcome` AFTER every tool call or tool failure (actual status,
   scrubbed response summary, latency in ms, matching correlation ID).
3. Linked OpenTelemetry parent/child spans across request -> agent -> LLM -> tool.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.observability.logging_config import (
    get_logger,
    reset_log_trace_context,
    set_log_trace_context,
)
from app.observability.pii_redactor import get_pii_redactor
from app.observability.tracing import get_tracer

logger = get_logger("audit")


class IntentOutcomeAuditPlugin(BasePlugin):
    """Runner-wide ADK plugin for Intent-vs-Outcome logging and distributed tracing."""

    def __init__(self) -> None:
        super().__init__(name="intent_outcome_audit_plugin")
        self._run_spans: dict[str, tuple[Any, Any, Any]] = {}
        self._agent_spans: dict[str, tuple[Any, Any]] = {}
        self._llm_spans: dict[str, tuple[Any, Any, float]] = {}
        self._tool_inflight: dict[str, tuple[str, float, Any, Any]] = {}

    @staticmethod
    def _format_ids(span: Any) -> tuple[str, str]:
        span_ctx = span.get_span_context()
        if span_ctx and span_ctx.is_valid:
            return f"{span_ctx.trace_id:032x}", f"{span_ctx.span_id:016x}"
        return "", ""

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        tracer = get_tracer()
        inv_id = invocation_context.invocation_id
        session_id = invocation_context.session.id
        span = tracer.start_span(
            "swiftship.invocation",
            attributes={
                "gen_ai.system": "google_adk",
                "swiftship.session_id": session_id,
                "swiftship.invocation_id": inv_id,
                "swiftship.user_id": invocation_context.user_id,
            },
        )
        otel_token = otel_context.attach(trace.set_span_in_context(span))
        trace_id, span_id = self._format_ids(span)
        log_token = set_log_trace_context(
            trace_id=trace_id,
            span_id=span_id,
            session_id=session_id,
            invocation_id=inv_id,
            project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        )
        self._run_spans[inv_id] = (span, otel_token, log_token)
        logger.info(
            "Invocation started",
            extra={
                "event_type": "invocation_start",
                "session_id": session_id,
                "invocation_id": inv_id,
            },
        )
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        inv_id = invocation_context.invocation_id
        entry = self._run_spans.pop(inv_id, None)
        if entry is None:
            return
        span, otel_token, log_token = entry
        logger.info(
            "Invocation completed",
            extra={
                "event_type": "invocation_end",
                "session_id": invocation_context.session.id,
                "invocation_id": inv_id,
            },
        )
        span.set_status(Status(StatusCode.OK))
        span.end()
        reset_log_trace_context(log_token)
        otel_context.detach(otel_token)

    async def before_agent_callback(
        self, *, agent: Any, callback_context: CallbackContext
    ) -> types.Content | None:
        tracer = get_tracer()
        key = f"{callback_context.invocation_id}:{agent.name}"
        span = tracer.start_span(
            f"swiftship.agent.{agent.name}",
            attributes={
                "gen_ai.agent.name": agent.name,
                "swiftship.invocation_id": callback_context.invocation_id,
            },
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        self._agent_spans[key] = (span, token)
        return None

    async def after_agent_callback(
        self, *, agent: Any, callback_context: CallbackContext
    ) -> types.Content | None:
        key = f"{callback_context.invocation_id}:{agent.name}"
        entry = self._agent_spans.pop(key, None)
        if entry is not None:
            span, token = entry
            span.set_status(Status(StatusCode.OK))
            span.end()
            otel_context.detach(token)
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        tracer = get_tracer()
        key = f"{callback_context.invocation_id}:{callback_context.agent_name}"
        model_name = str(llm_request.model or "default")
        span = tracer.start_span(
            f"swiftship.llm.{model_name}",
            attributes={
                "gen_ai.request.model": model_name,
                "gen_ai.agent.name": callback_context.agent_name,
            },
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        self._llm_spans[key] = (span, token, time.perf_counter())
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        key = f"{callback_context.invocation_id}:{callback_context.agent_name}"
        entry = self._llm_spans.pop(key, None)
        if entry is not None:
            span, token, started = entry
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            span.set_attribute("swiftship.llm.latency_ms", latency_ms)
            span.set_status(Status(StatusCode.OK))
            span.end()
            otel_context.detach(token)
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        redactor = get_pii_redactor()
        scrubbed_args = redactor.scrub_data(tool_args)
        correlation_id = f"intent-{uuid.uuid4().hex[:12]}"
        call_key = (
            f"{tool_context.invocation_id}:{tool.name}:"
            f"{tool_context.function_call_id or correlation_id}"
        )

        tracer = get_tracer()
        span = tracer.start_span(
            f"swiftship.tool.{tool.name}",
            attributes={
                "gen_ai.tool.name": tool.name,
                "gen_ai.agent.name": tool_context.agent_name,
                "swiftship.intent_correlation_id": correlation_id,
            },
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        self._tool_inflight[call_key] = (
            correlation_id,
            time.perf_counter(),
            span,
            token,
        )

        logger.info(
            "Recorded agent tool execution intent",
            extra={
                "event_type": "agent_action_intent",
                "intent_correlation_id": correlation_id,
                "agent_name": tool_context.agent_name,
                "intended_tool": tool.name,
                "intended_arguments": scrubbed_args,
                "session_id": tool_context.session.id,
                "invocation_id": tool_context.invocation_id,
            },
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        redactor = get_pii_redactor()
        scrubbed_result = redactor.scrub_data(result)
        scrubbed_args = redactor.scrub_data(tool_args)
        call_key = (
            f"{tool_context.invocation_id}:{tool.name}:"
            f"{tool_context.function_call_id or ''}"
        )
        inflight = self._tool_inflight.pop(call_key, None)
        if inflight is None and self._tool_inflight:
            # Fallback if function_call_id was unset
            last_key = next(reversed(self._tool_inflight))
            inflight = self._tool_inflight.pop(last_key)

        if inflight is not None:
            correlation_id, started, span, token = inflight
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
        else:
            correlation_id = "intent-unmatched"
            duration_ms = 0.0
            span, token = None, None

        actual_status = (
            str(result.get("status", "success"))
            if isinstance(result, dict)
            else "success"
        )
        intent_succeeded = actual_status != "error"

        if span is not None:
            span.set_attribute("swiftship.tool.actual_status", actual_status)
            span.set_attribute("swiftship.tool.duration_ms", duration_ms)
            span.set_status(
                Status(StatusCode.OK if intent_succeeded else StatusCode.ERROR)
            )
            span.end()
        if token is not None:
            otel_context.detach(token)

        logger.info(
            "Recorded agent tool execution outcome",
            extra={
                "event_type": "agent_action_outcome",
                "intent_correlation_id": correlation_id,
                "agent_name": tool_context.agent_name,
                "intended_tool": tool.name,
                "intended_arguments": scrubbed_args,
                "actual_status": actual_status,
                "intent_succeeded": intent_succeeded,
                "actual_outcome": scrubbed_result,
                "duration_ms": duration_ms,
                "session_id": tool_context.session.id,
                "invocation_id": tool_context.invocation_id,
            },
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        redactor = get_pii_redactor()
        scrubbed_args = redactor.scrub_data(tool_args)
        call_key = (
            f"{tool_context.invocation_id}:{tool.name}:"
            f"{tool_context.function_call_id or ''}"
        )
        inflight = self._tool_inflight.pop(call_key, None)
        if inflight is not None:
            correlation_id, started, span, token = inflight
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()
            otel_context.detach(token)
        else:
            correlation_id = "intent-error"
            duration_ms = 0.0

        logger.error(
            "Recorded agent tool execution exception outcome",
            extra={
                "event_type": "agent_action_outcome",
                "intent_correlation_id": correlation_id,
                "agent_name": tool_context.agent_name,
                "intended_tool": tool.name,
                "intended_arguments": scrubbed_args,
                "actual_status": "exception",
                "intent_succeeded": False,
                "error_type": type(error).__name__,
                "error_message": redactor.scrub_text(str(error)),
                "duration_ms": duration_ms,
                "session_id": tool_context.session.id,
                "invocation_id": tool_context.invocation_id,
            },
        )
        return None
