"""OpenTelemetry distributed tracing setup for SwiftShip Operations Agent.

Links spans end-to-end from user query ingress through coordinator triage,
specialist sub-agents, LLM calls, PII scrubbing, and tool executions, exporting
to Google Cloud Trace in cloud environments and capturing spans in-memory for
deterministic testing and local diagnostics.
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode, Tracer

_IN_MEMORY_EXPORTER = InMemorySpanExporter()
_TRACING_INITIALIZED = False


def configure_tracing(
    *,
    service_name: str | None = None,
    export_to_cloud: bool | None = None,
) -> TracerProvider:
    """Initialize OpenTelemetry TracerProvider with in-memory + Cloud Trace export."""
    global _TRACING_INITIALIZED
    current_provider = trace.get_tracer_provider()
    if _TRACING_INITIALIZED and isinstance(current_provider, TracerProvider):
        return current_provider

    resolved_service = service_name or os.environ.get(
        "OTEL_SERVICE_NAME", "swiftship-ops-agent"
    )
    resource = Resource.create(
        {
            "service.name": resolved_service,
            "service.version": "0.1.0",
            "cloud.provider": "gcp",
        }
    )

    if isinstance(current_provider, TracerProvider):
        provider = current_provider
    else:
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

    provider.add_span_processor(SimpleSpanProcessor(_IN_MEMORY_EXPORTER))

    if export_to_cloud is None:
        export_to_cloud = os.environ.get(
            "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", ""
        ).lower() in ("1", "true") or os.environ.get(
            "ENABLE_CLOUD_TRACE", ""
        ).lower() in ("1", "true")

    if export_to_cloud and os.environ.get("GOOGLE_CLOUD_PROJECT"):
        try:
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

            cloud_exporter = CloudTraceSpanExporter(
                project_id=os.environ.get("GOOGLE_CLOUD_PROJECT")
            )
            provider.add_span_processor(BatchSpanProcessor(cloud_exporter))
        except Exception:
            # Degrade gracefully when running offline or in unit tests.
            pass

    _TRACING_INITIALIZED = True
    return provider


def get_tracer(instrumenting_module_name: str = "swiftship.agentops") -> Tracer:
    """Return an OpenTelemetry Tracer backed by the configured provider."""
    configure_tracing()
    return trace.get_tracer(instrumenting_module_name)


def get_in_memory_spans() -> tuple[ReadableSpan, ...]:
    """Return recorded spans from the in-memory exporter for verification/tests."""
    return _IN_MEMORY_EXPORTER.get_finished_spans()


def clear_in_memory_spans() -> None:
    """Clear finished spans from the in-memory exporter."""
    _IN_MEMORY_EXPORTER.clear()


def traced_operation(
    span_name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator creating a linked OpenTelemetry child span around sync/async calls."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    if attributes:
                        for k, v in attributes.items():
                            span.set_attribute(k, v)
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        return sync_wrapper

    return decorator
