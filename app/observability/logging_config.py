"""Structured JSON logging configuration for Cloud Logging and AgentOps auditing.

Replaces unstructured print statements with rich, trace-correlated JSON records
compatible with Google Cloud Logging (`StructuredLogHandler` / `JsonFormatter`).
"""

from __future__ import annotations

import contextvars
import datetime
import logging
import os
import sys
from typing import Any

from pythonjsonlogger.json import JsonFormatter

from app.observability.pii_redactor import get_pii_redactor

_TRACE_CONTEXT: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "swiftship_trace_context",
    default=None,
)
_CONFIGURED = False


def set_log_trace_context(
    *,
    trace_id: str = "",
    span_id: str = "",
    session_id: str = "",
    invocation_id: str = "",
    project_id: str = "",
) -> contextvars.Token[dict[str, str] | None]:
    """Bind trace and session identifiers to the current async execution context."""
    formatted_trace = (
        f"projects/{project_id}/traces/{trace_id}"
        if project_id and trace_id and not trace_id.startswith("projects/")
        else trace_id
    )
    payload = {
        "trace_id": formatted_trace,
        "span_id": span_id,
        "session_id": session_id,
        "invocation_id": invocation_id,
    }
    return _TRACE_CONTEXT.set({k: v for k, v in payload.items() if v})


def reset_log_trace_context(
    token: contextvars.Token[dict[str, str] | None],
) -> None:
    """Restore previous trace context."""
    _TRACE_CONTEXT.reset(token)


class CloudStructuredJsonFormatter(JsonFormatter):
    """JSON log formatter that injects Cloud Trace correlation and scrubs PII."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
        log_record["severity"] = record.levelname
        log_record["logger"] = record.name
        log_record["service"] = os.environ.get(
            "OTEL_SERVICE_NAME", "swiftship-ops-agent"
        )

        ctx = _TRACE_CONTEXT.get() or {}
        if trace_val := ctx.get("trace_id"):
            log_record["logging.googleapis.com/trace"] = trace_val
        if span_val := ctx.get("span_id"):
            log_record["logging.googleapis.com/spanId"] = span_val
        for key in ("session_id", "invocation_id"):
            if key not in log_record and key in ctx:
                log_record[key] = ctx[key]

        redactor = get_pii_redactor()
        for key, value in list(log_record.items()):
            if key in (
                "timestamp",
                "severity",
                "logger",
                "service",
                "logging.googleapis.com/trace",
                "logging.googleapis.com/spanId",
            ):
                continue
            log_record[key] = redactor.scrub_data(value)


def configure_structured_logging(level: int | str | None = None) -> logging.Logger:
    """Configure the application logger hierarchy to emit structured JSON."""
    global _CONFIGURED
    resolved_level = level or os.environ.get("LOG_LEVEL", "INFO").upper()
    app_logger = logging.getLogger("swiftship")
    app_logger.setLevel(resolved_level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = CloudStructuredJsonFormatter(
            "%(timestamp)s %(severity)s %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
        app_logger.handlers.clear()
        app_logger.addHandler(handler)
        app_logger.propagate = False
        _CONFIGURED = True

    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the structured `swiftship` namespace."""
    configure_structured_logging()
    clean_name = name if name.startswith("swiftship") else f"swiftship.{name}"
    return logging.getLogger(clean_name)
