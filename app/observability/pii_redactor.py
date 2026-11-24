"""Active PII scrubbing service for logging and long-term memory pipelines.

Integrates Google Cloud Sensitive Data Protection (`google-cloud-dlp`) with an
always-on deterministic regex fallback so sensitive customer identifiers
(emails, phone numbers, payment cards, SSNs, street addresses) are scrubbed
before storage in Cloud Logging or Vertex AI Memory Bank.
"""

from __future__ import annotations

import os
import re
from typing import Any

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"
)
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_STREET_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\s]{2,35}\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b",
    re.IGNORECASE,
)
# Preserve SwiftShip business identifiers such as ORD-10042 or TRK-88492011
_ORDER_OR_TRACKING_RE = re.compile(r"\b(?:ORD|TRK|POL|SLA)-[A-Z0-9-]+\b")

_DLP_INFO_TYPES = [
    {"name": "EMAIL_ADDRESS"},
    {"name": "PHONE_NUMBER"},
    {"name": "CREDIT_CARD_NUMBER"},
    {"name": "US_SOCIAL_SECURITY_NUMBER"},
    {"name": "STREET_ADDRESS"},
]


class PiiRedactionService:
    """Scrubs PII from text and structured payloads using Cloud DLP + regex rules."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        enable_cloud_dlp: bool | None = None,
        dlp_client: Any | None = None,
    ) -> None:
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if enable_cloud_dlp is None:
            enable_cloud_dlp = os.environ.get("ENABLE_CLOUD_DLP", "false").lower() in (
                "1",
                "true",
                "yes",
            )
        self._enable_cloud_dlp = bool(enable_cloud_dlp and self._project_id)
        self._dlp_client = dlp_client

    def _get_dlp_client(self) -> Any | None:
        if not self._enable_cloud_dlp:
            return None
        if self._dlp_client is None:
            from google.cloud import dlp_v2

            self._dlp_client = dlp_v2.DlpServiceClient()
        return self._dlp_client

    @staticmethod
    def _scrub_with_regex(text: str) -> str:
        """Fast deterministic PII scrubbing that preserves business order IDs."""
        if not text:
            return text
        protected_tokens: dict[str, str] = {}

        def _protect(match: re.Match[str]) -> str:
            placeholder = f"__SWIFTSHIP_ID_{len(protected_tokens)}__"
            protected_tokens[placeholder] = match.group(0)
            return placeholder

        working = _ORDER_OR_TRACKING_RE.sub(_protect, text)
        working = _EMAIL_RE.sub("[REDACTED_EMAIL]", working)
        working = _SSN_RE.sub("[REDACTED_SSN]", working)
        working = _CREDIT_CARD_RE.sub("[REDACTED_CARD]", working)
        working = _PHONE_RE.sub("[REDACTED_PHONE]", working)
        working = _STREET_ADDRESS_RE.sub("[REDACTED_ADDRESS]", working)

        for placeholder, original in protected_tokens.items():
            working = working.replace(placeholder, original)
        return working

    def scrub_text(self, text: str) -> str:
        """Scrub sensitive data from a string via Cloud DLP (if enabled) and regex."""
        if not text:
            return text
        scrubbed = self._scrub_with_regex(text)
        client = self._get_dlp_client()
        if client is None:
            return scrubbed

        try:
            parent = f"projects/{self._project_id}/locations/global"
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            }
            inspect_config = {"info_types": _DLP_INFO_TYPES}
            response = client.deidentify_content(
                request={
                    "parent": parent,
                    "inspect_config": inspect_config,
                    "deidentify_config": deidentify_config,
                    "item": {"value": scrubbed},
                }
            )
            if response and response.item and response.item.value:
                return str(response.item.value)
        except Exception:
            # Fail safe: deterministic regex scrubbing has already been applied.
            return scrubbed
        return scrubbed

    def scrub_data(self, payload: Any) -> Any:
        """Recursively scrub PII across strings, dicts, lists, and tuples."""
        if isinstance(payload, str):
            return self.scrub_text(payload)
        if isinstance(payload, dict):
            return {
                str(k): (
                    "[REDACTED_SECRET]"
                    if any(
                        secret_word in str(k).lower()
                        for secret_word in (
                            "secret",
                            "api_key",
                            "apikey",
                            "password",
                            "token",
                            "authorization",
                        )
                    )
                    else self.scrub_data(v)
                )
                for k, v in payload.items()
            }
        if isinstance(payload, list):
            return [self.scrub_data(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self.scrub_data(item) for item in payload)
        return payload


_DEFAULT_REDACTOR: PiiRedactionService | None = None


def get_pii_redactor() -> PiiRedactionService:
    """Return process-wide singleton PII redactor."""
    global _DEFAULT_REDACTOR
    if _DEFAULT_REDACTOR is None:
        _DEFAULT_REDACTOR = PiiRedactionService()
    return _DEFAULT_REDACTOR


def set_pii_redactor(redactor: PiiRedactionService | None) -> None:
    """Override the singleton PII redactor (used in unit tests)."""
    global _DEFAULT_REDACTOR
    _DEFAULT_REDACTOR = redactor
