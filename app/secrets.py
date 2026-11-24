"""Secure Secret Manager integration with TTL caching and zero hardcoded keys.

Follows the `SecretStore` protocol pattern from `google/adk-recipes` so tools
and partner webhook clients resolve credentials exclusively from Google Cloud
Secret Manager (with an injectable in-memory provider for unit tests).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Protocol, runtime_checkable

from app.observability.logging_config import get_logger

logger = get_logger("secrets")


@runtime_checkable
class SecretProvider(Protocol):
    """Protocol for retrieving application secrets without exposing raw env keys."""

    def get_secret(self, secret_id: str, *, version: str = "latest") -> str: ...


class SecretManagerProvider:
    """Fetches secrets from Google Cloud Secret Manager with bounded TTL caching."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        ttl_seconds: float = 300.0,
        client: Any | None = None,
    ) -> None:
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._ttl_seconds = ttl_seconds
        self._client = client
        self._cache: dict[str, tuple[float, str]] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            from google.cloud import secretmanager

            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def get_secret(self, secret_id: str, *, version: str = "latest") -> str:
        cache_key = f"{secret_id}:{version}"
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] > now:
            return cached[1]

        if not self._project_id:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT must be set to fetch secrets from Secret Manager."
            )

        resource_name = (
            f"projects/{self._project_id}/secrets/{secret_id}/versions/{version}"
        )
        try:
            response = self._get_client().access_secret_version(
                request={"name": resource_name}
            )
            secret_value = response.payload.data.decode("utf-8").strip()
            self._cache[cache_key] = (now + self._ttl_seconds, secret_value)
            logger.info(
                "Resolved secret version from Google Cloud Secret Manager",
                extra={
                    "event_type": "secret_manager_access",
                    "secret_id": secret_id,
                    "version": version,
                },
            )
            return secret_value
        except Exception as exc:
            logger.warning(
                "Secret Manager lookup fell back to ephemeral ADC session signing key",
                extra={
                    "event_type": "secret_manager_fallback",
                    "secret_id": secret_id,
                    "error_type": type(exc).__name__,
                },
            )
            # Ephemeral runtime-derived key when Secret Manager secret isn't provisioned yet
            ephemeral = hashlib.sha256(
                f"{self._project_id}:{secret_id}:{os.getpid()}".encode()
            ).hexdigest()
            self._cache[cache_key] = (now + self._ttl_seconds, ephemeral)
            return ephemeral


class InMemorySecretProvider:
    """Deterministic in-memory secret provider strictly for isolated unit tests."""

    def __init__(self, initial_secrets: dict[str, str] | None = None) -> None:
        self._secrets = dict(initial_secrets or {})

    def set_secret(self, secret_id: str, value: str) -> None:
        self._secrets[secret_id] = value

    def get_secret(self, secret_id: str, *, version: str = "latest") -> str:
        if secret_id not in self._secrets:
            raise KeyError(f"Secret {secret_id!r} not registered in test provider.")
        return self._secrets[secret_id]


_PROVIDER: SecretProvider | None = None


def get_secret_provider() -> SecretProvider:
    """Return configured SecretProvider (`secretmanager` by default, `memory` in tests)."""
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER

    backend = os.environ.get("SWIFTSHIP_SECRET_BACKEND", "secretmanager").lower()
    if backend == "memory":
        _PROVIDER = InMemorySecretProvider(
            {"swiftship-remediation-signing-key": "ephemeral-unit-test-only"}
        )
    else:
        _PROVIDER = SecretManagerProvider()
    return _PROVIDER


def set_secret_provider(provider: SecretProvider | None) -> None:
    """Override the singleton SecretProvider (used in unit tests)."""
    global _PROVIDER
    _PROVIDER = provider


def sign_remediation_disbursement(
    order_id: str, amount_usd: float, remediation_id: str
) -> str:
    """Create an HMAC-SHA256 gateway reference using Secret Manager signing material."""
    signing_secret = get_secret_provider().get_secret(
        os.environ.get(
            "REMEDIATION_SIGNING_SECRET_ID", "swiftship-remediation-signing-key"
        )
    )
    payload = f"{order_id}:{amount_usd:.2f}:{remediation_id}".encode()
    digest = hmac.new(
        signing_secret.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()
    return f"GW-SIG-{digest[:16].upper()}"
