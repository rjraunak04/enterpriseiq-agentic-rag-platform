"""Environment-backed runtime configuration for production deployment."""

from __future__ import annotations

import os

from enterpriseiq.api.config import APIConfig, APIConfigError


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise APIConfigError(f"{name} must be a boolean value.")


def config_from_env() -> APIConfig:
    """Build immutable API configuration from deployment environment variables."""

    raw_max_length = os.getenv("ENTERPRISEIQ_MAX_QUERY_LENGTH", "4000")
    try:
        max_query_length = int(raw_max_length)
    except ValueError as exc:
        raise APIConfigError(
            "ENTERPRISEIQ_MAX_QUERY_LENGTH must be an integer."
        ) from exc

    return APIConfig(
        service_name=os.getenv("ENTERPRISEIQ_SERVICE_NAME", "EnterpriseIQ"),
        service_version=os.getenv("ENTERPRISEIQ_SERVICE_VERSION", "0.7.0"),
        environment=os.getenv("ENTERPRISEIQ_ENVIRONMENT", "production"),
        api_prefix=os.getenv("ENTERPRISEIQ_API_PREFIX", "/v1"),
        request_id_header=os.getenv("ENTERPRISEIQ_REQUEST_ID_HEADER", "X-Request-ID"),
        max_query_length=max_query_length,
        expose_docs=_bool_env("ENTERPRISEIQ_EXPOSE_DOCS", False),
    )
