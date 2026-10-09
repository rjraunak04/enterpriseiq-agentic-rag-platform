"""Configuration contracts for the EnterpriseIQ API service."""

from __future__ import annotations

from dataclasses import dataclass


class APIConfigError(ValueError):
    """Raised when API service configuration is invalid."""


@dataclass(frozen=True, slots=True)
class APIConfig:
    """Immutable configuration for the EnterpriseIQ service."""

    service_name: str = "EnterpriseIQ"
    service_version: str = "0.6.0"
    environment: str = "development"
    api_prefix: str = "/v1"
    request_id_header: str = "X-Request-ID"
    max_query_length: int = 4000
    expose_docs: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "service_name",
            "service_version",
            "environment",
            "api_prefix",
            "request_id_header",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise APIConfigError(
                    f"{field_name} must be a non-empty string."
                )

        if not self.api_prefix.startswith("/"):
            raise APIConfigError(
                "api_prefix must start with '/'."
            )

        if (
            isinstance(self.max_query_length, bool)
            or not isinstance(self.max_query_length, int)
            or self.max_query_length <= 0
        ):
            raise APIConfigError(
                "max_query_length must be a positive integer."
            )

        if not isinstance(self.expose_docs, bool):
            raise APIConfigError(
                "expose_docs must be a boolean."
            )