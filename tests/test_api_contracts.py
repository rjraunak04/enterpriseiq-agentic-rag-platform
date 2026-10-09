"""Tests for EnterpriseIQ API contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from enterpriseiq.api import (
    AgentQueryRequest,
    AgentQueryResponse,
    APIConfig,
    APIConfigError,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
)


def test_default_api_config() -> None:
    config = APIConfig()

    assert config.service_name == "EnterpriseIQ"
    assert config.service_version == "0.6.0"
    assert config.api_prefix == "/v1"
    assert config.max_query_length == 4000


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("service_name", ""),
        ("service_version", " "),
        ("environment", ""),
        ("api_prefix", ""),
        ("request_id_header", " "),
    ),
)
def test_config_rejects_blank_strings(
    field_name: str,
    value: str,
) -> None:
    with pytest.raises(APIConfigError):
        APIConfig(**{field_name: value})


def test_config_requires_prefixed_api_path() -> None:
    with pytest.raises(APIConfigError):
        APIConfig(api_prefix="v1")


@pytest.mark.parametrize(
    "value",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_config_rejects_invalid_query_limit(
    value: object,
) -> None:
    with pytest.raises(APIConfigError):
        APIConfig(max_query_length=value)  # type: ignore[arg-type]


def test_config_rejects_non_boolean_docs_flag() -> None:
    with pytest.raises(APIConfigError):
        APIConfig(expose_docs="yes")  # type: ignore[arg-type]


def test_agent_request_accepts_valid_payload() -> None:
    request = AgentQueryRequest(
        query="What is the remote work policy?",
        query_id="query-001",
        metadata={"department": "HR"},
    )

    assert request.query_id == "query-001"
    assert request.metadata["department"] == "HR"


def test_agent_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        AgentQueryRequest(query="")


def test_agent_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentQueryRequest(
            query="policy",
            unexpected=True,  # type: ignore[call-arg]
        )


def test_response_contract() -> None:
    response = AgentQueryResponse(
        query_id="query-001",
        status="completed",
        decision="direct_rag",
        answer="Grounded answer.",
    )

    assert response.tool_results == []
    assert response.metadata == {}


def test_health_contract() -> None:
    response = HealthResponse(
        status="ok",
        service="EnterpriseIQ",
        version="0.6.0",
    )

    assert response.status == "ok"


def test_readiness_contract() -> None:
    response = ReadinessResponse(
        status="ready",
        ready=True,
        checks={"agent": True},
    )

    assert response.ready is True


def test_error_contract() -> None:
    response = ErrorResponse(
        error="Request failed.",
        request_id="req-001",
    )

    assert response.request_id == "req-001"