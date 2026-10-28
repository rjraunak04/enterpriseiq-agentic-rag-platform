"""Tests for EnterpriseIQ request correlation and structured logging."""

from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from enterpriseiq.api import APIConfig, ServiceContainer
from enterpriseiq.api.app import create_app


def test_request_id_is_preserved_on_response() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/", headers={"X-Request-ID": "request-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-123"


def test_request_id_is_generated_when_missing() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/")
    request_id = response.headers["X-Request-ID"]
    assert request_id
    assert len(request_id) == 36


def test_invalid_request_id_is_replaced() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get(
            "/",
            headers={"X-Request-ID": "unsafe request id"},
        )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "unsafe request id"


def test_agent_error_contains_request_id() -> None:
    app = create_app(container=ServiceContainer())
    with TestClient(app) as client:
        response = client.post(
            "/v1/agent",
            headers={"X-Request-ID": "agent-request-1"},
            json={"query": "Remote work policy"},
        )
    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "agent-request-1"
    assert response.json() == {
        "error": "Agent runtime is unavailable.",
        "request_id": "agent-request-1",
    }


def test_structured_log_avoids_query_payload(caplog) -> None:
    app = create_app(container=ServiceContainer())
    with caplog.at_level(logging.INFO, logger="enterpriseiq.api"):
        with TestClient(app) as client:
            response = client.post(
                "/v1/agent",
                headers={"X-Request-ID": "safe-log-1"},
                json={"query": "PRIVATE_QUERY_TEXT"},
            )
    assert response.status_code == 503
    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "enterpriseiq.api"
    ]
    assert records
    assert any(record["request_id"] == "safe-log-1" for record in records)
    assert all("PRIVATE_QUERY_TEXT" not in record.message for record in caplog.records)


def test_custom_request_id_header_is_supported() -> None:
    app = create_app(
        config=APIConfig(request_id_header="X-Correlation-ID")
    )
    with TestClient(app) as client:
        response = client.get(
            "/",
            headers={"X-Correlation-ID": "correlation-7"},
        )
    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "correlation-7"
