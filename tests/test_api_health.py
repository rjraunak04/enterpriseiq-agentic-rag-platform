"""Tests for EnterpriseIQ health and readiness endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from enterpriseiq.api import APIConfig, ServiceContainer
from enterpriseiq.api.app import create_app


def test_health_endpoint_reports_service_liveness() -> None:
    config = APIConfig(
        environment="test",
    )
    app = create_app(config=config)

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "EnterpriseIQ",
        "version": "0.6.0",
    }


def test_health_does_not_require_agent_runtime() -> None:
    container = ServiceContainer()
    app = create_app(container=container)

    assert container.agent_ready is False

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_returns_503_without_agent() -> None:
    container = ServiceContainer()
    app = create_app(container=container)

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/v1/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "ready": False,
        "checks": {
            "agent": False,
        },
    }


def test_readiness_returns_200_with_agent() -> None:
    container = ServiceContainer(
        agent=object(),
    )
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.get("/v1/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "ready": True,
        "checks": {
            "agent": True,
        },
    }


def test_health_uses_custom_service_metadata() -> None:
    config = APIConfig(
        service_name="EnterpriseIQ-Test",
        service_version="9.9.9",
        environment="test",
    )
    app = create_app(config=config)

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "EnterpriseIQ-Test",
        "version": "9.9.9",
    }


def test_health_routes_respect_custom_api_prefix() -> None:
    config = APIConfig(
        api_prefix="/api/v2",
    )
    app = create_app(config=config)

    with TestClient(app) as client:
        custom_response = client.get("/api/v2/health")
        old_response = client.get("/v1/health")

    assert custom_response.status_code == 200
    assert old_response.status_code == 404


def test_ready_routes_respect_custom_api_prefix() -> None:
    config = APIConfig(
        api_prefix="/enterprise/v1",
    )
    container = ServiceContainer(
        agent=object(),
    )
    app = create_app(
        config=config,
        container=container,
    )

    with TestClient(app) as client:
        response = client.get("/enterprise/v1/ready")

    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_openapi_contains_health_endpoint() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/v1/health" in paths
    assert "/v1/ready" in paths


def test_openapi_documents_readiness_503() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/openapi.json")

    responses = response.json()["paths"]["/v1/ready"]["get"]["responses"]

    assert "200" in responses
    assert "503" in responses


def test_readiness_does_not_expose_agent_representation() -> None:
    class SensitiveAgent:
        def __repr__(self) -> str:
            return (
                "SensitiveAgent("
                "api_key='SUPER-SECRET-KEY'"
                ")"
            )

    container = ServiceContainer(
        agent=SensitiveAgent(),
    )
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.get("/v1/ready")

    assert response.status_code == 200

    body = response.text

    assert "SUPER-SECRET-KEY" not in body
    assert "SensitiveAgent" not in body


def test_readiness_response_contains_only_boolean_checks() -> None:
    container = ServiceContainer(
        agent=object(),
    )
    app = create_app(container=container)

    with TestClient(app) as client:
        response = client.get("/v1/ready")

    payload = response.json()

    assert payload["checks"] == {
        "agent": True,
    }
    assert all(
        isinstance(value, bool)
        for value in payload["checks"].values()
    )