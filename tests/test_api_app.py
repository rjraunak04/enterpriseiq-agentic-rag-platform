"""Tests for the EnterpriseIQ FastAPI application foundation."""

from __future__ import annotations

from fastapi import Request
from fastapi.testclient import TestClient

from enterpriseiq.api import (
    APIConfig,
    ServiceContainer,
    ServiceDependencyError,
)
from enterpriseiq.api.app import (
    create_app,
    get_service_container,
)


def test_create_app_uses_default_service_metadata() -> None:
    app = create_app()

    assert app.title == "EnterpriseIQ"
    assert app.version == "0.6.0"


def test_create_app_stores_api_config() -> None:
    config = APIConfig(
        environment="test",
    )

    app = create_app(config=config)

    assert app.state.api_config is config
    assert app.state.api_config.environment == "test"


def test_create_app_stores_service_container() -> None:
    container = ServiceContainer()

    app = create_app(container=container)

    assert app.state.service_container is container


def test_root_endpoint_returns_service_metadata() -> None:
    config = APIConfig(
        environment="test",
    )
    app = create_app(config=config)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "EnterpriseIQ",
        "version": "0.6.0",
        "environment": "test",
        "api_prefix": "/v1",
        "docs_enabled": True,
    }


def test_docs_are_enabled_by_default() -> None:
    app = create_app()

    with TestClient(app) as client:
        docs_response = client.get("/docs")
        openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert openapi_response.status_code == 200


def test_docs_can_be_disabled() -> None:
    config = APIConfig(
        expose_docs=False,
    )
    app = create_app(config=config)

    with TestClient(app) as client:
        docs_response = client.get("/docs")
        redoc_response = client.get("/redoc")
        openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 404
    assert redoc_response.status_code == 404
    assert openapi_response.status_code == 404


def test_service_container_reports_agent_readiness() -> None:
    container = ServiceContainer()

    assert container.agent_ready is False

    marker = object()
    container.agent = marker

    assert container.agent_ready is True
    assert container.require_agent() is marker


def test_service_container_rejects_missing_agent() -> None:
    container = ServiceContainer()

    try:
        container.require_agent()
    except ServiceDependencyError as exc:
        assert str(exc) == (
            "EnterpriseIQ agent runtime is not configured."
        )
    else:
        raise AssertionError(
            "Expected ServiceDependencyError."
        )


def test_lifespan_preserves_service_container() -> None:
    container = ServiceContainer()
    app = create_app(container=container)

    with TestClient(app):
        assert app.state.service_container is container


def test_request_can_resolve_service_container() -> None:
    container = ServiceContainer()
    app = create_app(container=container)

    @app.get("/test-container")
    async def test_container_endpoint(
        request: Request,
    ) -> dict[str, bool]:
        resolved = get_service_container(request)

        return {
            "same_container": resolved is container,
        }

    with TestClient(app) as client:
        response = client.get("/test-container")

    assert response.status_code == 200
    assert response.json() == {
        "same_container": True,
    }


def test_unhandled_exception_is_sanitized() -> None:
    app = create_app()

    @app.get("/explode")
    async def explode() -> None:
        raise RuntimeError(
            "SECRET_DATABASE_PASSWORD=do-not-leak"
        )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/explode")

    assert response.status_code == 500
    assert response.json() == {
        "error": "Internal service error.",
        "request_id": None,
    }

    body = response.text

    assert "SECRET_DATABASE_PASSWORD" not in body
    assert "do-not-leak" not in body


def test_openapi_contains_root_endpoint() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert "/" in schema["paths"]