"""Tests for the recruiter demo and API hardening."""

from __future__ import annotations

from fastapi.testclient import TestClient

from enterpriseiq.api import APIConfig, ServiceContainer
from enterpriseiq.api.app import create_app


class EchoAgent:
    def run(self, query: str) -> dict[str, object]:
        return {
            "status": "completed",
            "decision": "direct_rag",
            "answer": query,
            "citations": [],
            "tool_results": [],
            "metadata": {},
        }


def test_demo_interface_is_available() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/demo")

    assert response.status_code == 200
    assert "EnterpriseIQ" in response.text
    assert "/v1/agent" in response.text
    assert 'maxlength="4000"' in response.text


def test_demo_uses_custom_api_configuration() -> None:
    app = create_app(config=APIConfig(api_prefix="/api/v2", max_query_length=25))
    with TestClient(app) as client:
        response = client.get("/demo")

    assert "/api/v2/agent" in response.text
    assert 'maxlength="25"' in response.text


def test_runtime_query_limit_uses_api_config() -> None:
    app = create_app(
        config=APIConfig(max_query_length=5),
        container=ServiceContainer(agent=EchoAgent()),
    )
    with TestClient(app) as client:
        response = client.post("/v1/agent", json={"query": "123456"})

    assert response.status_code == 422
    assert response.json()["error"] == "Query exceeds maximum length of 5."
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_security_headers_are_attached() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_demo_does_not_enable_permissive_cors() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/demo", headers={"Origin": "https://untrusted.example"})

    assert "access-control-allow-origin" not in response.headers
