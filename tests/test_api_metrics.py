"""Tests for EnterpriseIQ operational metrics."""

from __future__ import annotations

from fastapi.testclient import TestClient

from enterpriseiq.api import ServiceContainer
from enterpriseiq.api.app import create_app


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    app = create_app()

    with TestClient(app) as client:
        client.get("/")
        response = client.get("/v1/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE enterpriseiq_http_requests_total counter" in response.text
    assert 'method="GET",path="/",status="200"} 1' in response.text


def test_metrics_count_status_codes() -> None:
    app = create_app(container=ServiceContainer())

    with TestClient(app) as client:
        client.get("/v1/health")
        client.post("/v1/agent", json={"query": "policy"})
        response = client.get("/v1/metrics")

    assert 'path="/v1/health",status="200"} 1' in response.text
    assert 'path="/v1/agent",status="503"} 1' in response.text


def test_metrics_track_server_errors() -> None:
    class FailingAgent:
        def run(self, query: str) -> None:
            del query
            raise RuntimeError("internal failure")

    app = create_app(
        container=ServiceContainer(agent=FailingAgent()),
    )

    with TestClient(app) as client:
        failure = client.post("/v1/agent", json={"query": "policy"})
        metrics = client.get("/v1/metrics")

    assert failure.status_code == 500
    assert "enterpriseiq_http_errors_total 1" in metrics.text


def test_metrics_track_request_duration() -> None:
    app = create_app()

    with TestClient(app) as client:
        client.get("/v1/health")
        response = client.get("/v1/metrics")

    assert "enterpriseiq_http_request_duration_seconds_sum" in response.text
    assert "enterpriseiq_http_request_duration_seconds_count 1" in response.text


def test_metrics_registry_is_isolated_per_app() -> None:
    first_app = create_app()
    second_app = create_app()

    with TestClient(first_app) as first:
        first.get("/v1/health")
        first_metrics = first.get("/v1/metrics")

    with TestClient(second_app) as second:
        second_metrics = second.get("/v1/metrics")

    assert 'path="/v1/health",status="200"} 1' in first_metrics.text
    assert 'path="/v1/health",status="200"}' not in second_metrics.text


def test_metrics_endpoint_uses_custom_api_prefix() -> None:
    from enterpriseiq.api import APIConfig

    app = create_app(config=APIConfig(api_prefix="/api/v2"))

    with TestClient(app) as client:
        response = client.get("/api/v2/metrics")
        old_response = client.get("/v1/metrics")

    assert response.status_code == 200
    assert old_response.status_code == 404
