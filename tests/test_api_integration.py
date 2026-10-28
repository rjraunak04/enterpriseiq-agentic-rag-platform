"""End-to-end integration coverage for the Day-06 service boundary."""

from __future__ import annotations

from fastapi.testclient import TestClient

from enterpriseiq.api import ServiceContainer
from enterpriseiq.api.app import create_app


class IntegrationAgent:
    def run(self, query: str) -> dict[str, object]:
        return {
            "status": "completed",
            "decision": "direct_rag",
            "answer": f"Grounded: {query}",
            "citations": [
                {
                    "source_id": "S1",
                    "document_id": "remote-work-policy",
                    "title": "Remote Work Policy",
                }
            ],
            "tool_results": [],
            "metadata": {"grounded": True},
        }


def test_service_flow_is_correlated_observable_and_grounded() -> None:
    app = create_app(container=ServiceContainer(agent=IntegrationAgent()))

    with TestClient(app) as client:
        health = client.get("/v1/health")
        ready = client.get("/v1/ready")
        answer = client.post(
            "/v1/agent",
            headers={"X-Request-ID": "integration-001"},
            json={"query": "What is the remote work policy?"},
        )
        metrics = client.get("/v1/metrics")
        demo = client.get("/demo")

    assert health.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert answer.status_code == 200
    assert answer.headers["X-Request-ID"] == "integration-001"
    assert answer.json()["answer"].startswith("Grounded:")
    assert answer.json()["citations"][0]["source_id"] == "S1"
    assert 'path="/v1/agent",status="200"} 1' in metrics.text
    assert "enterpriseiq_http_request_duration_seconds_count" in metrics.text
    assert demo.status_code == 200
    assert "/v1/agent" in demo.text
