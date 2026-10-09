"""Tests for the EnterpriseIQ agent HTTP endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fastapi.testclient import TestClient

from enterpriseiq.api import APIConfig, ServiceContainer
from enterpriseiq.api.app import create_app


class FakeStatus(StrEnum):
    """Test-only agent status."""

    COMPLETED = "completed"


class FakeDecisionType(StrEnum):
    """Test-only routing decision."""

    DIRECT_RAG = "direct_rag"


@dataclass(frozen=True, slots=True)
class FakeDecision:
    """Test-only decision contract."""

    decision_type: FakeDecisionType


@dataclass(frozen=True, slots=True)
class FakeCitation:
    """Test-only citation."""

    source_id: str
    document_id: str
    title: str


@dataclass(frozen=True, slots=True)
class FakeToolResult:
    """Test-only tool result."""

    tool_name: str
    status: str
    output: dict[str, Any]
    call_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FakeAgentResponse:
    """Test-only internal agent response."""

    status: FakeStatus
    decision: FakeDecision
    answer: str
    citations: tuple[FakeCitation, ...] = ()
    tool_results: tuple[FakeToolResult, ...] = ()
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class FakeAgent:
    """Synchronous test agent."""

    def __init__(
        self,
        response: FakeAgentResponse,
    ) -> None:
        self.response = response
        self.queries: list[str] = []

    def run(
        self,
        query: str,
    ) -> FakeAgentResponse:
        self.queries.append(query)
        return self.response


class AsyncFakeAgent:
    """Asynchronous test agent."""

    def __init__(
        self,
        response: FakeAgentResponse,
    ) -> None:
        self.response = response

    async def run(
        self,
        query: str,
    ) -> FakeAgentResponse:
        del query
        return self.response


def _response() -> FakeAgentResponse:
    """Build a representative successful agent response."""

    return FakeAgentResponse(
        status=FakeStatus.COMPLETED,
        decision=FakeDecision(
            decision_type=FakeDecisionType.DIRECT_RAG,
        ),
        answer=(
            "Employees may work remotely up to "
            "three days per workweek."
        ),
        citations=(
            FakeCitation(
                source_id="S1",
                document_id="remote-work-policy",
                title="Remote Work Policy",
            ),
        ),
        tool_results=(
            FakeToolResult(
                tool_name="search_knowledge",
                status="success",
                call_id="call-001",
                output={
                    "matches": 1,
                },
            ),
        ),
        metadata={
            "grounded": True,
        },
    )


def test_agent_endpoint_executes_configured_agent() -> None:
    agent = FakeAgent(
        _response()
    )
    container = ServiceContainer(
        agent=agent,
    )
    app = create_app(
        container=container,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/agent",
            json={
                "query": "What is the remote work policy?",
                "query_id": "query-001",
            },
        )

    assert response.status_code == 200
    assert agent.queries == [
        "What is the remote work policy?"
    ]

    assert response.json() == {
        "query_id": "query-001",
        "status": "completed",
        "decision": "direct_rag",
        "answer": (
            "Employees may work remotely up to "
            "three days per workweek."
        ),
        "citations": [
            {
                "source_id": "S1",
                "document_id": "remote-work-policy",
                "title": "Remote Work Policy",
            }
        ],
        "tool_results": [
            {
                "tool_name": "search_knowledge",
                "status": "success",
                "call_id": "call-001",
                "output": {
                    "matches": 1,
                },
                "error": None,
            }
        ],
        "metadata": {
            "grounded": True,
        },
    }


def test_agent_endpoint_supports_async_agent() -> None:
    container = ServiceContainer(
        agent=AsyncFakeAgent(
            _response()
        ),
    )
    app = create_app(
        container=container,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/agent",
            json={
                "query": "Remote work policy",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_agent_endpoint_returns_503_without_runtime() -> None:
    app = create_app(
        container=ServiceContainer(),
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/v1/agent",
            json={
                "query": "Remote work policy",
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "error": "Agent runtime is unavailable.",
        "request_id": response.headers["X-Request-ID"],
    }


def test_agent_endpoint_rejects_empty_query() -> None:
    app = create_app(
        container=ServiceContainer(
            agent=FakeAgent(
                _response()
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/agent",
            json={
                "query": "",
            },
        )

    assert response.status_code == 422


def test_agent_endpoint_rejects_whitespace_query() -> None:
    app = create_app(
        container=ServiceContainer(
            agent=FakeAgent(
                _response()
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/agent",
            json={
                "query": "   ",
            },
        )

    assert response.status_code == 422
    assert response.json() == {
        "error": "Query must contain non-whitespace characters.",
        "request_id": response.headers["X-Request-ID"],
    }


def test_agent_endpoint_rejects_oversized_query() -> None:
    app = create_app(
        container=ServiceContainer(
            agent=FakeAgent(
                _response()
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/agent",
            json={
                "query": "x" * 4001,
            },
        )

    assert response.status_code == 422


def test_agent_endpoint_rejects_unknown_request_fields() -> None:
    app = create_app(
        container=ServiceContainer(
            agent=FakeAgent(
                _response()
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/agent",
            json={
                "query": "Remote work policy",
                "unexpected": "field",
            },
        )

    assert response.status_code == 422


def test_agent_endpoint_respects_custom_api_prefix() -> None:
    config = APIConfig(
        api_prefix="/api/v2",
    )
    container = ServiceContainer(
        agent=FakeAgent(
            _response()
        ),
    )
    app = create_app(
        config=config,
        container=container,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v2/agent",
            json={
                "query": "Remote work policy",
            },
        )
        old_response = client.post(
            "/v1/agent",
            json={
                "query": "Remote work policy",
            },
        )

    assert response.status_code == 200
    assert old_response.status_code == 404


def test_agent_endpoint_sanitizes_runtime_failure() -> None:
    class FailingAgent:
        def run(
            self,
            query: str,
        ) -> None:
            del query
            raise RuntimeError(
                "SECRET_API_KEY=do-not-leak"
            )

    app = create_app(
        container=ServiceContainer(
            agent=FailingAgent(),
        ),
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/v1/agent",
            json={
                "query": "Remote work policy",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": "Agent execution failed.",
        "request_id": response.headers["X-Request-ID"],
    }
    assert "SECRET_API_KEY" not in response.text
    assert "do-not-leak" not in response.text


def test_agent_endpoint_strips_query_before_execution() -> None:
    agent = FakeAgent(
        _response()
    )
    app = create_app(
        container=ServiceContainer(
            agent=agent,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/agent",
            json={
                "query": "  Remote work policy  ",
            },
        )

    assert response.status_code == 200
    assert agent.queries == [
        "Remote work policy"
    ]


def test_agent_endpoint_does_not_leak_internal_object_repr() -> None:
    class SecretObject:
        def __repr__(self) -> str:
            return "SecretObject(token='SUPER-SECRET')"

    response_model = FakeAgentResponse(
        status=FakeStatus.COMPLETED,
        decision=FakeDecision(
            decision_type=FakeDecisionType.DIRECT_RAG,
        ),
        answer="Grounded answer.",
        tool_results=(
            FakeToolResult(
                tool_name="search_knowledge",
                status="success",
                output={
                    "internal": SecretObject(),
                },
            ),
        ),
    )

    app = create_app(
        container=ServiceContainer(
            agent=FakeAgent(
                response_model
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/agent",
            json={
                "query": "Policy question",
            },
        )

    assert response.status_code == 200
    assert "SUPER-SECRET" not in response.text
    assert "SecretObject" not in response.text

    assert response.json()["tool_results"][0]["output"] == {
        "internal": None,
    }


def test_openapi_documents_agent_endpoint() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get(
            "/openapi.json"
        )

    assert response.status_code == 200

    endpoint = response.json()["paths"]["/v1/agent"]["post"]

    assert "200" in endpoint["responses"]
    assert "422" in endpoint["responses"]
    assert "500" in endpoint["responses"]
    assert "503" in endpoint["responses"]