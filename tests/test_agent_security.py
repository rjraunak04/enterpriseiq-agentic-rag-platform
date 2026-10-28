"""Security and failure regression tests for EnterpriseIQ agents."""

from __future__ import annotations

from typing import Any

import pytest

from enterpriseiq.agent.execution import (
    SafeToolExecutor,
    ToolExecutionConfig,
)
from enterpriseiq.agent.models import (
    AgentContractError,
    AgentDecision,
    AgentQuery,
    AgentStatus,
    DecisionType,
    ToolCall,
    ToolExecutionStatus,
)
from enterpriseiq.agent.orchestrator import (
    AgenticRAGOrchestrator,
    AgentOrchestrationError,
)
from enterpriseiq.agent.routing import (
    DeterministicAgentRouter,
    RoutingConfig,
)
from enterpriseiq.agent.tools import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
)
from enterpriseiq.rag import GroundedRAGPipeline
from enterpriseiq.rag.models import (
    GroundedResponse,
    RAGQuery,
)


def safe_handler(
    *,
    context: ToolContext,
    value: str,
) -> dict[str, Any]:
    """Return deterministic read-only output."""
    return {
        "value": value,
        "query_id": context.query_id,
    }


def secret_failure_handler(
    *,
    context: ToolContext,
) -> None:
    """Simulate a backend failure containing sensitive details."""
    del context

    raise RuntimeError(
        "postgresql://admin:"
        "super-secret-password@internal-db:5432/prod"
    )


def write_handler(
    *,
    context: ToolContext,
) -> str:
    """Represent a write-capable tool used for rejection tests."""
    del context
    return "write completed"


def build_registry() -> ToolRegistry:
    """Build safe and intentionally unsafe tool fixtures."""
    return ToolRegistry(
        (
            ToolDefinition(
                name="safe_lookup",
                description="Read-only enterprise lookup.",
                handler=safe_handler,
                parameters=(
                    ToolParameter(
                        name="value",
                        value_type=str,
                        description="Lookup value.",
                    ),
                ),
                read_only=True,
            ),
            ToolDefinition(
                name="secret_failure",
                description="Failure fixture.",
                handler=secret_failure_handler,
                read_only=True,
            ),
            ToolDefinition(
                name="write_operation",
                description="Write-capable fixture.",
                handler=write_handler,
                read_only=False,
            ),
        )
    )


class StaticRouter:
    """Return a predefined routing decision."""

    def __init__(
        self,
        decision: AgentDecision,
    ) -> None:
        self._decision = decision

    def route(
        self,
        query: AgentQuery,
    ) -> AgentDecision:
        del query
        return self._decision


class ExplodingRouter:
    """Simulate an internal routing failure."""

    def route(
        self,
        query: AgentQuery,
    ) -> AgentDecision:
        del query

        raise RuntimeError(
            "router-api-key=extremely-sensitive-secret"
        )


class StubRAGPipeline(GroundedRAGPipeline):
    """Minimal grounded-RAG test double."""

    def __init__(
        self,
        *,
        response: GroundedResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error

    def answer(
        self,
        query: str,
        *,
        filters: object = None,
    ) -> GroundedResponse:
        del query, filters

        if self._error is not None:
            raise self._error

        if self._response is None:
            raise RuntimeError(
                "No response configured."
            )

        return self._response


def grounded_response(
    *,
    answer: str = "Grounded enterprise answer.",
    abstained: bool = False,
) -> GroundedResponse:
    """Create a deterministic grounded response fixture."""
    return GroundedResponse(
        query=RAGQuery(
            text="enterprise query"
        ),
        answer=answer,
        citations=(),
        grounded=not abstained,
        abstained=abstained,
        evidence_count=(
            0
            if abstained
            else 1
        ),
        metadata={},
    )


def build_orchestrator(
    decision: AgentDecision,
    *,
    rag_response: GroundedResponse | None = None,
    rag_error: Exception | None = None,
) -> AgenticRAGOrchestrator:
    """Build an orchestrator with production execution controls."""
    return AgenticRAGOrchestrator(
        router=StaticRouter(
            decision
        ),
        rag_pipeline=StubRAGPipeline(
            response=(
                rag_response
                or grounded_response()
            ),
            error=rag_error,
        ),
        tool_executor=SafeToolExecutor(
            build_registry()
        ),
    )


@pytest.mark.parametrize(
    "query",
    (
        "delete the production database",
        "drop database customers",
        "drop table employees",
        "run python to delete files",
        "execute shell command",
        "use powershell to remove file",
        "call os.system",
        "launch subprocess command",
    ),
)
def test_router_abstains_from_unsafe_execution_requests(
    query: str,
) -> None:
    registry = build_registry()

    router = DeterministicAgentRouter(
        registry=registry,
    )

    decision = router.route(
        AgentQuery(
            text=query
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_router_does_not_route_unregistered_explicit_tool() -> None:
    registry = build_registry()

    router = DeterministicAgentRouter(
        registry=registry,
    )

    decision = router.route(
        AgentQuery(
            text="perform enterprise operation",
            metadata={
                "tool_name": "arbitrary_shell",
            },
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_router_cannot_bypass_allowlist_with_metadata() -> None:
    registry = build_registry()

    router = DeterministicAgentRouter(
        registry=registry,
    )

    decision = router.route(
        AgentQuery(
            text="perform lookup",
            metadata={
                "tool_name": "__import__",
                "tool_arguments": {
                    "module": "os",
                },
            },
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN


def test_executor_rejects_unknown_tool() -> None:
    executor = SafeToolExecutor(
        build_registry()
    )

    result = executor.execute(
        ToolCall(
            tool_name="not_registered",
        )
    )

    assert result.status is ToolExecutionStatus.REJECTED
    assert result.output is None


def test_executor_rejects_write_tool_by_default() -> None:
    executor = SafeToolExecutor(
        build_registry()
    )

    result = executor.execute(
        ToolCall(
            tool_name="write_operation",
        )
    )

    assert result.status is ToolExecutionStatus.REJECTED
    assert result.output is None


def test_executor_can_only_run_write_tool_when_explicitly_enabled() -> None:
    executor = SafeToolExecutor(
        build_registry(),
        config=ToolExecutionConfig(
            require_read_only=False,
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="write_operation",
        )
    )

    assert result.status is ToolExecutionStatus.SUCCESS
    assert result.output == "write completed"


def test_executor_rejects_unknown_arguments() -> None:
    executor = SafeToolExecutor(
        build_registry()
    )

    result = executor.execute(
        ToolCall(
            tool_name="safe_lookup",
            arguments={
                "value": "policy",
                "unexpected": "payload",
            },
        )
    )

    assert result.status is ToolExecutionStatus.REJECTED


def test_executor_rejects_missing_required_arguments() -> None:
    executor = SafeToolExecutor(
        build_registry()
    )

    result = executor.execute(
        ToolCall(
            tool_name="safe_lookup",
        )
    )

    assert result.status is ToolExecutionStatus.REJECTED


def test_executor_rejects_wrong_argument_type() -> None:
    executor = SafeToolExecutor(
        build_registry()
    )

    result = executor.execute(
        ToolCall(
            tool_name="safe_lookup",
            arguments={
                "value": 123,
            },
        )
    )

    assert result.status is ToolExecutionStatus.REJECTED


def test_executor_hides_internal_handler_exception() -> None:
    executor = SafeToolExecutor(
        build_registry()
    )

    result = executor.execute(
        ToolCall(
            tool_name="secret_failure",
        )
    )

    assert result.status is ToolExecutionStatus.FAILED
    assert result.output is None

    assert "super-secret-password" not in (
        result.error or ""
    )

    assert "internal-db" not in (
        result.error or ""
    )


def test_executor_error_details_require_explicit_opt_in() -> None:
    executor = SafeToolExecutor(
        build_registry(),
        config=ToolExecutionConfig(
            expose_error_details=True,
        ),
    )

    result = executor.execute(
        ToolCall(
            tool_name="secret_failure",
        )
    )

    assert result.status is ToolExecutionStatus.FAILED

    assert "super-secret-password" in (
        result.error or ""
    )


def test_agent_does_not_expose_tool_backend_secret() -> None:
    decision = AgentDecision(
        decision_type=DecisionType.TOOL_CALL,
        tool_call=ToolCall(
            tool_name="secret_failure",
        ),
    )

    orchestrator = build_orchestrator(
        decision
    )

    response = orchestrator.run(
        "perform lookup"
    )

    assert response.status is AgentStatus.FAILED

    assert (
        "super-secret-password"
        not in response.answer
    )

    assert (
        "internal-db"
        not in response.answer
    )


def test_agent_does_not_expose_rag_backend_secret() -> None:
    decision = AgentDecision(
        decision_type=DecisionType.DIRECT_RAG,
    )

    orchestrator = build_orchestrator(
        decision,
        rag_error=RuntimeError(
            "vector-db-token=secret-token-123"
        ),
    )

    response = orchestrator.run(
        "enterprise policy"
    )

    assert response.status is AgentStatus.FAILED

    assert (
        "secret-token-123"
        not in response.answer
    )

    assert (
        response.metadata["failure_stage"]
        == "rag_pipeline"
    )


def test_router_exception_does_not_expose_secret_in_wrapper_message() -> None:
    orchestrator = AgenticRAGOrchestrator(
        router=ExplodingRouter(),
        rag_pipeline=StubRAGPipeline(
            response=grounded_response()
        ),
        tool_executor=SafeToolExecutor(
            build_registry()
        ),
    )

    with pytest.raises(
        AgentOrchestrationError
    ) as exc_info:
        orchestrator.run(
            "enterprise query"
        )

    message = str(
        exc_info.value
    )

    assert (
        "extremely-sensitive-secret"
        not in message
    )

    assert message == "Agent routing failed."


def test_successful_tool_execution_preserves_audit_identity() -> None:
    decision = AgentDecision(
        decision_type=DecisionType.TOOL_CALL,
        tool_call=ToolCall(
            tool_name="safe_lookup",
            arguments={
                "value": "remote-work",
            },
            call_id="security-call-001",
        ),
    )

    orchestrator = build_orchestrator(
        decision
    )

    response = orchestrator.run(
        AgentQuery(
            text="lookup remote work",
            query_id="security-query-001",
        )
    )

    assert response.status is AgentStatus.COMPLETED
    assert len(response.tool_results) == 1

    result = response.tool_results[0]

    assert result.call_id == "security-call-001"
    assert result.tool_name == "safe_lookup"
    assert result.status is ToolExecutionStatus.SUCCESS


def test_tool_rejection_becomes_contract_valid_abstention() -> None:
    decision = AgentDecision(
        decision_type=DecisionType.TOOL_CALL,
        tool_call=ToolCall(
            tool_name="not_registered",
        ),
    )

    orchestrator = build_orchestrator(
        decision
    )

    response = orchestrator.run(
        "perform unsupported operation"
    )

    assert response.status is AgentStatus.ABSTAINED

    assert (
        response.decision.decision_type
        is DecisionType.ABSTAIN
    )

    assert (
        response.metadata["tool_status"]
        == ToolExecutionStatus.REJECTED.value
    )


def test_rag_abstention_becomes_contract_valid_abstention() -> None:
    decision = AgentDecision(
        decision_type=DecisionType.DIRECT_RAG,
    )

    orchestrator = build_orchestrator(
        decision,
        rag_response=grounded_response(
            answer=(
                "Insufficient enterprise evidence."
            ),
            abstained=True,
        ),
    )

    response = orchestrator.run(
        "unknown policy"
    )

    assert response.status is AgentStatus.ABSTAINED

    assert (
        response.decision.decision_type
        is DecisionType.ABSTAIN
    )

    assert response.metadata["abstained"] is True


def test_original_tool_route_remains_in_audit_trace_after_rejection() -> None:
    original_decision = AgentDecision(
        decision_type=DecisionType.TOOL_CALL,
        tool_call=ToolCall(
            tool_name="not_registered",
        ),
    )

    orchestrator = build_orchestrator(
        original_decision
    )

    response = orchestrator.run(
        "unsupported tool"
    )

    assert (
        response.decision.decision_type
        is DecisionType.ABSTAIN
    )

    trace_decision = (
        response.trace.steps[0].decision
    )

    assert trace_decision is not None

    assert (
        trace_decision.decision_type
        is DecisionType.TOOL_CALL
    )

    assert trace_decision.tool_call is not None

    assert (
        trace_decision.tool_call.tool_name
        == "not_registered"
    )


def test_original_direct_rag_route_remains_in_trace_after_abstention() -> None:
    original_decision = AgentDecision(
        decision_type=DecisionType.DIRECT_RAG,
    )

    orchestrator = build_orchestrator(
        original_decision,
        rag_response=grounded_response(
            answer="No grounded evidence.",
            abstained=True,
        ),
    )

    response = orchestrator.run(
        "unknown enterprise fact"
    )

    assert (
        response.decision.decision_type
        is DecisionType.ABSTAIN
    )

    trace_decision = (
        response.trace.steps[0].decision
    )

    assert trace_decision is not None

    assert (
        trace_decision.decision_type
        is DecisionType.DIRECT_RAG
    )


def test_query_metadata_cannot_override_routing_with_unknown_tool() -> None:
    registry = build_registry()

    router = DeterministicAgentRouter(
        registry=registry,
        config=RoutingConfig(),
    )

    query = AgentQuery(
        text="enterprise lookup",
        metadata={
            "tool_name": "shell_exec",
            "tool_arguments": {
                "command": "rm -rf /",
            },
        },
    )

    decision = router.route(
        query
    )

    assert decision.decision_type is DecisionType.ABSTAIN


@pytest.mark.parametrize(
    "payload",
    (
        "",
        " ",
        "\n",
        "\t",
    ),
)
def test_empty_or_whitespace_queries_remain_invalid(
    payload: str,
) -> None:
    with pytest.raises(
        AgentContractError
    ):
        AgentQuery(
            text=payload
        )


def test_agent_trace_does_not_contain_raw_backend_exception() -> None:
    decision = AgentDecision(
        decision_type=DecisionType.TOOL_CALL,
        tool_call=ToolCall(
            tool_name="secret_failure",
        ),
    )

    orchestrator = build_orchestrator(
        decision
    )

    response = orchestrator.run(
        "failure audit"
    )

    serialized_trace = repr(
        response.trace
    )

    assert (
        "super-secret-password"
        not in serialized_trace
    )

    assert (
        "internal-db"
        not in serialized_trace
    )


def test_repeated_safe_tool_calls_are_deterministic() -> None:
    decision = AgentDecision(
        decision_type=DecisionType.TOOL_CALL,
        tool_call=ToolCall(
            tool_name="safe_lookup",
            arguments={
                "value": "security-policy",
            },
            call_id="deterministic-call",
        ),
    )

    orchestrator = build_orchestrator(
        decision
    )

    first = orchestrator.run(
        AgentQuery(
            text="lookup security policy",
            query_id="deterministic-query",
        )
    )

    second = orchestrator.run(
        AgentQuery(
            text="lookup security policy",
            query_id="deterministic-query",
        )
    )

    assert first.status is second.status
    assert first.answer == second.answer
    assert first.tool_results == second.tool_results