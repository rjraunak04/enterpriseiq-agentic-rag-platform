"""Tests for the EnterpriseIQ agentic RAG orchestrator."""

from __future__ import annotations

from typing import Any

import pytest

from enterpriseiq.agent.execution import SafeToolExecutor
from enterpriseiq.agent.models import (
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
    AgentOrchestratorConfig,
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


class StaticRouter:
    """Return one deterministic decision."""

    def __init__(self, decision: AgentDecision) -> None:
        self.decision = decision
        self.received: list[AgentQuery] = []

    def route(
        self,
        query: AgentQuery,
    ) -> AgentDecision:
        self.received.append(query)
        return self.decision


class FailingRouter:
    def route(
        self,
        query: AgentQuery,
    ) -> AgentDecision:
        del query
        raise RuntimeError("router backend secret")


def echo_handler(
    *,
    context: ToolContext,
    text: str,
) -> dict[str, Any]:
    return {
        "text": text,
        "query_id": context.query_id,
    }


def failing_handler(
    *,
    context: ToolContext,
) -> None:
    del context
    raise RuntimeError("tool backend secret")


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        (
            ToolDefinition(
                name="echo",
                description="Echo supplied text.",
                handler=echo_handler,
                parameters=(
                    ToolParameter(
                        name="text",
                        value_type=str,
                        description="Text to echo.",
                    ),
                ),
                read_only=True,
            ),
            ToolDefinition(
                name="failing_tool",
                description="Fail during execution.",
                handler=failing_handler,
                read_only=True,
            ),
        )
    )


def direct_decision() -> AgentDecision:
    return AgentDecision(
        decision_type=DecisionType.DIRECT_RAG,
        reason="Use grounded enterprise retrieval.",
    )


def tool_decision(
    name: str = "echo",
    arguments: dict[str, Any] | None = None,
) -> AgentDecision:
    return AgentDecision(
        decision_type=DecisionType.TOOL_CALL,
        tool_call=ToolCall(
            tool_name=name,
            arguments=arguments or {},
            call_id="call-001",
        ),
        reason="Use an approved enterprise tool.",
    )


def abstain_decision(
    reason: str | None = "Request is outside approved capabilities.",
) -> AgentDecision:
    return AgentDecision(
        decision_type=DecisionType.ABSTAIN,
        reason=reason,
    )


class DummyRetriever:
    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: object = None,
        min_score: float | None = None,
    ) -> tuple[object, ...]:
        del query, top_k, filters, min_score
        return ()


class DummyGenerator:
    def generate(self, prompt: object) -> object:
        del prompt
        raise RuntimeError("unused generator")


def make_real_rag_pipeline() -> GroundedRAGPipeline:
    return GroundedRAGPipeline(
        retriever=DummyRetriever(),
        generator=DummyGenerator(),
    )


class StubGroundedRAGPipeline(GroundedRAGPipeline):
    """GroundedRAGPipeline test double with deterministic output."""

    def __init__(
        self,
        response: GroundedResponse | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.received: list[str] = []

    def answer(
        self,
        query: str,
        *,
        filters: object = None,
    ) -> GroundedResponse:
        del filters
        self.received.append(query)

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise RuntimeError("No response configured.")

        return self.response


def grounded_response(
    *,
    answer: str = "Employees may work remotely up to three days.",
    grounded: bool = True,
    abstained: bool = False,
    evidence_count: int = 2,
) -> GroundedResponse:
    return GroundedResponse(
        query=RAGQuery(text="remote work policy"),
        answer=answer,
        citations=(),
        grounded=grounded,
        abstained=abstained,
        evidence_count=evidence_count,
        metadata={
            "retrieved_count": evidence_count,
        },
    )


def build_orchestrator(
    decision: AgentDecision,
    *,
    rag_response: GroundedResponse | None = None,
    rag_error: Exception | None = None,
) -> AgenticRAGOrchestrator:
    router = StaticRouter(decision)

    rag_pipeline = StubGroundedRAGPipeline(
        response=rag_response or grounded_response(),
        error=rag_error,
    )

    return AgenticRAGOrchestrator(
        router=router,
        rag_pipeline=rag_pipeline,
        tool_executor=SafeToolExecutor(
            build_registry()
        ),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("tool_success_answer", ""),
        ("tool_success_answer", "   "),
        ("tool_failure_answer", ""),
        ("default_abstention_answer", ""),
    ),
)
def test_config_rejects_empty_messages(
    field: str,
    value: str,
) -> None:
    kwargs = {field: value}

    with pytest.raises(AgentOrchestrationError):
        AgentOrchestratorConfig(**kwargs)


def test_orchestrator_rejects_invalid_router() -> None:
    with pytest.raises(AgentOrchestrationError):
        AgenticRAGOrchestrator(
            router=object(),  # type: ignore[arg-type]
            rag_pipeline=make_real_rag_pipeline(),
            tool_executor=SafeToolExecutor(
                build_registry()
            ),
        )


def test_orchestrator_rejects_invalid_rag_pipeline() -> None:
    with pytest.raises(AgentOrchestrationError):
        AgenticRAGOrchestrator(
            router=StaticRouter(direct_decision()),
            rag_pipeline=object(),  # type: ignore[arg-type]
            tool_executor=SafeToolExecutor(
                build_registry()
            ),
        )


def test_orchestrator_rejects_invalid_executor() -> None:
    with pytest.raises(AgentOrchestrationError):
        AgenticRAGOrchestrator(
            router=StaticRouter(direct_decision()),
            rag_pipeline=make_real_rag_pipeline(),
            tool_executor=object(),  # type: ignore[arg-type]
        )


def test_orchestrator_rejects_invalid_config() -> None:
    with pytest.raises(AgentOrchestrationError):
        AgenticRAGOrchestrator(
            router=StaticRouter(direct_decision()),
            rag_pipeline=make_real_rag_pipeline(),
            tool_executor=SafeToolExecutor(
                build_registry()
            ),
            config=object(),  # type: ignore[arg-type]
        )


def test_string_query_is_normalized() -> None:
    orchestrator = build_orchestrator(
        direct_decision()
    )

    response = orchestrator.run(
        "What is the remote work policy?"
    )

    assert isinstance(response.query, AgentQuery)
    assert response.query.text == (
        "What is the remote work policy?"
    )


def test_invalid_query_is_rejected() -> None:
    orchestrator = build_orchestrator(
        direct_decision()
    )

    with pytest.raises(AgentOrchestrationError):
        orchestrator.run(123)  # type: ignore[arg-type]


def test_direct_rag_returns_completed_response() -> None:
    orchestrator = build_orchestrator(
        direct_decision(),
        rag_response=grounded_response(
            answer="Remote work is allowed.",
            grounded=True,
            abstained=False,
            evidence_count=3,
        ),
    )

    response = orchestrator.run(
        AgentQuery(
            text="remote work policy",
            query_id="query-001",
        )
    )

    assert response.status is AgentStatus.COMPLETED
    assert response.answer == "Remote work is allowed."
    assert response.decision.decision_type is (
        DecisionType.DIRECT_RAG
    )
    assert len(response.trace.steps) == 2
    assert response.trace.steps[0].step_number == 1
    assert response.trace.steps[1].step_number == 2
    assert response.metadata["path"] == "direct_rag"
    assert response.metadata["grounded"] is True
    assert response.metadata["evidence_count"] == 3


def test_direct_rag_abstention_is_propagated() -> None:
    orchestrator = build_orchestrator(
        direct_decision(),
        rag_response=grounded_response(
            answer="Insufficient enterprise evidence.",
            grounded=False,
            abstained=True,
            evidence_count=0,
        ),
    )

    response = orchestrator.run(
        "unknown enterprise question"
    )

    assert response.status is AgentStatus.ABSTAINED
    assert response.answer == (
        "Insufficient enterprise evidence."
    )
    assert response.metadata["abstained"] is True


def test_direct_rag_failure_is_sanitized() -> None:
    orchestrator = build_orchestrator(
        direct_decision(),
        rag_error=RuntimeError(
            "sensitive vector database detail"
        ),
    )

    response = orchestrator.run(
        "remote work policy"
    )

    assert response.status is AgentStatus.FAILED
    assert response.answer == (
        "I could not generate a grounded enterprise answer."
    )
    assert (
        "sensitive vector database detail"
        not in response.answer
    )
    assert response.metadata["failure_stage"] == (
        "rag_pipeline"
    )


def test_successful_tool_call_returns_completed_response() -> None:
    orchestrator = build_orchestrator(
        tool_decision(
            "echo",
            {"text": "enterprise"},
        )
    )

    response = orchestrator.run(
        AgentQuery(
            text="echo enterprise",
            query_id="query-tool",
        )
    )

    assert response.status is AgentStatus.COMPLETED
    assert response.decision.decision_type is (
        DecisionType.TOOL_CALL
    )
    assert len(response.tool_results) == 1

    result = response.tool_results[0]

    assert result.status is ToolExecutionStatus.SUCCESS
    assert result.tool_name == "echo"
    assert result.call_id == "call-001"
    assert result.output == {
        "text": "enterprise",
        "query_id": "query-tool",
    }

    assert len(response.trace.steps) == 2
    assert response.trace.steps[1].tool_result == result
    assert response.metadata["path"] == "tool_call"


def test_tool_context_preserves_query_id() -> None:
    orchestrator = build_orchestrator(
        tool_decision(
            "echo",
            {"text": "context"},
        )
    )

    response = orchestrator.run(
        AgentQuery(
            text="echo context",
            query_id="enterprise-query-42",
        )
    )

    assert response.tool_results[0].output[
        "query_id"
    ] == "enterprise-query-42"


def test_unknown_tool_is_safely_abstained() -> None:
    orchestrator = build_orchestrator(
        tool_decision("unknown_tool")
    )

    response = orchestrator.run(
        "run unknown tool"
    )

    assert response.status is AgentStatus.ABSTAINED
    assert len(response.tool_results) == 1
    assert response.tool_results[0].status is (
        ToolExecutionStatus.REJECTED
    )
    assert response.metadata["tool_status"] == (
        ToolExecutionStatus.REJECTED.value
    )


def test_failed_tool_returns_failed_agent_response() -> None:
    orchestrator = build_orchestrator(
        tool_decision("failing_tool")
    )

    response = orchestrator.run(
        "run failing tool"
    )

    assert response.status is AgentStatus.FAILED
    assert len(response.tool_results) == 1
    assert response.tool_results[0].status is (
        ToolExecutionStatus.FAILED
    )
    assert "tool backend secret" not in response.answer


def test_abstain_decision_returns_abstained_response() -> None:
    orchestrator = build_orchestrator(
        abstain_decision(
            "Unsafe operation was rejected."
        )
    )

    response = orchestrator.run(
        "delete enterprise files"
    )

    assert response.status is AgentStatus.ABSTAINED
    assert response.answer == (
        "Unsafe operation was rejected."
    )
    assert response.decision.decision_type is (
        DecisionType.ABSTAIN
    )
    assert len(response.trace.steps) == 1
    assert response.trace.steps[0].action == "abstain"
    assert response.metadata["path"] == "abstain"


def test_abstain_without_reason_uses_default_message() -> None:
    orchestrator = build_orchestrator(
        abstain_decision(None)
    )

    response = orchestrator.run(
        "unsupported request"
    )

    assert response.status is AgentStatus.ABSTAINED
    assert response.answer == (
        orchestrator.config.default_abstention_answer
    )


def test_router_failure_is_wrapped() -> None:
    orchestrator = AgenticRAGOrchestrator(
        router=FailingRouter(),
        rag_pipeline=StubGroundedRAGPipeline(
            response=grounded_response()
        ),
        tool_executor=SafeToolExecutor(
            build_registry()
        ),
    )

    with pytest.raises(
        AgentOrchestrationError,
        match="Agent routing failed",
    ):
        orchestrator.run(
            "enterprise query"
        )


def test_trace_step_numbers_are_contiguous() -> None:
    orchestrator = build_orchestrator(
        tool_decision(
            "echo",
            {"text": "trace"},
        )
    )

    response = orchestrator.run(
        "trace query"
    )

    assert [
        step.step_number
        for step in response.trace.steps
    ] == [1, 2]


def test_tool_result_is_present_in_trace_and_response() -> None:
    orchestrator = build_orchestrator(
        tool_decision(
            "echo",
            {"text": "audit"},
        )
    )

    response = orchestrator.run(
        "audit query"
    )

    assert response.trace.steps[1].tool_result == (
        response.tool_results[0]
    )


def test_direct_rag_has_no_tool_results() -> None:
    orchestrator = build_orchestrator(
        direct_decision()
    )

    response = orchestrator.run(
        "remote policy"
    )

    assert response.tool_results == ()


def test_abstention_has_no_tool_results() -> None:
    orchestrator = build_orchestrator(
        abstain_decision()
    )

    response = orchestrator.run(
        "unsafe request"
    )

    assert response.tool_results == ()