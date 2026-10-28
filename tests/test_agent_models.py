"""Tests for immutable agent execution contracts."""

from dataclasses import FrozenInstanceError

import pytest

from enterpriseiq.agent import (
    AgentContractError,
    AgentDecision,
    AgentQuery,
    AgentResponse,
    AgentStatus,
    AgentStep,
    AgentTrace,
    DecisionType,
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
)


def test_agent_query_normalizes_text_and_metadata_is_immutable() -> None:
    query = AgentQuery(
        text="  What is the remote work policy?  ",
        query_id=" q-1 ",
        metadata={"tenant": "demo"},
    )

    assert query.text == "What is the remote work policy?"
    assert query.query_id == "q-1"

    with pytest.raises(TypeError):
        query.metadata["tenant"] = "changed"


@pytest.mark.parametrize("value", ["", "   ", None])
def test_agent_query_rejects_invalid_text(value: object) -> None:
    with pytest.raises(AgentContractError):
        AgentQuery(text=value)  # type: ignore[arg-type]


def test_tool_call_is_immutable_and_normalized() -> None:
    call = ToolCall(
        tool_name=" search_knowledge ",
        arguments={"query": "remote work"},
        call_id=" call-1 ",
    )

    assert call.tool_name == "search_knowledge"
    assert call.call_id == "call-1"

    with pytest.raises(TypeError):
        call.arguments["query"] = "changed"


def test_tool_call_rejects_blank_name() -> None:
    with pytest.raises(AgentContractError):
        ToolCall(tool_name=" ")


def test_tool_decision_requires_tool_call() -> None:
    with pytest.raises(AgentContractError):
        AgentDecision(decision_type=DecisionType.TOOL_CALL)


def test_direct_rag_decision_rejects_tool_call() -> None:
    call = ToolCall(tool_name="search_knowledge")

    with pytest.raises(AgentContractError):
        AgentDecision(
            decision_type=DecisionType.DIRECT_RAG,
            tool_call=call,
        )


def test_valid_tool_decision() -> None:
    call = ToolCall(
        tool_name="list_departments",
        call_id="call-2",
    )
    decision = AgentDecision(
        decision_type=DecisionType.TOOL_CALL,
        tool_call=call,
        reason="Department listing requested",
    )

    assert decision.tool_call == call


def test_successful_tool_result_rejects_error() -> None:
    with pytest.raises(AgentContractError):
        ToolResult(
            tool_name="list_departments",
            status=ToolExecutionStatus.SUCCESS,
            error="unexpected",
        )


@pytest.mark.parametrize(
    "status",
    [
        ToolExecutionStatus.REJECTED,
        ToolExecutionStatus.FAILED,
    ],
)
def test_failed_tool_result_requires_error(
    status: ToolExecutionStatus,
) -> None:
    with pytest.raises(AgentContractError):
        ToolResult(
            tool_name="search_knowledge",
            status=status,
        )


def test_valid_successful_tool_result() -> None:
    result = ToolResult(
        tool_name="list_departments",
        status=ToolExecutionStatus.SUCCESS,
        output=["Finance", "HR"],
        call_id="call-3",
    )

    assert result.output == ["Finance", "HR"]
    assert result.error is None


@pytest.mark.parametrize("step_number", [0, -1, True, 1.5])
def test_agent_step_requires_positive_integer(
    step_number: object,
) -> None:
    with pytest.raises(AgentContractError):
        AgentStep(
            step_number=step_number,  # type: ignore[arg-type]
            action="route",
        )


def test_trace_requires_contiguous_step_numbers() -> None:
    with pytest.raises(AgentContractError):
        AgentTrace(
            steps=(
                AgentStep(step_number=1, action="route"),
                AgentStep(step_number=3, action="execute"),
            )
        )


def test_trace_requires_tuple() -> None:
    with pytest.raises(AgentContractError):
        AgentTrace(
            steps=[AgentStep(step_number=1, action="route")]  # type: ignore[arg-type]
        )


def test_valid_trace_preserves_order() -> None:
    trace = AgentTrace(
        steps=(
            AgentStep(step_number=1, action="route"),
            AgentStep(step_number=2, action="respond"),
        )
    )

    assert [step.step_number for step in trace.steps] == [1, 2]


def test_completed_response_requires_answer() -> None:
    query = AgentQuery(text="What is the remote work policy?")
    decision = AgentDecision(
        decision_type=DecisionType.DIRECT_RAG
    )

    with pytest.raises(AgentContractError):
        AgentResponse(
            query=query,
            status=AgentStatus.COMPLETED,
            answer=" ",
            decision=decision,
        )


def test_abstained_response_requires_abstain_decision() -> None:
    query = AgentQuery(text="Unknown question")
    decision = AgentDecision(
        decision_type=DecisionType.DIRECT_RAG
    )

    with pytest.raises(AgentContractError):
        AgentResponse(
            query=query,
            status=AgentStatus.ABSTAINED,
            answer="Unable to answer.",
            decision=decision,
        )


def test_valid_completed_response() -> None:
    query = AgentQuery(text="List departments")
    call = ToolCall(tool_name="list_departments")
    decision = AgentDecision(
        decision_type=DecisionType.TOOL_CALL,
        tool_call=call,
    )
    result = ToolResult(
        tool_name="list_departments",
        status=ToolExecutionStatus.SUCCESS,
        output=["Finance", "HR"],
    )
    trace = AgentTrace(
        steps=(
            AgentStep(
                step_number=1,
                action="route",
                decision=decision,
            ),
            AgentStep(
                step_number=2,
                action="execute_tool",
                tool_result=result,
            ),
        )
    )

    response = AgentResponse(
        query=query,
        status=AgentStatus.COMPLETED,
        answer="Finance and HR are available.",
        decision=decision,
        trace=trace,
        tool_results=(result,),
        metadata={"route": "tool"},
    )

    assert response.status is AgentStatus.COMPLETED
    assert response.tool_results == (result,)
    assert len(response.trace.steps) == 2


def test_agent_contracts_are_frozen() -> None:
    query = AgentQuery(text="List departments")

    with pytest.raises(FrozenInstanceError):
        query.text = "changed"  # type: ignore[misc]


def test_metadata_keys_must_be_non_empty_strings() -> None:
    with pytest.raises(AgentContractError):
        AgentQuery(
            text="question",
            metadata={"": "invalid"},
        )


def test_decision_type_must_be_enum() -> None:
    with pytest.raises(AgentContractError):
        AgentDecision(
            decision_type="direct_rag",  # type: ignore[arg-type]
        )


def test_tool_execution_status_must_be_enum() -> None:
    with pytest.raises(AgentContractError):
        ToolResult(
            tool_name="search_knowledge",
            status="success",  # type: ignore[arg-type]
        )
