"""Tests for controlled EnterpriseIQ tool execution."""

from __future__ import annotations

from typing import Any

import pytest

from enterpriseiq.agent.execution import (
    SafeToolExecutor,
    ToolExecutionConfig,
    ToolExecutionError,
    execute_tool,
)
from enterpriseiq.agent.models import (
    ToolCall,
    ToolExecutionStatus,
)
from enterpriseiq.agent.tools import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
)


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
    raise RuntimeError("sensitive backend detail")


def mutating_handler(
    *,
    context: ToolContext,
) -> str:
    del context
    return "mutation"


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        (
            ToolDefinition(
                name="echo",
                description="Return supplied text.",
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
                description="Fail for execution testing.",
                handler=failing_handler,
                read_only=True,
            ),
            ToolDefinition(
                name="mutating_tool",
                description="Represent a write-capable operation.",
                handler=mutating_handler,
                read_only=False,
            ),
        )
    )


def make_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str | None = None,
) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        arguments=arguments or {},
        call_id=call_id,
    )


def test_execution_config_defaults_are_safe() -> None:
    config = ToolExecutionConfig()

    assert config.require_read_only is True
    assert config.expose_error_details is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("require_read_only", 1),
        ("require_read_only", "yes"),
        ("expose_error_details", 1),
        ("expose_error_details", "yes"),
    ),
)
def test_execution_config_rejects_non_boolean_values(
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}

    with pytest.raises(ToolExecutionError):
        ToolExecutionConfig(**kwargs)


def test_executor_requires_registry() -> None:
    with pytest.raises(ToolExecutionError):
        SafeToolExecutor(object())  # type: ignore[arg-type]


def test_executor_rejects_invalid_config() -> None:
    with pytest.raises(ToolExecutionError):
        SafeToolExecutor(
            build_registry(),
            config=object(),  # type: ignore[arg-type]
        )


def test_executor_exposes_registry_and_config() -> None:
    registry = build_registry()
    config = ToolExecutionConfig()

    executor = SafeToolExecutor(
        registry,
        config=config,
    )

    assert executor.registry is registry
    assert executor.config is config


def test_execute_requires_tool_call() -> None:
    executor = SafeToolExecutor(build_registry())

    with pytest.raises(ToolExecutionError):
        executor.execute(object())  # type: ignore[arg-type]


def test_execute_rejects_invalid_context() -> None:
    executor = SafeToolExecutor(build_registry())

    with pytest.raises(ToolExecutionError):
        executor.execute(
            make_call("echo", {"text": "hello"}),
            context=object(),  # type: ignore[arg-type]
        )


def test_execute_success() -> None:
    executor = SafeToolExecutor(build_registry())

    call = make_call(
        "echo",
        {"text": "enterprise"},
        call_id="call-001",
    )

    result = executor.execute(
        call,
        context=ToolContext(query_id="query-001"),
    )

    assert result.status is ToolExecutionStatus.SUCCESS
    assert result.tool_name == "echo"
    assert result.call_id == "call-001"
    assert result.output == {
        "text": "enterprise",
        "query_id": "query-001",
    }
    assert result.error is None


def test_execute_preserves_call_id() -> None:
    executor = SafeToolExecutor(build_registry())

    result = executor.execute(
        make_call(
            "echo",
            {"text": "hello"},
            call_id="tool-call-123",
        )
    )

    assert result.status is ToolExecutionStatus.SUCCESS
    assert result.tool_name == "echo"
    assert result.call_id == "tool-call-123"


def test_execute_rejects_unknown_tool() -> None:
    executor = SafeToolExecutor(build_registry())

    result = executor.execute(
        make_call(
            "not_registered",
            call_id="unknown-001",
        )
    )

    assert result.status is ToolExecutionStatus.REJECTED
    assert result.tool_name == "not_registered"
    assert result.call_id == "unknown-001"
    assert result.output is None
    assert result.error is not None
    assert "not registered" in result.error.lower()


def test_execute_rejects_write_capable_tool_by_default() -> None:
    executor = SafeToolExecutor(build_registry())

    result = executor.execute(
        make_call("mutating_tool")
    )

    assert result.status is ToolExecutionStatus.REJECTED
    assert result.tool_name == "mutating_tool"
    assert result.output is None
    assert result.error is not None
    assert "read-only" in result.error.lower()


def test_write_capable_tool_can_run_when_policy_explicitly_allows_it() -> None:
    executor = SafeToolExecutor(
        build_registry(),
        config=ToolExecutionConfig(
            require_read_only=False,
        ),
    )

    result = executor.execute(
        make_call("mutating_tool")
    )

    assert result.status is ToolExecutionStatus.SUCCESS
    assert result.tool_name == "mutating_tool"
    assert result.output == "mutation"
    assert result.error is None


def test_execute_rejects_missing_required_argument() -> None:
    executor = SafeToolExecutor(build_registry())

    result = executor.execute(
        make_call("echo")
    )

    assert result.status is ToolExecutionStatus.REJECTED
    assert result.tool_name == "echo"
    assert result.output is None
    assert result.error is not None
    assert "missing required argument" in result.error.lower()


def test_execute_rejects_unknown_argument() -> None:
    executor = SafeToolExecutor(build_registry())

    result = executor.execute(
        make_call(
            "echo",
            {
                "text": "hello",
                "unexpected": "value",
            },
        )
    )

    assert result.status is ToolExecutionStatus.REJECTED
    assert result.tool_name == "echo"
    assert result.output is None
    assert result.error is not None
    assert "unknown tool arguments" in result.error.lower()


def test_execute_rejects_argument_type_mismatch() -> None:
    executor = SafeToolExecutor(build_registry())

    result = executor.execute(
        make_call(
            "echo",
            {"text": 123},
        )
    )

    assert result.status is ToolExecutionStatus.REJECTED
    assert result.tool_name == "echo"
    assert result.output is None
    assert result.error is not None
    assert "must be str" in result.error.lower()


def test_execute_hides_internal_exception_details_by_default() -> None:
    executor = SafeToolExecutor(build_registry())

    result = executor.execute(
        make_call("failing_tool")
    )

    assert result.status is ToolExecutionStatus.FAILED
    assert result.tool_name == "failing_tool"
    assert result.output is None
    assert result.error == (
        "Tool 'failing_tool' failed during execution."
    )
    assert "sensitive backend detail" not in result.error


def test_execute_can_expose_exception_details_when_configured() -> None:
    executor = SafeToolExecutor(
        build_registry(),
        config=ToolExecutionConfig(
            expose_error_details=True,
        ),
    )

    result = executor.execute(
        make_call("failing_tool")
    )

    assert result.status is ToolExecutionStatus.FAILED
    assert result.tool_name == "failing_tool"
    assert result.error is not None
    assert "sensitive backend detail" in result.error


def test_execute_many_preserves_order() -> None:
    executor = SafeToolExecutor(build_registry())

    first = make_call(
        "echo",
        {"text": "first"},
        call_id="first-call",
    )
    second = make_call(
        "echo",
        {"text": "second"},
        call_id="second-call",
    )

    results = executor.execute_many(
        (first, second),
        context=ToolContext(query_id="batch-query"),
    )

    assert len(results) == 2

    assert results[0].status is ToolExecutionStatus.SUCCESS
    assert results[0].tool_name == "echo"
    assert results[0].call_id == "first-call"
    assert results[0].output["text"] == "first"

    assert results[1].status is ToolExecutionStatus.SUCCESS
    assert results[1].tool_name == "echo"
    assert results[1].call_id == "second-call"
    assert results[1].output["text"] == "second"


def test_execute_many_returns_independent_failure_results() -> None:
    executor = SafeToolExecutor(build_registry())

    results = executor.execute_many(
        (
            make_call(
                "echo",
                {"text": "before"},
            ),
            make_call("failing_tool"),
            make_call(
                "echo",
                {"text": "after"},
            ),
        )
    )

    assert [
        result.status
        for result in results
    ] == [
        ToolExecutionStatus.SUCCESS,
        ToolExecutionStatus.FAILED,
        ToolExecutionStatus.SUCCESS,
    ]

    assert [
        result.tool_name
        for result in results
    ] == [
        "echo",
        "failing_tool",
        "echo",
    ]


def test_execute_many_requires_tuple() -> None:
    executor = SafeToolExecutor(build_registry())

    with pytest.raises(ToolExecutionError):
        executor.execute_many(  # type: ignore[arg-type]
            [make_call("echo", {"text": "hello"})]
        )


def test_execute_many_rejects_invalid_call_member() -> None:
    executor = SafeToolExecutor(build_registry())

    with pytest.raises(ToolExecutionError):
        executor.execute_many(
            (
                make_call("echo", {"text": "hello"}),
                object(),  # type: ignore[arg-type]
            )
        )


def test_execute_many_rejects_invalid_context() -> None:
    executor = SafeToolExecutor(build_registry())

    with pytest.raises(ToolExecutionError):
        executor.execute_many(
            (),
            context=object(),  # type: ignore[arg-type]
        )


def test_execute_tool_convenience_wrapper() -> None:
    registry = build_registry()

    result = execute_tool(
        registry,
        make_call(
            "echo",
            {"text": "wrapper"},
            call_id="wrapper-call",
        ),
        context=ToolContext(
            query_id="wrapper-query"
        ),
    )

    assert result.status is ToolExecutionStatus.SUCCESS
    assert result.tool_name == "echo"
    assert result.call_id == "wrapper-call"
    assert result.output == {
        "text": "wrapper",
        "query_id": "wrapper-query",
    }


def test_same_call_produces_deterministic_success_result() -> None:
    executor = SafeToolExecutor(build_registry())

    call = make_call(
        "echo",
        {"text": "stable"},
        call_id="stable-call",
    )

    context = ToolContext(
        query_id="stable-query"
    )

    first = executor.execute(
        call,
        context=context,
    )
    second = executor.execute(
        call,
        context=context,
    )

    assert first == second
    assert first.tool_name == "echo"
    assert first.call_id == "stable-call"