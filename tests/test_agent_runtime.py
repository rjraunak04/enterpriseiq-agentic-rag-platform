"""Regression tests for the production agent runtime tool schemas."""

import pytest

from enterpriseiq.agent.models import ToolCall, ToolExecutionStatus
from enterpriseiq.agent.tools import ToolContext
from enterpriseiq.api.agent_runtime import build_agent_runtime


@pytest.fixture(scope="module")
def runtime():
    """Build the bundled runtime once for schema regression tests."""
    return build_agent_runtime()


def test_list_documents_accepts_optional_none_department(runtime) -> None:
    result = runtime.tool_executor.execute(
        ToolCall(
            tool_name="list_documents",
            arguments={"department": None},
        ),
        context=ToolContext(),
    )

    assert result.status is ToolExecutionStatus.SUCCESS
    assert result.error is None
    assert isinstance(result.output, dict)
    assert result.output["count"] > 0
    assert len(result.output["documents"]) == result.output["count"]


def test_list_documents_filters_by_department(runtime) -> None:
    departments = runtime.tool_executor.execute(
        ToolCall(tool_name="list_departments"),
        context=ToolContext(),
    )
    assert departments.status is ToolExecutionStatus.SUCCESS
    assert isinstance(departments.output, dict)
    department_names = departments.output["departments"]
    assert department_names

    department = department_names[0]
    result = runtime.tool_executor.execute(
        ToolCall(
            tool_name="list_documents",
            arguments={"department": department},
        ),
        context=ToolContext(),
    )

    assert result.status is ToolExecutionStatus.SUCCESS
    assert result.error is None
    assert result.output["department"] == department
    assert result.output["count"] > 0
    assert len(result.output["documents"]) == result.output["count"]
