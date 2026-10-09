"""Immutable contracts for EnterpriseIQ agent execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class AgentContractError(ValueError):
    """Raised when an agent contract is invalid."""


class DecisionType(StrEnum):
    """Supported high-level agent decisions."""

    DIRECT_RAG = "direct_rag"
    TOOL_CALL = "tool_call"
    ABSTAIN = "abstain"


class AgentStatus(StrEnum):
    """Final agent execution states."""

    COMPLETED = "completed"
    ABSTAINED = "abstained"
    FAILED = "failed"


class ToolExecutionStatus(StrEnum):
    """Lifecycle states for controlled tool execution."""

    PENDING = "pending"
    SUCCESS = "success"
    REJECTED = "rejected"
    FAILED = "failed"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _immutable_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgentContractError(f"{field_name} must be a mapping")

    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise AgentContractError(
                f"{field_name} keys must be non-empty strings"
            )
        normalized[key] = item

    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class AgentQuery:
    """Normalized input to the agent layer."""

    text: str
    query_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _required_text(self.text, "text"))
        object.__setattr__(
            self,
            "query_id",
            _optional_text(self.query_id, "query_id"),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Validated intent to invoke a registered tool."""

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_name",
            _required_text(self.tool_name, "tool_name"),
        )
        object.__setattr__(
            self,
            "call_id",
            _optional_text(self.call_id, "call_id"),
        )
        object.__setattr__(
            self,
            "arguments",
            _immutable_mapping(self.arguments, "arguments"),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class AgentDecision:
    """Routing decision produced for an agent query."""

    decision_type: DecisionType
    tool_call: ToolCall | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision_type, DecisionType):
            raise AgentContractError(
                "decision_type must be a DecisionType"
            )

        if (
            self.decision_type is DecisionType.TOOL_CALL
            and self.tool_call is None
        ):
            raise AgentContractError(
                "tool_call is required for TOOL_CALL decisions"
            )

        if (
            self.decision_type is not DecisionType.TOOL_CALL
            and self.tool_call is not None
        ):
            raise AgentContractError(
                "tool_call is only valid for TOOL_CALL decisions"
            )

        object.__setattr__(
            self,
            "reason",
            _optional_text(self.reason, "reason"),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result returned by controlled tool execution."""

    tool_name: str
    status: ToolExecutionStatus
    output: Any = None
    call_id: str | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_name",
            _required_text(self.tool_name, "tool_name"),
        )

        if not isinstance(self.status, ToolExecutionStatus):
            raise AgentContractError(
                "status must be a ToolExecutionStatus"
            )

        object.__setattr__(
            self,
            "call_id",
            _optional_text(self.call_id, "call_id"),
        )
        object.__setattr__(
            self,
            "error",
            _optional_text(self.error, "error"),
        )

        if (
            self.status
            in {
                ToolExecutionStatus.REJECTED,
                ToolExecutionStatus.FAILED,
            }
            and self.error is None
        ):
            raise AgentContractError(
                "error is required for rejected or failed tool results"
            )

        if (
            self.status is ToolExecutionStatus.SUCCESS
            and self.error is not None
        ):
            raise AgentContractError(
                "successful tool results cannot contain an error"
            )

        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class AgentStep:
    """Single auditable step in an agent execution trace."""

    step_number: int
    action: str
    decision: AgentDecision | None = None
    tool_result: ToolResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_number, bool)
            or not isinstance(self.step_number, int)
            or self.step_number < 1
        ):
            raise AgentContractError(
                "step_number must be a positive integer"
            )

        object.__setattr__(
            self,
            "action",
            _required_text(self.action, "action"),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class AgentTrace:
    """Ordered immutable audit trail for one agent execution."""

    steps: tuple[AgentStep, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            raise AgentContractError("steps must be a tuple")

        if not all(isinstance(step, AgentStep) for step in self.steps):
            raise AgentContractError(
                "steps must contain only AgentStep values"
            )

        numbers = [step.step_number for step in self.steps]
        expected = list(range(1, len(self.steps) + 1))
        if numbers != expected:
            raise AgentContractError(
                "agent trace step numbers must be contiguous from 1"
            )

        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Final response envelope returned by the agent layer."""

    query: AgentQuery
    status: AgentStatus
    answer: str
    decision: AgentDecision
    trace: AgentTrace = field(default_factory=AgentTrace)
    tool_results: tuple[ToolResult, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, AgentQuery):
            raise AgentContractError("query must be an AgentQuery")

        if not isinstance(self.status, AgentStatus):
            raise AgentContractError("status must be an AgentStatus")

        if not isinstance(self.decision, AgentDecision):
            raise AgentContractError(
                "decision must be an AgentDecision"
            )

        if not isinstance(self.trace, AgentTrace):
            raise AgentContractError("trace must be an AgentTrace")

        if not isinstance(self.tool_results, tuple):
            raise AgentContractError("tool_results must be a tuple")

        if not all(
            isinstance(result, ToolResult)
            for result in self.tool_results
        ):
            raise AgentContractError(
                "tool_results must contain only ToolResult values"
            )

        if not isinstance(self.answer, str):
            raise AgentContractError("answer must be a string")

        normalized_answer = self.answer.strip()

        if (
            self.status is AgentStatus.COMPLETED
            and not normalized_answer
        ):
            raise AgentContractError(
                "completed agent responses require an answer"
            )

        if (
            self.status is AgentStatus.ABSTAINED
            and self.decision.decision_type is not DecisionType.ABSTAIN
        ):
            raise AgentContractError(
                "abstained status requires an ABSTAIN decision"
            )

        object.__setattr__(self, "answer", normalized_answer)
        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, "metadata"),
        )
