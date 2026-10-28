"""Controlled tool execution for EnterpriseIQ agentic workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from enterpriseiq.agent.models import (
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
)
from enterpriseiq.agent.tools import (
    ToolContext,
    ToolInputError,
    ToolNotFoundError,
    ToolRegistry,
)


class ToolExecutionError(RuntimeError):
    """Base error for controlled tool execution failures."""


class ToolExecutionRejectedError(ToolExecutionError):
    """Raised when execution is rejected by the safety policy."""


@dataclass(frozen=True, slots=True)
class ToolExecutionConfig:
    """Safety policy for enterprise tool execution."""

    require_read_only: bool = True
    expose_error_details: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.require_read_only, bool):
            raise ToolExecutionError(
                "require_read_only must be a boolean."
            )

        if not isinstance(self.expose_error_details, bool):
            raise ToolExecutionError(
                "expose_error_details must be a boolean."
            )


class SafeToolExecutor:
    """Execute explicitly allowlisted tools under a controlled policy.

    The executor performs no dynamic imports, shell execution, arbitrary
    Python evaluation, or implicit tool discovery. Every executable tool
    must already exist in the supplied ToolRegistry.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: ToolExecutionConfig | None = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise ToolExecutionError(
                "registry must be a ToolRegistry."
            )

        if config is not None and not isinstance(
            config,
            ToolExecutionConfig,
        ):
            raise ToolExecutionError(
                "config must be a ToolExecutionConfig when provided."
            )

        self._registry = registry
        self._config = config or ToolExecutionConfig()

    @property
    def registry(self) -> ToolRegistry:
        """Return the executor's allowlisted tool registry."""
        return self._registry

    @property
    def config(self) -> ToolExecutionConfig:
        """Return the active execution safety configuration."""
        return self._config

    def execute(
        self,
        call: ToolCall,
        *,
        context: ToolContext | None = None,
    ) -> ToolResult:
        """Execute one validated allowlisted tool call."""
        if not isinstance(call, ToolCall):
            raise ToolExecutionError(
                "call must be a ToolCall."
            )

        if context is None:
            context = ToolContext()
        elif not isinstance(context, ToolContext):
            raise ToolExecutionError(
                "context must be a ToolContext when provided."
            )

        tool_name = call.tool_name

        if not self._registry.contains(tool_name):
            return self._result(
                call,
                status=ToolExecutionStatus.REJECTED,
                error=(
                    f"Tool '{tool_name}' is not registered or "
                    "allowlisted."
                ),
            )

        try:
            definition = self._registry.get(tool_name)
        except ToolNotFoundError:
            return self._result(
                call,
                status=ToolExecutionStatus.REJECTED,
                error=(
                    f"Tool '{tool_name}' is not registered or "
                    "allowlisted."
                ),
            )

        if (
            self._config.require_read_only
            and not definition.read_only
        ):
            return self._result(
                call,
                status=ToolExecutionStatus.REJECTED,
                error=(
                    f"Tool '{tool_name}' is not permitted because "
                    "read-only execution is required."
                ),
            )

        try:
            arguments = definition.validate_arguments(
                call.arguments
            )
        except ToolInputError as exc:
            return self._result(
                call,
                status=ToolExecutionStatus.REJECTED,
                error=str(exc),
            )

        try:
            output = definition.handler(
                context=context,
                **dict(arguments),
            )
        except Exception as exc:
            return self._result(
                call,
                status=ToolExecutionStatus.FAILED,
                error=self._execution_error_message(
                    tool_name,
                    exc,
                ),
            )

        return self._result(
            call,
            status=ToolExecutionStatus.SUCCESS,
            output=output,
        )

    def execute_many(
        self,
        calls: tuple[ToolCall, ...],
        *,
        context: ToolContext | None = None,
    ) -> tuple[ToolResult, ...]:
        """Execute tool calls sequentially in deterministic order."""
        if not isinstance(calls, tuple):
            raise ToolExecutionError(
                "calls must be a tuple of ToolCall values."
            )

        if context is None:
            context = ToolContext()
        elif not isinstance(context, ToolContext):
            raise ToolExecutionError(
                "context must be a ToolContext when provided."
            )

        for call in calls:
            if not isinstance(call, ToolCall):
                raise ToolExecutionError(
                    "calls must contain only ToolCall values."
                )

        return tuple(
            self.execute(
                call,
                context=context,
            )
            for call in calls
        )

    def _execution_error_message(
        self,
        tool_name: str,
        exc: Exception,
    ) -> str:
        """Return a safe operational error message."""
        if self._config.expose_error_details:
            detail = str(exc).strip()

            if detail:
                return (
                    f"Tool '{tool_name}' failed: {detail}"
                )

        return f"Tool '{tool_name}' failed during execution."

    @staticmethod
    def _result(
        call: ToolCall,
        *,
        status: ToolExecutionStatus,
        output: Any = None,
        error: str | None = None,
    ) -> ToolResult:
        """Build a normalized ToolResult from one execution attempt."""
        return ToolResult(
            tool_name=call.tool_name,
            status=status,
            output=output,
            call_id=call.call_id,
            error=error,
            metadata={
                "read_only_execution": True,
            },
        )


def execute_tool(
    registry: ToolRegistry,
    call: ToolCall,
    *,
    context: ToolContext | None = None,
    config: ToolExecutionConfig | None = None,
) -> ToolResult:
    """Convenience wrapper for one controlled tool execution."""
    executor = SafeToolExecutor(
        registry,
        config=config,
    )

    return executor.execute(
        call,
        context=context,
    )