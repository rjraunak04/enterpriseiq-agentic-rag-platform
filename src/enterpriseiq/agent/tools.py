"""Typed and allowlisted tool registry for EnterpriseIQ agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from enterpriseiq.agent.models import AgentContractError


class ToolRegistryError(AgentContractError):
    """Base error for tool registration and discovery failures."""


class ToolRegistrationError(ToolRegistryError):
    """Raised when a tool cannot be registered."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when an unregistered tool is requested."""


class ToolInputError(AgentContractError):
    """Raised when tool arguments violate their declared schema."""


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolRegistrationError(
            f"{field_name} must be a non-empty string"
        )
    return value.strip()


def _immutable_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolRegistrationError(f"{field_name} must be a mapping")

    normalized: dict[str, Any] = {}

    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ToolRegistrationError(
                f"{field_name} keys must be non-empty strings"
            )
        normalized[key] = item

    return MappingProxyType(normalized)


def _matches_type(value: Any, expected: type) -> bool:
    if expected is int and isinstance(value, bool):
        return False

    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(
            value, bool
        )

    return isinstance(value, expected)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Immutable runtime context supplied to a tool."""

    query_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.query_id is not None:
            if (
                not isinstance(self.query_id, str)
                or not self.query_id.strip()
            ):
                raise ToolInputError(
                    "query_id must be a non-empty string when provided"
                )

            object.__setattr__(
                self,
                "query_id",
                self.query_id.strip(),
            )

        if not isinstance(self.metadata, Mapping):
            raise ToolInputError("metadata must be a mapping")

        normalized: dict[str, Any] = {}

        for key, value in self.metadata.items():
            if not isinstance(key, str) or not key.strip():
                raise ToolInputError(
                    "metadata keys must be non-empty strings"
                )
            normalized[key] = value

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(normalized),
        )


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """Declarative input parameter for a registered tool."""

    name: str
    value_type: type
    description: str
    required: bool = True
    default: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _required_text(self.name, "parameter name"),
        )
        object.__setattr__(
            self,
            "description",
            _required_text(
                self.description,
                "parameter description",
            ),
        )

        if not isinstance(self.value_type, type):
            raise ToolRegistrationError(
                "value_type must be a Python type"
            )

        if not isinstance(self.required, bool):
            raise ToolRegistrationError(
                "required must be a boolean"
            )

        if (
            not self.required
            and self.default is not None
            and not _matches_type(self.default, self.value_type)
        ):
            raise ToolRegistrationError(
                f"default for {self.name!r} must match "
                f"{self.value_type.__name__}"
            )


@runtime_checkable
class ToolHandler(Protocol):
    """Callable contract implemented by registered tools."""

    def __call__(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> Any:
        """Execute using already validated arguments."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Immutable definition of an allowlisted enterprise tool."""

    name: str
    description: str
    handler: ToolHandler
    parameters: tuple[ToolParameter, ...] = ()
    read_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _required_text(self.name, "tool name"),
        )
        object.__setattr__(
            self,
            "description",
            _required_text(
                self.description,
                "tool description",
            ),
        )

        if not callable(self.handler):
            raise ToolRegistrationError(
                "tool handler must be callable"
            )

        if not isinstance(self.parameters, tuple):
            raise ToolRegistrationError(
                "parameters must be a tuple"
            )

        if not all(
            isinstance(parameter, ToolParameter)
            for parameter in self.parameters
        ):
            raise ToolRegistrationError(
                "parameters must contain only ToolParameter values"
            )

        names = [parameter.name for parameter in self.parameters]

        if len(names) != len(set(names)):
            raise ToolRegistrationError(
                "tool parameter names must be unique"
            )

        if not isinstance(self.read_only, bool):
            raise ToolRegistrationError(
                "read_only must be a boolean"
            )

        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, "metadata"),
        )

    def validate_arguments(
        self,
        arguments: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Validate arguments against the declared schema."""

        if not isinstance(arguments, Mapping):
            raise ToolInputError(
                "tool arguments must be a mapping"
            )

        supplied = set(arguments)
        declared = {
            parameter.name for parameter in self.parameters
        }

        unknown = sorted(supplied - declared)

        if unknown:
            raise ToolInputError(
                "unknown tool arguments: " + ", ".join(unknown)
            )

        normalized: dict[str, Any] = {}

        for parameter in self.parameters:
            if parameter.name in arguments:
                value = arguments[parameter.name]
            elif parameter.required:
                raise ToolInputError(
                    f"missing required argument: {parameter.name}"
                )
            else:
                value = parameter.default

            if value is not None and not _matches_type(
                value,
                parameter.value_type,
            ):
                raise ToolInputError(
                    f"argument {parameter.name!r} must be "
                    f"{parameter.value_type.__name__}"
                )

            normalized[parameter.name] = value

        return MappingProxyType(normalized)


class ToolRegistry:
    """Deterministic registry containing explicitly allowed tools."""

    def __init__(
        self,
        tools: Sequence[ToolDefinition] = (),
    ) -> None:
        if isinstance(tools, (str, bytes)) or not isinstance(
            tools,
            Sequence,
        ):
            raise ToolRegistrationError(
                "tools must be a sequence of ToolDefinition values"
            )

        self._tools: dict[str, ToolDefinition] = {}

        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolDefinition) -> None:
        """Register one explicitly allowed tool."""

        if not isinstance(tool, ToolDefinition):
            raise ToolRegistrationError(
                "tool must be a ToolDefinition"
            )

        if tool.name in self._tools:
            raise ToolRegistrationError(
                f"tool {tool.name!r} is already registered"
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        """Resolve a registered tool."""

        if not isinstance(name, str) or not name.strip():
            raise ToolNotFoundError(
                "tool name must be a non-empty string"
            )

        normalized = name.strip()

        try:
            return self._tools[normalized]
        except KeyError as exc:
            raise ToolNotFoundError(
                f"tool {normalized!r} is not registered"
            ) from exc

    def contains(self, name: str) -> bool:
        """Return whether a tool is explicitly registered."""

        if not isinstance(name, str) or not name.strip():
            return False

        return name.strip() in self._tools

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        """Return tools in deterministic name order."""

        return tuple(
            self._tools[name]
            for name in sorted(self._tools)
        )

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""

        return tuple(
            tool.name for tool in self.list_tools()
        )

    def __len__(self) -> int:
        return len(self._tools)