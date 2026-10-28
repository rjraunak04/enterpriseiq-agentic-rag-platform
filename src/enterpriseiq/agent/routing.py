"""Deterministic routing policy for EnterpriseIQ agentic execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from enterpriseiq.agent.models import (
    AgentDecision,
    AgentQuery,
    DecisionType,
    ToolCall,
)
from enterpriseiq.agent.tools import ToolRegistry


class AgentRoutingError(ValueError):
    """Raised when agent routing configuration or input is invalid."""


@runtime_checkable
class AgentRouter(Protocol):
    """Contract for agent routing implementations."""

    def route(self, query: AgentQuery) -> AgentDecision:
        """Route an agent query to a supported execution path."""
        ...


@dataclass(frozen=True, slots=True)
class RoutingConfig:
    """Configuration for deterministic agent routing."""

    default_top_k: int = 5
    max_top_k: int = 20
    allow_direct_rag: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.default_top_k, bool)
            or not isinstance(self.default_top_k, int)
            or self.default_top_k <= 0
        ):
            raise AgentRoutingError(
                "default_top_k must be a positive integer."
            )

        if (
            isinstance(self.max_top_k, bool)
            or not isinstance(self.max_top_k, int)
            or self.max_top_k <= 0
        ):
            raise AgentRoutingError(
                "max_top_k must be a positive integer."
            )

        if self.default_top_k > self.max_top_k:
            raise AgentRoutingError(
                "default_top_k cannot exceed max_top_k."
            )

        if not isinstance(self.allow_direct_rag, bool):
            raise AgentRoutingError(
                "allow_direct_rag must be a boolean."
            )


def _query_text(query: AgentQuery) -> str:
    """Return normalized text from the canonical AgentQuery contract."""
    text = query.text.strip()

    if not text:
        raise AgentRoutingError(
            "Agent query text cannot be empty."
        )

    return text


def _metadata(query: AgentQuery) -> Mapping[str, Any]:
    """Return validated query metadata."""
    if not isinstance(query.metadata, Mapping):
        raise AgentRoutingError(
            "Agent query metadata must be a mapping."
        )

    return query.metadata


class DeterministicAgentRouter:
    """Safe deterministic routing policy for local and test execution.

    This router intentionally does not perform semantic reasoning.
    It provides predictable routing behavior for tests, local
    development, and controlled fallback execution.

    A production semantic router can implement the AgentRouter
    protocol without changing downstream execution contracts.
    """

    _UNSAFE_TERMS = (
        "delete database",
        "drop database",
        "drop table",
        "shell",
        "command",
        "powershell",
        "cmd.exe",
        "subprocess",
        "os.system",
        "write file",
        "delete file",
        "remove file",
        "modify file",
        "execute python",
        "run python",
    )

    _DESTRUCTIVE_DATABASE_VERBS = (
        "delete",
        "drop",
        "destroy",
        "erase",
        "remove",
        "truncate",
        "wipe",
    )

    _DATABASE_OBJECT_TERMS = (
        "database",
        "table",
        "schema",
    )

    _DESTRUCTIVE_FILE_VERBS = (
        "delete",
        "remove",
        "erase",
        "modify",
        "overwrite",
        "write",
    )

    _FILE_OBJECT_TERMS = (
        "file",
        "files",
    )

    _EXECUTION_VERBS = (
        "execute",
        "run",
        "launch",
        "invoke",
    )

    _EXECUTION_TARGETS = (
        "python",
        "shell",
        "powershell",
        "cmd.exe",
        "subprocess",
        "os.system",
    )

    _LIST_DEPARTMENT_TERMS = (
        "list departments",
        "show departments",
        "available departments",
        "what departments",
    )

    _LIST_DOCUMENT_TERMS = (
        "list documents",
        "show documents",
        "available documents",
        "what documents",
    )

    _GET_DOCUMENT_TERMS = (
        "get document",
        "retrieve document",
        "fetch document",
        "open document",
    )

    _SEARCH_TERMS = (
        "search knowledge",
        "search documents",
        "search enterprise",
        "find knowledge",
        "find documents",
    )

    def __init__(
        self,
        registry: ToolRegistry,
        config: RoutingConfig | None = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise AgentRoutingError(
                "registry must be a ToolRegistry."
            )

        if config is not None and not isinstance(
            config,
            RoutingConfig,
        ):
            raise AgentRoutingError(
                "config must be a RoutingConfig when provided."
            )

        self._registry = registry
        self._config = config or RoutingConfig()

    @property
    def registry(self) -> ToolRegistry:
        """Return the allowlisted tool registry."""
        return self._registry

    @property
    def config(self) -> RoutingConfig:
        """Return the routing configuration."""
        return self._config

    def route(self, query: AgentQuery) -> AgentDecision:
        """Route a query without executing any tool."""
        if not isinstance(query, AgentQuery):
            raise AgentRoutingError(
                "query must be an AgentQuery."
            )

        text = _query_text(query)
        metadata = _metadata(query)
        normalized = " ".join(text.lower().split())

        if self._is_unsafe(normalized):
            return self._abstain(
                "The request requires an unsafe or unsupported "
                "execution capability."
            )

        explicit_tool = metadata.get("tool_name")

        if explicit_tool is not None:
            return self._route_explicit_tool(
                explicit_tool,
                metadata,
            )

        if self._contains_any(
            normalized,
            self._LIST_DEPARTMENT_TERMS,
        ):
            return self._tool_decision(
                "list_departments",
                {},
                (
                    "The query explicitly requests the available "
                    "departments."
                ),
            )

        if self._contains_any(
            normalized,
            self._LIST_DOCUMENT_TERMS,
        ):
            department = metadata.get("department")

            if department is not None:
                if not isinstance(department, str):
                    return self._abstain(
                        "The department filter must be a string."
                    )

                department = department.strip()

                if not department:
                    return self._abstain(
                        "The department filter cannot be blank."
                    )

            return self._tool_decision(
                "list_documents",
                {"department": department},
                (
                    "The query explicitly requests enterprise "
                    "documents."
                ),
            )

        if self._contains_any(
            normalized,
            self._GET_DOCUMENT_TERMS,
        ):
            document_id = metadata.get("document_id")

            if not isinstance(document_id, str):
                return self._abstain(
                    "An exact document lookup requires a document_id."
                )

            document_id = document_id.strip()

            if not document_id:
                return self._abstain(
                    "An exact document lookup requires a non-empty "
                    "document_id."
                )

            return self._tool_decision(
                "get_document",
                {"document_id": document_id},
                (
                    "The query requests an exact enterprise "
                    "document."
                ),
            )

        if self._contains_any(
            normalized,
            self._SEARCH_TERMS,
        ):
            return self._route_search(
                text,
                metadata,
            )

        if self._config.allow_direct_rag:
            return AgentDecision(
                decision_type=DecisionType.DIRECT_RAG,
                reason=(
                    "No explicit tool operation is required; "
                    "use grounded RAG."
                ),
            )

        return self._abstain(
            "Direct grounded RAG is disabled and no supported "
            "tool route matched."
        )

    def _route_explicit_tool(
        self,
        tool_name: object,
        metadata: Mapping[str, Any],
    ) -> AgentDecision:
        """Route an explicitly requested allowlisted tool."""
        if not isinstance(tool_name, str):
            return self._abstain(
                "Explicit tool_name must be a string."
            )

        tool_name = tool_name.strip()

        if not tool_name:
            return self._abstain(
                "Explicit tool_name cannot be blank."
            )

        arguments = metadata.get(
            "tool_arguments",
            {},
        )

        if not isinstance(arguments, Mapping):
            return self._abstain(
                "Explicit tool_arguments must be a mapping."
            )

        return self._tool_decision(
            tool_name,
            dict(arguments),
            (
                "The query explicitly requests an allowlisted "
                "tool."
            ),
        )

    def _route_search(
        self,
        text: str,
        metadata: Mapping[str, Any],
    ) -> AgentDecision:
        """Route an enterprise knowledge search."""
        top_k = metadata.get(
            "top_k",
            self._config.default_top_k,
        )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
            or top_k > self._config.max_top_k
        ):
            return self._abstain(
                "Search top_k must be an integer between 1 and "
                f"{self._config.max_top_k}."
            )

        search_query = metadata.get(
            "search_query",
            text,
        )

        if not isinstance(search_query, str):
            return self._abstain(
                "search_query must be a string."
            )

        search_query = search_query.strip()

        if not search_query:
            return self._abstain(
                "search_query cannot be blank."
            )

        return self._tool_decision(
            "search_knowledge",
            {
                "query": search_query,
                "top_k": top_k,
            },
            (
                "The query explicitly requests enterprise "
                "knowledge search."
            ),
        )

    def _tool_decision(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        reason: str,
    ) -> AgentDecision:
        """Create a validated allowlisted tool-call decision."""
        if not self._registry.contains(tool_name):
            return self._abstain(
                f"Tool '{tool_name}' is not registered or allowlisted."
            )

        try:
            definition = self._registry.get(tool_name)
            validated = definition.validate_arguments(
                arguments
            )
        except Exception as exc:
            return self._abstain(
                f"Tool request failed validation: {exc}"
            )

        return AgentDecision(
            decision_type=DecisionType.TOOL_CALL,
            reason=reason,
            tool_call=ToolCall(
                tool_name=tool_name,
                arguments=dict(validated),
            ),
        )

    @classmethod
    def _is_unsafe(
        cls,
        normalized_text: str,
    ) -> bool:
        """Return whether text requests an unsafe execution capability."""
        if cls._contains_any(
            normalized_text,
            cls._UNSAFE_TERMS,
        ):
            return True

        if (
            cls._contains_any(
                normalized_text,
                cls._DESTRUCTIVE_DATABASE_VERBS,
            )
            and cls._contains_any(
                normalized_text,
                cls._DATABASE_OBJECT_TERMS,
            )
        ):
            return True

        if (
            cls._contains_any(
                normalized_text,
                cls._DESTRUCTIVE_FILE_VERBS,
            )
            and cls._contains_any(
                normalized_text,
                cls._FILE_OBJECT_TERMS,
            )
        ):
            return True

        return (
            cls._contains_any(
                normalized_text,
                cls._EXECUTION_VERBS,
            )
            and cls._contains_any(
                normalized_text,
                cls._EXECUTION_TARGETS,
            )
        )

    @staticmethod
    def _contains_any(
        text: str,
        terms: tuple[str, ...],
    ) -> bool:
        """Return whether any configured phrase occurs in text."""
        return any(
            term in text
            for term in terms
        )

    @staticmethod
    def _abstain(
        reason: str,
    ) -> AgentDecision:
        """Create a deterministic abstention decision."""
        return AgentDecision(
            decision_type=DecisionType.ABSTAIN,
            reason=reason,
        )