"""Read-only enterprise knowledge tools for agentic RAG."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from enterpriseiq.agent.tools import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
)


class KnowledgeToolError(ValueError):
    """Raised when an enterprise knowledge tool cannot serve a request."""


class KnowledgeRetriever(Protocol):
    """Minimal retrieval contract required by knowledge tools."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> Sequence[object]:
        """Return ranked enterprise knowledge results."""


def _read_attr(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _serialize_search_result(result: object) -> dict[str, Any]:
    fields = (
        "chunk_id",
        "document_id",
        "title",
        "text",
        "score",
        "rank",
        "department",
        "document_type",
        "classification",
        "source_path",
        "metadata",
    )

    payload: dict[str, Any] = {}

    for field_name in fields:
        value = _read_attr(result, field_name)
        if value is not None:
            payload[field_name] = _serialize_value(value)

    return payload


@dataclass(frozen=True, slots=True)
class EnterpriseKnowledgeTools:
    """Read-only tools over indexed enterprise knowledge."""

    documents: tuple[object, ...]
    retriever: KnowledgeRetriever

    def __post_init__(self) -> None:
        if not isinstance(self.documents, tuple):
            raise KnowledgeToolError("documents must be a tuple")

        document_ids: list[str] = []

        for document in self.documents:
            document_id = _read_attr(document, "document_id")

            if not isinstance(document_id, str) or not document_id.strip():
                raise KnowledgeToolError(
                    "every document must have a non-empty document_id"
                )

            document_ids.append(document_id.strip())

        if len(document_ids) != len(set(document_ids)):
            raise KnowledgeToolError(
                "document_id values must be unique"
            )

        if not hasattr(self.retriever, "search"):
            raise KnowledgeToolError(
                "retriever must provide a search method"
            )

    def search_knowledge(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> object:
        """Search indexed enterprise knowledge."""

        del context

        query = arguments["query"]
        top_k = arguments["top_k"]

        if not isinstance(query, str) or not query.strip():
            raise KnowledgeToolError(
                "query must be a non-empty string"
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
            or top_k > 20
        ):
            raise KnowledgeToolError(
                "top_k must be between 1 and 20"
            )

        results = self.retriever.search(
            query.strip(),
            top_k=top_k,
        )

        return {
            "query": query.strip(),
            "count": len(results),
            "results": [
                _serialize_search_result(result)
                for result in results
            ],
        }

    def get_document(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> object:
        """Return one enterprise document by exact ID."""

        del context

        requested = arguments["document_id"]

        if not isinstance(requested, str) or not requested.strip():
            raise KnowledgeToolError(
                "document_id must be a non-empty string"
            )

        requested = requested.strip()

        for document in self.documents:
            document_id = _read_attr(document, "document_id")

            if document_id == requested:
                return self._serialize_document(document)

        raise KnowledgeToolError(
            f"document {requested!r} was not found"
        )

    def list_documents(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> object:
        """List document metadata with optional department filtering."""

        del context

        department = arguments["department"]

        if department is not None:
            if not isinstance(department, str) or not department.strip():
                raise KnowledgeToolError(
                    "department must be non-empty when provided"
                )
            department = department.strip()

        documents = [
            self._document_summary(document)
            for document in self.documents
            if department is None
            or _read_attr(document, "department") == department
        ]

        documents.sort(
            key=lambda item: (
                str(item.get("department", "")),
                str(item.get("document_id", "")),
            )
        )

        return {
            "department": department,
            "count": len(documents),
            "documents": documents,
        }

    def list_departments(
        self,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> object:
        """List departments represented in the enterprise corpus."""

        del arguments
        del context

        departments = sorted(
            {
                department
                for document in self.documents
                if isinstance(
                    (
                        department := _read_attr(
                            document,
                            "department",
                        )
                    ),
                    str,
                )
                and department.strip()
            }
        )

        return {
            "count": len(departments),
            "departments": departments,
        }

    @staticmethod
    def _document_summary(document: object) -> dict[str, Any]:
        fields = (
            "document_id",
            "title",
            "department",
            "document_type",
            "version",
            "effective_date",
            "owner",
            "classification",
            "source_path",
        )

        return {
            field_name: _serialize_value(value)
            for field_name in fields
            if (
                value := _read_attr(document, field_name)
            )
            is not None
        }

    @classmethod
    def _serialize_document(
        cls,
        document: object,
    ) -> dict[str, Any]:
        payload = cls._document_summary(document)

        text = _read_attr(document, "text")
        if text is not None:
            payload["text"] = _serialize_value(text)

        content = _read_attr(document, "content")
        if content is not None and "text" not in payload:
            payload["content"] = _serialize_value(content)

        metadata = _read_attr(document, "metadata")
        if metadata is not None:
            payload["metadata"] = _serialize_value(metadata)

        return payload

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Build allowlisted read-only tool definitions."""

        return (
            ToolDefinition(
                name="search_knowledge",
                description=(
                    "Search the indexed enterprise knowledge base "
                    "for relevant evidence."
                ),
                handler=self.search_knowledge,
                parameters=(
                    ToolParameter(
                        name="query",
                        value_type=str,
                        description="Enterprise knowledge query.",
                    ),
                    ToolParameter(
                        name="top_k",
                        value_type=int,
                        description="Maximum number of results.",
                        required=False,
                        default=5,
                    ),
                ),
                read_only=True,
                metadata={"category": "knowledge"},
            ),
            ToolDefinition(
                name="get_document",
                description=(
                    "Retrieve one enterprise document by exact "
                    "document identifier."
                ),
                handler=self.get_document,
                parameters=(
                    ToolParameter(
                        name="document_id",
                        value_type=str,
                        description="Exact enterprise document ID.",
                    ),
                ),
                read_only=True,
                metadata={"category": "knowledge"},
            ),
            ToolDefinition(
                name="list_documents",
                description=(
                    "List available enterprise documents and metadata."
                ),
                handler=self.list_documents,
                parameters=(
                    ToolParameter(
                        name="department",
                        value_type=str,
                        description="Optional department filter.",
                        required=False,
                        default=None,
                    ),
                ),
                read_only=True,
                metadata={"category": "knowledge"},
            ),
            ToolDefinition(
                name="list_departments",
                description=(
                    "List departments represented in enterprise knowledge."
                ),
                handler=self.list_departments,
                read_only=True,
                metadata={"category": "knowledge"},
            ),
        )

    def registry(self) -> ToolRegistry:
        """Create the explicit read-only enterprise tool registry."""

        return ToolRegistry(self.definitions())