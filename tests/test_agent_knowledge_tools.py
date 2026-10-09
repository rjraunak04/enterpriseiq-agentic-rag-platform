"""Tests for read-only enterprise knowledge tools."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from enterpriseiq.agent import (
    EnterpriseKnowledgeTools,
    KnowledgeToolError,
    ToolContext,
)


@dataclass(frozen=True)
class FakeDocument:
    document_id: str
    title: str
    department: str
    document_type: str = "policy"
    version: str = "1.0"
    classification: str = "internal"
    text: str = "Enterprise policy content."
    source_path: Path = Path("data/raw/example.md")


@dataclass(frozen=True)
class FakeResult:
    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float
    rank: int
    department: str
    source_path: Path


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> tuple[FakeResult, ...]:
        self.calls.append((query, top_k))

        return (
            FakeResult(
                chunk_id="remote-work-001",
                document_id="remote-work",
                title="Remote Work Policy",
                text="Employees may work remotely.",
                score=0.91,
                rank=1,
                department="HR",
                source_path=Path(
                    "data/raw/hr/remote_work_policy.md"
                ),
            ),
        )


@pytest.fixture
def documents() -> tuple[FakeDocument, ...]:
    return (
        FakeDocument(
            document_id="remote-work",
            title="Remote Work Policy",
            department="HR",
        ),
        FakeDocument(
            document_id="expense-policy",
            title="Expense Reimbursement Policy",
            department="Finance",
        ),
        FakeDocument(
            document_id="security-policy",
            title="Information Security Policy",
            department="Security",
        ),
    )


@pytest.fixture
def retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture
def tools(
    documents: tuple[FakeDocument, ...],
    retriever: FakeRetriever,
) -> EnterpriseKnowledgeTools:
    return EnterpriseKnowledgeTools(
        documents=documents,
        retriever=retriever,
    )


def test_requires_tuple_documents(
    retriever: FakeRetriever,
) -> None:
    with pytest.raises(KnowledgeToolError):
        EnterpriseKnowledgeTools(
            documents=[],  # type: ignore[arg-type]
            retriever=retriever,
        )


def test_rejects_duplicate_document_ids(
    retriever: FakeRetriever,
) -> None:
    documents = (
        FakeDocument("same", "One", "HR"),
        FakeDocument("same", "Two", "Finance"),
    )

    with pytest.raises(KnowledgeToolError):
        EnterpriseKnowledgeTools(
            documents=documents,
            retriever=retriever,
        )


def test_search_knowledge(
    tools: EnterpriseKnowledgeTools,
    retriever: FakeRetriever,
) -> None:
    result = tools.search_knowledge(
        {"query": " remote work ", "top_k": 3},
        ToolContext(query_id="q-1"),
    )

    assert retriever.calls == [("remote work", 3)]
    assert result["count"] == 1
    assert result["results"][0]["document_id"] == "remote-work"
    assert result["results"][0]["score"] == 0.91


@pytest.mark.parametrize("top_k", [0, 21, True])
def test_search_rejects_invalid_top_k(
    tools: EnterpriseKnowledgeTools,
    top_k: object,
) -> None:
    with pytest.raises(KnowledgeToolError):
        tools.search_knowledge(
            {
                "query": "remote work",
                "top_k": top_k,
            },
            ToolContext(),
        )


def test_search_rejects_blank_query(
    tools: EnterpriseKnowledgeTools,
) -> None:
    with pytest.raises(KnowledgeToolError):
        tools.search_knowledge(
            {"query": " ", "top_k": 5},
            ToolContext(),
        )


def test_search_serializes_paths(
    tools: EnterpriseKnowledgeTools,
) -> None:
    result = tools.search_knowledge(
        {"query": "remote", "top_k": 5},
        ToolContext(),
    )

    assert isinstance(
        result["results"][0]["source_path"],
        str,
    )


def test_get_document(
    tools: EnterpriseKnowledgeTools,
) -> None:
    result = tools.get_document(
        {"document_id": "remote-work"},
        ToolContext(),
    )

    assert result["document_id"] == "remote-work"
    assert result["department"] == "HR"
    assert result["text"] == "Enterprise policy content."


def test_get_document_rejects_unknown_id(
    tools: EnterpriseKnowledgeTools,
) -> None:
    with pytest.raises(KnowledgeToolError):
        tools.get_document(
            {"document_id": "missing"},
            ToolContext(),
        )


def test_get_document_rejects_blank_id(
    tools: EnterpriseKnowledgeTools,
) -> None:
    with pytest.raises(KnowledgeToolError):
        tools.get_document(
            {"document_id": " "},
            ToolContext(),
        )


def test_list_documents(
    tools: EnterpriseKnowledgeTools,
) -> None:
    result = tools.list_documents(
        {"department": None},
        ToolContext(),
    )

    assert result["count"] == 3
    assert [
        item["department"]
        for item in result["documents"]
    ] == ["Finance", "HR", "Security"]


def test_list_documents_filters_department(
    tools: EnterpriseKnowledgeTools,
) -> None:
    result = tools.list_documents(
        {"department": "HR"},
        ToolContext(),
    )

    assert result["count"] == 1
    assert result["documents"][0]["document_id"] == "remote-work"


def test_list_documents_unknown_department_returns_empty(
    tools: EnterpriseKnowledgeTools,
) -> None:
    result = tools.list_documents(
        {"department": "Legal"},
        ToolContext(),
    )

    assert result["count"] == 0
    assert result["documents"] == []


def test_list_departments(
    tools: EnterpriseKnowledgeTools,
) -> None:
    result = tools.list_departments(
        {},
        ToolContext(),
    )

    assert result == {
        "count": 3,
        "departments": ["Finance", "HR", "Security"],
    }


def test_definitions_are_read_only(
    tools: EnterpriseKnowledgeTools,
) -> None:
    definitions = tools.definitions()

    assert len(definitions) == 4
    assert all(tool.read_only for tool in definitions)


def test_expected_tools_are_registered(
    tools: EnterpriseKnowledgeTools,
) -> None:
    registry = tools.registry()

    assert registry.names() == (
        "get_document",
        "list_departments",
        "list_documents",
        "search_knowledge",
    )


def test_registry_does_not_expose_dangerous_tools(
    tools: EnterpriseKnowledgeTools,
) -> None:
    registry = tools.registry()

    assert not registry.contains("shell")
    assert not registry.contains("python")
    assert not registry.contains("delete_document")
    assert not registry.contains("write_file")


def test_registry_validates_search_arguments(
    tools: EnterpriseKnowledgeTools,
) -> None:
    definition = tools.registry().get("search_knowledge")

    arguments = definition.validate_arguments(
        {"query": "remote work"}
    )

    assert arguments == {
        "query": "remote work",
        "top_k": 5,
    }


def test_registry_validates_optional_department(
    tools: EnterpriseKnowledgeTools,
) -> None:
    definition = tools.registry().get("list_documents")

    arguments = definition.validate_arguments({})

    assert arguments == {"department": None}


def test_registered_search_handler_executes(
    tools: EnterpriseKnowledgeTools,
) -> None:
    definition = tools.registry().get("search_knowledge")

    arguments = definition.validate_arguments(
        {
            "query": "remote work",
            "top_k": 2,
        }
    )

    result = definition.handler(
        arguments,
        ToolContext(query_id="q-2"),
    )

    assert result["count"] == 1
    assert result["query"] == "remote work"


def test_tool_definitions_are_deterministic(
    tools: EnterpriseKnowledgeTools,
) -> None:
    first = tuple(
        definition.name
        for definition in tools.definitions()
    )
    second = tuple(
        definition.name
        for definition in tools.definitions()
    )

    assert first == second