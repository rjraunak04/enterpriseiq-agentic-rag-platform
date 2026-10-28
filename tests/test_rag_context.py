"""Tests for deterministic grounded evidence context assembly."""

from pathlib import Path

import pytest

from enterpriseiq.rag import (
    ContextAssemblyError,
    ContextConfig,
    EvidenceContextBuilder,
)
from enterpriseiq.retrieval.models import SearchResult


def _result(
    chunk_id: str,
    *,
    rank: int,
    score: float = 0.9,
    text: str = "Employees may work remotely.",
    document_id: str = "HR-003",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=score,
        rank=rank,
        title="Remote Work Policy",
        department="HR",
        source_path=Path("hr/remote_work_policy.md"),
        metadata={"classification": "internal"},
    )


def test_default_config_is_production_bounded() -> None:
    config = ContextConfig()

    assert config.max_chunks == 8
    assert config.max_chars == 12000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_chunks", 0),
        ("max_chunks", -1),
        ("max_chunks", True),
        ("max_chars", 0),
        ("max_chars", -1),
        ("max_chars", False),
    ],
)
def test_config_rejects_invalid_limits(
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}

    with pytest.raises(ContextAssemblyError):
        ContextConfig(**kwargs)


def test_empty_results_produce_empty_context() -> None:
    context = EvidenceContextBuilder().build([])

    assert context.items == ()
    assert context.rendered_text == ""
    assert context.truncated is False


def test_builder_assigns_stable_source_ids() -> None:
    context = EvidenceContextBuilder().build(
        [
            _result("chunk-1", rank=1),
            _result("chunk-2", rank=2),
        ]
    )

    assert [item.source_id for item in context.items] == [
        "S1",
        "S2",
    ]


def test_builder_orders_by_retrieval_rank() -> None:
    context = EvidenceContextBuilder().build(
        [
            _result("chunk-2", rank=2),
            _result("chunk-1", rank=1),
        ]
    )

    assert [item.chunk_id for item in context.items] == [
        "chunk-1",
        "chunk-2",
    ]


def test_rank_ties_are_deterministic() -> None:
    context = EvidenceContextBuilder().build(
        [
            _result("chunk-b", rank=1),
            _result("chunk-a", rank=1),
        ]
    )

    assert [item.chunk_id for item in context.items] == [
        "chunk-a",
        "chunk-b",
    ]


def test_builder_preserves_retrieval_provenance() -> None:
    result = _result(
        "chunk-1",
        rank=3,
        score=0.812,
    )

    item = EvidenceContextBuilder().build([result]).items[0]

    assert item.document_id == result.document_id
    assert item.source_path == result.source_path
    assert item.retrieval_rank == 3
    assert item.retrieval_score == pytest.approx(0.812)
    assert item.metadata["classification"] == "internal"


def test_rendered_context_contains_source_label() -> None:
    context = EvidenceContextBuilder().build(
        [_result("chunk-1", rank=1)]
    )

    assert "[S1]" in context.rendered_text
    assert "Title: Remote Work Policy" in context.rendered_text
    assert "Document: HR-003" in context.rendered_text
    assert "Department: HR" in context.rendered_text
    assert "Chunk: chunk-1" in context.rendered_text
    assert "Evidence: Employees may work remotely." in (
        context.rendered_text
    )


def test_builder_enforces_max_chunks() -> None:
    builder = EvidenceContextBuilder(
        ContextConfig(max_chunks=2, max_chars=12000)
    )

    context = builder.build(
        [
            _result("chunk-1", rank=1),
            _result("chunk-2", rank=2),
            _result("chunk-3", rank=3),
        ]
    )

    assert len(context.items) == 2
    assert context.truncated is True


def test_builder_enforces_character_budget() -> None:
    first = _result(
        "chunk-1",
        rank=1,
        text="A" * 50,
    )
    second = _result(
        "chunk-2",
        rank=2,
        text="B" * 50,
    )

    single_context = EvidenceContextBuilder().build([first])
    budget = len(single_context.rendered_text)

    context = EvidenceContextBuilder(
        ContextConfig(max_chunks=8, max_chars=budget)
    ).build([first, second])

    assert len(context.items) == 1
    assert context.items[0].chunk_id == "chunk-1"
    assert context.truncated is True
    assert len(context.rendered_text) <= budget


def test_too_small_budget_returns_empty_truncated_context() -> None:
    context = EvidenceContextBuilder(
        ContextConfig(max_chunks=8, max_chars=1)
    ).build([_result("chunk-1", rank=1)])

    assert context.items == ()
    assert context.rendered_text == ""
    assert context.truncated is True


def test_duplicate_chunk_ids_are_rejected() -> None:
    with pytest.raises(
        ContextAssemblyError,
        match="duplicate chunk_id",
    ):
        EvidenceContextBuilder().build(
            [
                _result("chunk-1", rank=1),
                _result("chunk-1", rank=2),
            ]
        )


def test_non_search_result_is_rejected() -> None:
    with pytest.raises(
        ContextAssemblyError,
        match="SearchResult",
    ):
        EvidenceContextBuilder().build(["invalid"])


def test_string_input_is_rejected() -> None:
    with pytest.raises(ContextAssemblyError):
        EvidenceContextBuilder().build("invalid")


def test_repeated_builds_are_deterministic() -> None:
    results = [
        _result("chunk-2", rank=2),
        _result("chunk-1", rank=1),
    ]
    builder = EvidenceContextBuilder()

    first = builder.build(results)
    second = builder.build(results)

    assert first == second
    assert first.rendered_text == second.rendered_text


def test_source_ids_follow_final_bounded_order() -> None:
    builder = EvidenceContextBuilder(
        ContextConfig(max_chunks=2, max_chars=12000)
    )

    context = builder.build(
        [
            _result("chunk-c", rank=3),
            _result("chunk-a", rank=1),
            _result("chunk-b", rank=2),
        ]
    )

    assert [
        (item.source_id, item.chunk_id)
        for item in context.items
    ] == [
        ("S1", "chunk-a"),
        ("S2", "chunk-b"),
    ]
