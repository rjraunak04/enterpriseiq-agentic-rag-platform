"""Tests for EnterpriseIQ hybrid retrieval and RRF."""

from pathlib import Path

import pytest

from enterpriseiq.retrieval.hybrid import (
    HybridConfig,
    HybridRetrievalError,
    HybridRetriever,
    reciprocal_rank,
    reciprocal_rank_fusion,
)
from enterpriseiq.retrieval.models import SearchResult
from enterpriseiq.retrieval.semantic import SearchFilters


def make_result(
    chunk_id: str,
    *,
    rank: int,
    score: float,
    department: str = "HR",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        text=f"Evidence for {chunk_id}",
        score=score,
        rank=rank,
        title=f"{chunk_id} title",
        department=department,
        source_path=Path(f"data/raw/{chunk_id}.md"),
        metadata={"department": department},
    )


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "min_score": min_score,
            }
        )
        return self.results[:top_k]


def test_reciprocal_rank_formula() -> None:
    assert reciprocal_rank(1, rrf_k=60) == pytest.approx(1 / 61)
    assert reciprocal_rank(5, rrf_k=60) == pytest.approx(1 / 65)


@pytest.mark.parametrize("rank", [0, -1, True])
def test_reciprocal_rank_rejects_invalid_rank(rank: object) -> None:
    with pytest.raises(HybridRetrievalError):
        reciprocal_rank(rank)  # type: ignore[arg-type]


def test_rrf_rewards_result_found_by_both_retrievers() -> None:
    dense = [
        make_result("A::0000", rank=1, score=0.91),
        make_result("B::0000", rank=2, score=0.85),
    ]
    lexical = [
        make_result("B::0000", rank=1, score=8.2),
        make_result("C::0000", rank=2, score=7.1),
    ]

    results = reciprocal_rank_fusion(
        dense,
        lexical,
        rrf_k=60,
        top_k=3,
    )

    assert results[0].chunk_id == "B::0000"
    assert results[0].score == pytest.approx(
        1 / 62 + 1 / 61
    )


def test_rrf_does_not_mix_raw_retrieval_scores() -> None:
    dense = [
        make_result("A::0000", rank=1, score=0.01),
    ]
    lexical = [
        make_result("B::0000", rank=1, score=9999.0),
    ]

    results = reciprocal_rank_fusion(
        dense,
        lexical,
        rrf_k=60,
        top_k=2,
    )

    assert results[0].score == pytest.approx(1 / 61)
    assert results[1].score == pytest.approx(1 / 61)


def test_rrf_ties_are_deterministic_by_chunk_id() -> None:
    dense = [
        make_result("B::0000", rank=1, score=0.9),
    ]
    lexical = [
        make_result("A::0000", rank=1, score=10.0),
    ]

    results = reciprocal_rank_fusion(
        dense,
        lexical,
        top_k=2,
    )

    assert [result.chunk_id for result in results] == [
        "A::0000",
        "B::0000",
    ]


def test_rrf_removes_duplicate_cross_retriever_chunks() -> None:
    dense = [
        make_result("A::0000", rank=1, score=0.9),
    ]
    lexical = [
        make_result("A::0000", rank=1, score=8.0),
    ]

    results = reciprocal_rank_fusion(
        dense,
        lexical,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "A::0000"


def test_rrf_preserves_provenance() -> None:
    dense = [
        make_result(
            "HR-003::0000",
            rank=1,
            score=0.9,
            department="HR",
        ),
    ]

    results = reciprocal_rank_fusion(
        dense,
        [],
        top_k=1,
    )

    assert results[0].document_id == "HR-003"
    assert results[0].department == "HR"
    assert isinstance(results[0].source_path, Path)
    assert results[0].metadata["department"] == "HR"


def test_rrf_reassigns_contiguous_ranks() -> None:
    dense = [
        make_result("A::0000", rank=4, score=0.9),
        make_result("B::0000", rank=8, score=0.8),
    ]

    results = reciprocal_rank_fusion(
        dense,
        [],
        top_k=2,
    )

    assert [result.rank for result in results] == [1, 2]


def test_rrf_rejects_duplicate_dense_chunk_ids() -> None:
    dense = [
        make_result("A::0000", rank=1, score=0.9),
        make_result("A::0000", rank=2, score=0.8),
    ]

    with pytest.raises(
        HybridRetrievalError,
        match="duplicate chunk_id",
    ):
        reciprocal_rank_fusion(dense, [])


@pytest.mark.parametrize(
    ("rrf_k", "candidate_k"),
    [
        (0, 20),
        (-1, 20),
        (True, 20),
        (60, 0),
        (60, -1),
        (60, True),
    ],
)
def test_hybrid_config_rejects_invalid_values(
    rrf_k: object,
    candidate_k: object,
) -> None:
    with pytest.raises(HybridRetrievalError):
        HybridConfig(
            rrf_k=rrf_k,  # type: ignore[arg-type]
            candidate_k=candidate_k,  # type: ignore[arg-type]
        )


def test_hybrid_retriever_combines_both_sources() -> None:
    dense = FakeRetriever(
        [
            make_result("A::0000", rank=1, score=0.9),
            make_result("B::0000", rank=2, score=0.8),
        ]
    )
    lexical = FakeRetriever(
        [
            make_result("B::0000", rank=1, score=9.0),
            make_result("C::0000", rank=2, score=7.0),
        ]
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    results = retriever.search("remote work", top_k=3)

    assert results[0].chunk_id == "B::0000"
    assert {result.chunk_id for result in results} == {
        "A::0000",
        "B::0000",
        "C::0000",
    }


def test_hybrid_passes_candidate_k_and_filters() -> None:
    dense = FakeRetriever([])
    lexical = FakeRetriever([])
    filters = SearchFilters(departments=("HR",))

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
        config=HybridConfig(candidate_k=12),
    )

    retriever.search(
        "remote policy",
        top_k=5,
        filters=filters,
    )

    assert dense.calls[0]["top_k"] == 12
    assert lexical.calls[0]["top_k"] == 12
    assert dense.calls[0]["filters"] == filters
    assert lexical.calls[0]["filters"] == filters


def test_hybrid_min_score_applies_to_fused_score() -> None:
    dense = FakeRetriever(
        [make_result("A::0000", rank=1, score=0.9)]
    )
    lexical = FakeRetriever([])

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    assert retriever.search(
        "remote",
        min_score=0.5,
    ) == []


@pytest.mark.parametrize("query", ["", " ", "\n"])
def test_hybrid_rejects_empty_query(query: str) -> None:
    retriever = HybridRetriever(
        dense_retriever=FakeRetriever([]),
        lexical_retriever=FakeRetriever([]),
    )

    with pytest.raises(
        HybridRetrievalError,
        match="non-empty string",
    ):
        retriever.search(query)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_hybrid_rejects_invalid_top_k(top_k: object) -> None:
    retriever = HybridRetriever(
        dense_retriever=FakeRetriever([]),
        lexical_retriever=FakeRetriever([]),
    )

    with pytest.raises(
        HybridRetrievalError,
        match="positive integer",
    ):
        retriever.search(
            "remote",
            top_k=top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "min_score",
    [-1.0, float("nan"), float("inf"), True],
)
def test_hybrid_rejects_invalid_min_score(
    min_score: object,
) -> None:
    retriever = HybridRetriever(
        dense_retriever=FakeRetriever([]),
        lexical_retriever=FakeRetriever([]),
    )

    with pytest.raises(
        HybridRetrievalError,
        match="finite non-negative",
    ):
        retriever.search(
            "remote",
            min_score=min_score,  # type: ignore[arg-type]
        )
