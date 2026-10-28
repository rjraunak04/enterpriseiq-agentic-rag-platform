"""Tests for EnterpriseIQ retrieval reranking."""

from pathlib import Path

import pytest

from enterpriseiq.retrieval.models import SearchResult
from enterpriseiq.retrieval.reranking import (
    DeterministicReranker,
    RerankedRetriever,
    RerankingConfig,
    RerankingError,
    tokenize_for_reranking,
)


def make_result(
    chunk_id: str,
    *,
    rank: int,
    score: float,
    text: str,
    title: str = "Enterprise Policy",
    department: str = "HR",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        text=text,
        score=score,
        rank=rank,
        title=title,
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
        filters: object | None = None,
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


def test_tokenizer_is_deterministic() -> None:
    text = "P1 response: 15 minutes; INR 1,500."

    assert tokenize_for_reranking(text) == (
        "p1",
        "response",
        "15",
        "minutes",
        "inr",
        "1,500",
    )


def test_reranker_promotes_query_relevant_candidate() -> None:
    candidates = [
        make_result(
            "A::0000",
            rank=1,
            score=0.040,
            text="General employee information.",
        ),
        make_result(
            "B::0000",
            rank=2,
            score=0.039,
            text="Employees may work remotely three days per week.",
        ),
    ]

    reranker = DeterministicReranker()

    results = reranker.rerank(
        "remote work three days",
        candidates,
        top_k=2,
    )

    assert results[0].chunk_id == "B::0000"


def test_reranker_preserves_candidate_provenance() -> None:
    candidate = make_result(
        "FIN-001::0000",
        rank=3,
        score=0.03,
        text="Receipts are required for reimbursement.",
        department="Finance",
    )

    results = DeterministicReranker().rerank(
        "receipt reimbursement",
        [candidate],
        top_k=1,
    )

    assert results[0].document_id == "FIN-001"
    assert results[0].department == "Finance"
    assert results[0].source_path == candidate.source_path
    assert results[0].metadata == candidate.metadata


def test_reranker_reassigns_contiguous_ranks() -> None:
    candidates = [
        make_result(
            "A::0000",
            rank=5,
            score=0.03,
            text="remote work",
        ),
        make_result(
            "B::0000",
            rank=9,
            score=0.02,
            text="travel policy",
        ),
    ]

    results = DeterministicReranker().rerank(
        "remote work",
        candidates,
        top_k=2,
    )

    assert [result.rank for result in results] == [1, 2]


def test_reranker_respects_top_k() -> None:
    candidates = [
        make_result(
            f"DOC-{index}::0000",
            rank=index,
            score=1 / index,
            text=f"policy {index}",
        )
        for index in range(1, 6)
    ]

    results = DeterministicReranker().rerank(
        "policy",
        candidates,
        top_k=2,
    )

    assert len(results) == 2


def test_reranker_returns_empty_for_empty_candidates() -> None:
    assert (
        DeterministicReranker().rerank(
            "remote work",
            [],
            top_k=5,
        )
        == []
    )


def test_reranker_rejects_duplicate_candidates() -> None:
    candidate = make_result(
        "A::0000",
        rank=1,
        score=0.1,
        text="remote work",
    )

    with pytest.raises(
        RerankingError,
        match="duplicate candidate",
    ):
        DeterministicReranker().rerank(
            "remote",
            [candidate, candidate],
            top_k=2,
        )


@pytest.mark.parametrize("query", ["", " ", "\n"])
def test_reranker_rejects_empty_query(query: str) -> None:
    with pytest.raises(
        RerankingError,
        match="non-empty string",
    ):
        DeterministicReranker().rerank(
            query,
            [],
            top_k=5,
        )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_reranker_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(
        RerankingError,
        match="positive integer",
    ):
        DeterministicReranker().rerank(
            "remote",
            [],
            top_k=top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("candidate_weight", "lexical_weight"),
    [
        (-1.0, 1.0),
        (1.0, -1.0),
        (float("nan"), 1.0),
        (1.0, float("inf")),
        (True, 1.0),
        (0.0, 0.0),
    ],
)
def test_config_rejects_invalid_weights(
    candidate_weight: object,
    lexical_weight: object,
) -> None:
    with pytest.raises(RerankingError):
        RerankingConfig(
            candidate_weight=candidate_weight,  # type: ignore[arg-type]
            lexical_weight=lexical_weight,  # type: ignore[arg-type]
        )


def test_reranked_retriever_requests_candidate_pool() -> None:
    backend = FakeRetriever(
        [
            make_result(
                "A::0000",
                rank=1,
                score=0.04,
                text="remote work",
            )
        ]
    )

    retriever = RerankedRetriever(
        retriever=backend,
        reranker=DeterministicReranker(),
        candidate_k=17,
    )

    retriever.search(
        "remote work",
        top_k=5,
        filters="FILTER",
    )

    assert backend.calls[0]["top_k"] == 17
    assert backend.calls[0]["filters"] == "FILTER"


def test_reranked_retriever_applies_min_score_after_reranking() -> None:
    backend = FakeRetriever(
        [
            make_result(
                "A::0000",
                rank=1,
                score=0.04,
                text="unrelated evidence",
            )
        ]
    )

    retriever = RerankedRetriever(
        retriever=backend,
        reranker=DeterministicReranker(
            RerankingConfig(
                candidate_weight=0.0,
                lexical_weight=1.0,
            )
        ),
    )

    assert retriever.search(
        "remote work",
        min_score=0.5,
    ) == []


@pytest.mark.parametrize("candidate_k", [0, -1, True, 1.5])
def test_reranked_retriever_rejects_candidate_k(
    candidate_k: object,
) -> None:
    with pytest.raises(
        RerankingError,
        match="positive integer",
    ):
        RerankedRetriever(
            retriever=FakeRetriever([]),
            reranker=DeterministicReranker(),
            candidate_k=candidate_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "min_score",
    [-1.0, float("nan"), float("inf"), True],
)
def test_reranked_retriever_rejects_min_score(
    min_score: object,
) -> None:
    retriever = RerankedRetriever(
        retriever=FakeRetriever([]),
        reranker=DeterministicReranker(),
    )

    with pytest.raises(
        RerankingError,
        match="finite non-negative",
    ):
        retriever.search(
            "remote",
            min_score=min_score,  # type: ignore[arg-type]
        )
