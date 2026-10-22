"""Hybrid dense + lexical retrieval using Reciprocal Rank Fusion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from enterpriseiq.retrieval.models import SearchResult
from enterpriseiq.retrieval.semantic import SearchFilters


class HybridRetrievalError(ValueError):
    """Raised when hybrid retrieval configuration or input is invalid."""


@runtime_checkable
class Retriever(Protocol):
    """Common interface implemented by EnterpriseIQ retrievers."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Retrieve ranked enterprise evidence."""
        ...


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Configuration for reciprocal-rank hybrid retrieval."""

    rrf_k: int = 60
    candidate_k: int = 20

    def __post_init__(self) -> None:
        if (
            isinstance(self.rrf_k, bool)
            or not isinstance(self.rrf_k, int)
            or self.rrf_k <= 0
        ):
            raise HybridRetrievalError(
                "rrf_k must be a positive integer"
            )

        if (
            isinstance(self.candidate_k, bool)
            or not isinstance(self.candidate_k, int)
            or self.candidate_k <= 0
        ):
            raise HybridRetrievalError(
                "candidate_k must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """Internal representation of a fused retrieval candidate."""

    result: SearchResult
    rrf_score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


def reciprocal_rank(rank: int, *, rrf_k: int = 60) -> float:
    """Calculate a single Reciprocal Rank Fusion contribution."""

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise HybridRetrievalError("rank must be a positive integer")

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise HybridRetrievalError("rrf_k must be a positive integer")

    return 1.0 / (rrf_k + rank)


def _validate_results(
    results: Sequence[SearchResult],
    *,
    source: str,
) -> None:
    seen: set[str] = set()

    for result in results:
        if result.chunk_id in seen:
            raise HybridRetrievalError(
                f"duplicate chunk_id returned by {source} retriever: "
                f"{result.chunk_id}"
            )
        seen.add(result.chunk_id)


def reciprocal_rank_fusion(
    dense_results: Sequence[SearchResult],
    lexical_results: Sequence[SearchResult],
    *,
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[SearchResult]:
    """Fuse dense and lexical rankings using deterministic RRF."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise HybridRetrievalError("top_k must be a positive integer")

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise HybridRetrievalError("rrf_k must be a positive integer")

    _validate_results(dense_results, source="dense")
    _validate_results(lexical_results, source="lexical")

    candidates: dict[str, FusedCandidate] = {}

    for result in dense_results:
        contribution = reciprocal_rank(result.rank, rrf_k=rrf_k)
        candidates[result.chunk_id] = FusedCandidate(
            result=result,
            rrf_score=contribution,
            dense_rank=result.rank,
        )

    for result in lexical_results:
        contribution = reciprocal_rank(result.rank, rrf_k=rrf_k)
        existing = candidates.get(result.chunk_id)

        if existing is None:
            candidates[result.chunk_id] = FusedCandidate(
                result=result,
                rrf_score=contribution,
                lexical_rank=result.rank,
            )
            continue

        candidates[result.chunk_id] = FusedCandidate(
            result=existing.result,
            rrf_score=existing.rrf_score + contribution,
            dense_rank=existing.dense_rank,
            lexical_rank=result.rank,
        )

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.rrf_score,
            candidate.result.chunk_id,
        ),
    )

    return [
        replace(
            candidate.result,
            rank=rank,
            score=candidate.rrf_score,
        )
        for rank, candidate in enumerate(ranked[:top_k], start=1)
    ]


class HybridRetriever:
    """Combine semantic and lexical retrieval with RRF."""

    def __init__(
        self,
        *,
        dense_retriever: Retriever,
        lexical_retriever: Retriever,
        config: HybridConfig | None = None,
    ) -> None:
        if not isinstance(dense_retriever, Retriever):
            raise TypeError("dense_retriever must implement Retriever")

        if not isinstance(lexical_retriever, Retriever):
            raise TypeError("lexical_retriever must implement Retriever")

        self._dense_retriever = dense_retriever
        self._lexical_retriever = lexical_retriever
        self._config = config or HybridConfig()

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Retrieve and fuse dense + lexical enterprise evidence."""

        if not isinstance(query, str) or not query.strip():
            raise HybridRetrievalError(
                "query must be a non-empty string"
            )

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise HybridRetrievalError(
                "top_k must be a positive integer"
            )

        if min_score is not None and (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not math.isfinite(float(min_score))
            or min_score < 0
        ):
            raise HybridRetrievalError(
                "min_score must be a finite non-negative number or None"
            )

        dense_results = self._dense_retriever.search(
            query,
            top_k=self._config.candidate_k,
            filters=filters,
        )
        lexical_results = self._lexical_retriever.search(
            query,
            top_k=self._config.candidate_k,
            filters=filters,
        )

        fused = reciprocal_rank_fusion(
            dense_results,
            lexical_results,
            rrf_k=self._config.rrf_k,
            top_k=top_k,
        )

        if min_score is None:
            return fused

        return [
            result
            for result in fused
            if result.score >= min_score
        ]
