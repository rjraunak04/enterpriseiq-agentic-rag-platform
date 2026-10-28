"""Pluggable reranking for EnterpriseIQ retrieval candidates."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from enterpriseiq.retrieval.models import SearchResult


class RerankingError(ValueError):
    """Raised when reranking input or configuration is invalid."""


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.,][0-9]+)*")


def tokenize_for_reranking(text: str) -> tuple[str, ...]:
    """Tokenize text deterministically for the fallback reranker."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return tuple(
        match.group(0).casefold()
        for match in _TOKEN_PATTERN.finditer(text)
    )


@runtime_checkable
class Reranker(Protocol):
    """Protocol implemented by candidate reranking backends."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        """Rerank retrieval candidates for a query."""
        ...


@dataclass(frozen=True, slots=True)
class RerankingConfig:
    """Configuration for deterministic candidate reranking."""

    candidate_weight: float = 0.35
    lexical_weight: float = 0.65

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_weight", self.candidate_weight),
            ("lexical_weight", self.lexical_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise RerankingError(
                    f"{name} must be a finite non-negative number"
                )

        if self.candidate_weight + self.lexical_weight <= 0:
            raise RerankingError(
                "at least one reranking weight must be positive"
            )


def _validate_candidates(
    candidates: Sequence[SearchResult],
) -> None:
    seen: set[str] = set()

    for candidate in candidates:
        if not isinstance(candidate, SearchResult):
            raise TypeError(
                "candidates must contain SearchResult instances"
            )

        if candidate.chunk_id in seen:
            raise RerankingError(
                f"duplicate candidate chunk_id: {candidate.chunk_id}"
            )

        seen.add(candidate.chunk_id)


def _normalized_candidate_score(
    candidates: Sequence[SearchResult],
) -> dict[str, float]:
    if not candidates:
        return {}

    scores = [float(candidate.score) for candidate in candidates]
    minimum = min(scores)
    maximum = max(scores)

    if math.isclose(minimum, maximum):
        return {
            candidate.chunk_id: 1.0
            for candidate in candidates
        }

    span = maximum - minimum
    return {
        candidate.chunk_id: (float(candidate.score) - minimum) / span
        for candidate in candidates
    }


def _lexical_overlap_score(
    query: str,
    candidate: SearchResult,
) -> float:
    query_tokens = set(tokenize_for_reranking(query))
    document_tokens = set(
        tokenize_for_reranking(
            " ".join(
                (
                    candidate.title,
                    candidate.text,
                    candidate.department,
                    candidate.document_id,
                )
            )
        )
    )

    if not query_tokens or not document_tokens:
        return 0.0

    return len(query_tokens & document_tokens) / len(query_tokens)


class DeterministicReranker:
    """Dependency-free reranker for deterministic local/test execution."""

    def __init__(
        self,
        config: RerankingConfig | None = None,
    ) -> None:
        self._config = config or RerankingConfig()

    @property
    def config(self) -> RerankingConfig:
        return self._config

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        """Rerank candidates using lexical relevance + prior rank signal."""

        if not isinstance(query, str) or not query.strip():
            raise RerankingError(
                "query must be a non-empty string"
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise RerankingError(
                "top_k must be a positive integer"
            )

        _validate_candidates(candidates)

        if not candidates:
            return []

        normalized_scores = _normalized_candidate_score(candidates)

        scored: list[tuple[float, SearchResult]] = []

        for candidate in candidates:
            lexical_score = _lexical_overlap_score(
                query,
                candidate,
            )
            prior_score = normalized_scores[candidate.chunk_id]

            score = (
                self._config.lexical_weight * lexical_score
                + self._config.candidate_weight * prior_score
            )

            scored.append((score, candidate))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].chunk_id,
            )
        )

        return [
            replace(
                candidate,
                rank=rank,
                score=score,
            )
            for rank, (score, candidate) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]


class RerankedRetriever:
    """Apply a reranker to candidates produced by another retriever."""

    def __init__(
        self,
        *,
        retriever: object,
        reranker: Reranker,
        candidate_k: int = 20,
    ) -> None:
        if not hasattr(retriever, "search"):
            raise TypeError(
                "retriever must provide a search method"
            )

        if not isinstance(reranker, Reranker):
            raise TypeError(
                "reranker must implement Reranker"
            )

        if (
            isinstance(candidate_k, bool)
            or not isinstance(candidate_k, int)
            or candidate_k <= 0
        ):
            raise RerankingError(
                "candidate_k must be a positive integer"
            )

        self._retriever = retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: object | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Retrieve a candidate pool and rerank it."""

        if not isinstance(query, str) or not query.strip():
            raise RerankingError(
                "query must be a non-empty string"
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise RerankingError(
                "top_k must be a positive integer"
            )

        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not math.isfinite(float(min_score))
            or min_score < 0
        ):
            raise RerankingError(
                "min_score must be a finite non-negative number"
            )

        candidates = self._retriever.search(
            query,
            top_k=self._candidate_k,
            filters=filters,
        )

        reranked = self._reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )

        return [
            result
            for result in reranked
            if result.score >= min_score
        ]
