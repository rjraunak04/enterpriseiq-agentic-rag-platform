"""Dependency-light BM25 lexical retrieval for EnterpriseIQ."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from enterpriseiq.retrieval.models import SearchResult
from enterpriseiq.retrieval.semantic import SearchFilters
from enterpriseiq.retrieval.vector_index import VectorIndexRecord

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.,][0-9]+)*")


class LexicalRetrievalError(ValueError):
    """Raised when lexical retrieval input or configuration is invalid."""


@dataclass(frozen=True, slots=True)
class BM25Config:
    """Configuration for BM25 lexical ranking."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if isinstance(self.k1, bool) or not isinstance(self.k1, (int, float)):
            raise LexicalRetrievalError("k1 must be numeric")
        if isinstance(self.b, bool) or not isinstance(self.b, (int, float)):
            raise LexicalRetrievalError("b must be numeric")
        if not math.isfinite(float(self.k1)) or self.k1 <= 0:
            raise LexicalRetrievalError("k1 must be finite and greater than zero")
        if not math.isfinite(float(self.b)) or not 0 <= self.b <= 1:
            raise LexicalRetrievalError("b must be finite and between 0 and 1")


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize enterprise text deterministically for lexical retrieval."""

    if not isinstance(text, str):
        raise LexicalRetrievalError("text must be a string")

    return tuple(
        match.group(0).casefold()
        for match in _TOKEN_PATTERN.finditer(text)
    )


def _matches_filter(
    value: str,
    allowed: frozenset[str] | None,
) -> bool:
    if not allowed:
        return True
    return value.casefold() in {str(item).casefold() for item in allowed}


def _matches_filters(
    record: VectorIndexRecord,
    filters: SearchFilters | None,
) -> bool:
    if filters is None:
        return True

    return (
        _matches_filter(record.department, filters.departments)
        and _matches_filter(record.classification, filters.classifications)
        and _matches_filter(record.document_type, filters.document_types)
        and _matches_filter(record.document_id, filters.document_ids)
    )


class BM25Retriever:
    """Rank enterprise chunks using BM25 lexical relevance."""

    def __init__(
        self,
        records: Sequence[VectorIndexRecord] | Iterable[VectorIndexRecord],
        config: BM25Config | None = None,
    ) -> None:
        self._records = tuple(records)
        self._config = config or BM25Config()

        chunk_ids = [record.chunk_id for record in self._records]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise LexicalRetrievalError("duplicate chunk_id in lexical corpus")

        self._document_tokens = tuple(
            tokenize(record.text)
            for record in self._records
        )
        self._term_frequencies = tuple(
            Counter(tokens)
            for tokens in self._document_tokens
        )
        self._document_lengths = tuple(
            len(tokens)
            for tokens in self._document_tokens
        )

        if self._records:
            self._average_document_length = (
                sum(self._document_lengths) / len(self._document_lengths)
            )
        else:
            self._average_document_length = 0.0

        document_frequency: Counter[str] = Counter()
        for tokens in self._document_tokens:
            document_frequency.update(set(tokens))

        self._document_frequency = document_frequency

    @property
    def record_count(self) -> int:
        return len(self._records)

    def _idf(self, term: str) -> float:
        n = len(self._records)
        df = self._document_frequency.get(term, 0)

        if n == 0 or df == 0:
            return 0.0

        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _score(
        self,
        query_terms: tuple[str, ...],
        record_index: int,
    ) -> float:
        if not query_terms:
            return 0.0

        frequencies = self._term_frequencies[record_index]
        document_length = self._document_lengths[record_index]

        if document_length == 0 or self._average_document_length == 0:
            return 0.0

        k1 = float(self._config.k1)
        b = float(self._config.b)
        score = 0.0

        for term in set(query_terms):
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue

            denominator = frequency + k1 * (
                1.0
                - b
                + b * document_length / self._average_document_length
            )

            score += self._idf(term) * (
                frequency * (k1 + 1.0) / denominator
            )

        return score

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Return deterministic BM25-ranked enterprise evidence."""

        if not isinstance(query, str) or not query.strip():
            raise LexicalRetrievalError("query must be a non-empty string")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise LexicalRetrievalError("top_k must be a positive integer")

        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, (int, float))
            or not math.isfinite(float(min_score))
            or min_score < 0
        ):
            raise LexicalRetrievalError(
                "min_score must be a finite non-negative number"
            )

        query_terms = tokenize(query)
        scored: list[tuple[float, VectorIndexRecord]] = []

        for index, record in enumerate(self._records):
            if not _matches_filters(record, filters):
                continue

            score = self._score(query_terms, index)

            if score <= 0 or score < min_score:
                continue

            scored.append((score, record))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))

        results: list[SearchResult] = []

        for rank, (score, record) in enumerate(
            scored[:top_k],
            start=1,
        ):
            results.append(
                SearchResult(
                    rank=rank,
                    score=score,
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    title=record.title,
                    department=record.department,
                    text=record.text,
                    source_path=Path(record.source_path),
                    metadata=record.metadata,
                )
            )

        return results




