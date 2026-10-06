"""Metadata-aware semantic retrieval for EnterpriseIQ."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterpriseiq.embeddings.contracts import EmbeddingProvider, Vector
from enterpriseiq.retrieval.models import SearchResult
from enterpriseiq.retrieval.vector_index import (
    PersistentVectorIndex,
    VectorIndexRecord,
)


class RetrievalError(ValueError):
    """Raised when a semantic retrieval request is invalid."""


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Optional enterprise metadata filters applied before ranking."""

    departments: tuple[str, ...] = ()
    classifications: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "departments",
            "classifications",
            "document_types",
            "document_ids",
        ):
            values = getattr(self, field_name)

            if not isinstance(values, tuple):
                raise TypeError(f"{field_name} must be a tuple")

            normalized: list[str] = []

            for value in values:
                if not isinstance(value, str):
                    raise TypeError(
                        f"{field_name} values must be strings"
                    )

                cleaned = value.strip()
                if not cleaned:
                    raise RetrievalError(
                        f"{field_name} values must not be empty"
                    )

                normalized.append(cleaned.casefold())

            object.__setattr__(
                self,
                field_name,
                tuple(normalized),
            )


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Compute cosine similarity between two finite vectors."""

    if len(left) != len(right):
        raise RetrievalError(
            "cosine similarity requires equal vector dimensions"
        )

    if not left:
        raise RetrievalError(
            "cosine similarity requires non-empty vectors"
        )

    try:
        left_values = tuple(float(value) for value in left)
        right_values = tuple(float(value) for value in right)
    except (TypeError, ValueError) as exc:
        raise RetrievalError(
            "cosine similarity vectors must be numeric"
        ) from exc

    if not all(
        math.isfinite(value)
        for value in (*left_values, *right_values)
    ):
        raise RetrievalError(
            "cosine similarity vectors must contain finite values"
        )

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(
            left_values,
            right_values,
            strict=True,
        )
    )

    left_norm = math.sqrt(
        sum(value * value for value in left_values)
    )
    right_norm = math.sqrt(
        sum(value * value for value in right_values)
    )

    if left_norm == 0.0 or right_norm == 0.0:
        raise RetrievalError(
            "cosine similarity is undefined for zero vectors"
        )

    score = dot_product / (left_norm * right_norm)

    # Protect ranking from tiny floating-point overshoots.
    return max(-1.0, min(1.0, score))


def _matches_filter(
    value: str,
    accepted: tuple[str, ...],
) -> bool:
    if not accepted:
        return True

    return value.casefold() in accepted


def _matches_filters(
    record: VectorIndexRecord,
    filters: SearchFilters,
) -> bool:
    return (
        _matches_filter(
            record.department,
            filters.departments,
        )
        and _matches_filter(
            record.classification,
            filters.classifications,
        )
        and _matches_filter(
            record.document_type,
            filters.document_types,
        )
        and _matches_filter(
            record.document_id,
            filters.document_ids,
        )
    )


def _result_metadata(
    record: VectorIndexRecord,
) -> Mapping[str, Any]:
    metadata = dict(record.metadata)

    metadata.update(
        {
            "document_type": record.document_type,
            "version": record.version,
            "effective_date": record.effective_date,
            "owner": record.owner,
            "classification": record.classification,
            "chunk_index": record.chunk_index,
        }
    )

    return metadata


class SemanticRetriever:
    """Rank EnterpriseIQ vector-index records for a natural-language query."""

    def __init__(
        self,
        *,
        index: PersistentVectorIndex,
        provider: EmbeddingProvider,
    ) -> None:
        if not isinstance(index, PersistentVectorIndex):
            raise TypeError(
                "index must be a PersistentVectorIndex"
            )

        if provider.dimension != index.dimension:
            raise RetrievalError(
                "embedding provider/index dimension mismatch: "
                f"provider={provider.dimension}, "
                f"index={index.dimension}"
            )

        if provider.model_name != index.model_name:
            raise RetrievalError(
                "embedding provider/index model mismatch: "
                f"provider={provider.model_name}, "
                f"index={index.model_name}"
            )

        self._index = index
        self._provider = provider

    @property
    def index(self) -> PersistentVectorIndex:
        return self._index

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Return ranked semantic matches with enterprise provenance."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()
        if not query:
            raise RetrievalError("query must not be empty")

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
        ):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise RetrievalError(
                "top_k must be greater than zero"
            )

        if filters is None:
            filters = SearchFilters()
        elif not isinstance(filters, SearchFilters):
            raise TypeError(
                "filters must be a SearchFilters instance"
            )

        if min_score is not None:
            if (
                isinstance(min_score, bool)
                or not isinstance(min_score, (int, float))
            ):
                raise TypeError(
                    "min_score must be numeric or None"
                )

            min_score = float(min_score)

            if not math.isfinite(min_score):
                raise RetrievalError(
                    "min_score must be finite"
                )

            if not -1.0 <= min_score <= 1.0:
                raise RetrievalError(
                    "min_score must be between -1 and 1"
                )

        query_vector: Vector = self._provider.embed_query(
            query
        )

        if len(query_vector) != self._index.dimension:
            raise RetrievalError(
                "query embedding dimension does not match index"
            )

        candidates: list[
            tuple[float, VectorIndexRecord]
        ] = []

        for record in self._index.records:
            if not _matches_filters(record, filters):
                continue

            score = cosine_similarity(
                query_vector,
                record.vector,
            )

            if (
                min_score is not None
                and score < min_score
            ):
                continue

            candidates.append((score, record))

        # Score descending, then stable deterministic chunk ID tie-break.
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1].chunk_id,
            )
        )

        results: list[SearchResult] = []

        for rank, (score, record) in enumerate(
            candidates[:top_k],
            start=1,
        ):
            results.append(
                SearchResult(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    text=record.text,
                    score=score,
                    rank=rank,
                    title=record.title,
                    department=record.department,
                    source_path=Path(record.source_path),
                    metadata=_result_metadata(record),
                )
            )

        return results
