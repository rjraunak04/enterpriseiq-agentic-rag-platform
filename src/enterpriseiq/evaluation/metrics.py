"""Retrieval evaluation metrics for EnterpriseIQ."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class MetricError(ValueError):
    """Raised when retrieval metric inputs are invalid."""


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise MetricError("k must be a positive integer")


def _validate_retrieved(
    retrieved_document_ids: Sequence[str],
) -> None:
    seen: set[str] = set()

    for document_id in retrieved_document_ids:
        if not isinstance(document_id, str) or not document_id.strip():
            raise MetricError(
                "retrieved document IDs must be non-empty strings"
            )

        if document_id in seen:
            raise MetricError(
                f"duplicate retrieved document ID: {document_id}"
            )

        seen.add(document_id)


def _validate_relevance(
    relevance: Mapping[str, int],
) -> None:
    for document_id, grade in relevance.items():
        if not isinstance(document_id, str) or not document_id.strip():
            raise MetricError(
                "relevance document IDs must be non-empty strings"
            )

        if (
            isinstance(grade, bool)
            or not isinstance(grade, int)
            or grade <= 0
        ):
            raise MetricError(
                "relevance grades must be positive integers"
            )


def _validate_inputs(
    retrieved_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    k: int,
) -> None:
    _validate_k(k)
    _validate_retrieved(retrieved_document_ids)
    _validate_relevance(relevance)


def hit_at_k(
    retrieved_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return 1 when any relevant document occurs in the first k results."""

    _validate_inputs(retrieved_document_ids, relevance, k)

    if not relevance:
        return 0.0

    return float(
        any(
            document_id in relevance
            for document_id in retrieved_document_ids[:k]
        )
    )


def precision_at_k(
    retrieved_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return binary relevance precision over the requested cutoff."""

    _validate_inputs(retrieved_document_ids, relevance, k)

    relevant_count = sum(
        document_id in relevance
        for document_id in retrieved_document_ids[:k]
    )

    return relevant_count / k


def recall_at_k(
    retrieved_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return fraction of relevant documents recovered in the first k."""

    _validate_inputs(retrieved_document_ids, relevance, k)

    if not relevance:
        return 0.0

    relevant_count = sum(
        document_id in relevance
        for document_id in retrieved_document_ids[:k]
    )

    return relevant_count / len(relevance)


def reciprocal_rank(
    retrieved_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int | None = None,
) -> float:
    """Return reciprocal rank of the first relevant retrieved document."""

    _validate_retrieved(retrieved_document_ids)
    _validate_relevance(relevance)

    if k is not None:
        _validate_k(k)
        candidates = retrieved_document_ids[:k]
    else:
        candidates = retrieved_document_ids

    if not relevance:
        return 0.0

    for rank, document_id in enumerate(candidates, start=1):
        if document_id in relevance:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return graded discounted cumulative gain at k."""

    _validate_inputs(retrieved_document_ids, relevance, k)

    score = 0.0

    for rank, document_id in enumerate(
        retrieved_document_ids[:k],
        start=1,
    ):
        grade = relevance.get(document_id, 0)

        if grade:
            gain = (2**grade) - 1
            score += gain / math.log2(rank + 1)

    return score


def ndcg_at_k(
    retrieved_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return normalized graded discounted cumulative gain at k."""

    _validate_inputs(retrieved_document_ids, relevance, k)

    if not relevance:
        return 0.0

    actual = dcg_at_k(
        retrieved_document_ids,
        relevance,
        k=k,
    )

    ideal_grades = sorted(
        relevance.values(),
        reverse=True,
    )[:k]

    ideal = sum(
        ((2**grade) - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )

    if math.isclose(ideal, 0.0):
        return 0.0

    return actual / ideal


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Metric values for one benchmark query at a cutoff."""

    k: int
    hit: float
    precision: float
    recall: float
    reciprocal_rank: float
    ndcg: float


def evaluate_query(
    retrieved_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int,
) -> QueryMetrics:
    """Evaluate one ranked document list against relevance judgments."""

    _validate_inputs(retrieved_document_ids, relevance, k)

    return QueryMetrics(
        k=k,
        hit=hit_at_k(
            retrieved_document_ids,
            relevance,
            k=k,
        ),
        precision=precision_at_k(
            retrieved_document_ids,
            relevance,
            k=k,
        ),
        recall=recall_at_k(
            retrieved_document_ids,
            relevance,
            k=k,
        ),
        reciprocal_rank=reciprocal_rank(
            retrieved_document_ids,
            relevance,
            k=k,
        ),
        ndcg=ndcg_at_k(
            retrieved_document_ids,
            relevance,
            k=k,
        ),
    )
