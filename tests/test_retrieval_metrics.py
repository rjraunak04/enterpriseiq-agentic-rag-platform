"""Tests for EnterpriseIQ retrieval evaluation metrics."""

import math

import pytest

from enterpriseiq.evaluation.metrics import (
    MetricError,
    QueryMetrics,
    dcg_at_k,
    evaluate_query,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RELEVANCE = {"DOC-A": 2, "DOC-B": 1}


def test_hit_at_k() -> None:
    assert hit_at_k(["X", "DOC-A"], RELEVANCE, k=2) == 1.0
    assert hit_at_k(["X", "Y"], RELEVANCE, k=2) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k(
        ["DOC-A", "X", "DOC-B"], RELEVANCE, k=3
    ) == pytest.approx(2 / 3)


def test_precision_uses_k_denominator() -> None:
    assert precision_at_k(
        ["DOC-A"], RELEVANCE, k=5
    ) == pytest.approx(0.2)


def test_recall_at_k() -> None:
    assert recall_at_k(
        ["X", "DOC-A"], RELEVANCE, k=2
    ) == pytest.approx(0.5)


def test_recall_all_relevant() -> None:
    assert recall_at_k(
        ["DOC-A", "X", "DOC-B"], RELEVANCE, k=3
    ) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(
        ["X", "Y", "DOC-B"], RELEVANCE
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_respects_k() -> None:
    assert reciprocal_rank(
        ["X", "DOC-A"], RELEVANCE, k=1
    ) == 0.0


def test_dcg_at_k() -> None:
    expected = 3.0 + 1.0 / math.log2(3)
    assert dcg_at_k(
        ["DOC-A", "DOC-B"], RELEVANCE, k=2
    ) == pytest.approx(expected)


def test_ndcg_ideal_ranking() -> None:
    assert ndcg_at_k(
        ["DOC-A", "DOC-B"], RELEVANCE, k=2
    ) == pytest.approx(1.0)


def test_ndcg_penalizes_wrong_order() -> None:
    score = ndcg_at_k(
        ["DOC-B", "DOC-A"], RELEVANCE, k=2
    )
    assert 0.0 < score < 1.0


def test_empty_relevance() -> None:
    assert hit_at_k(["X"], {}, k=1) == 0.0
    assert recall_at_k(["X"], {}, k=1) == 0.0
    assert reciprocal_rank(["X"], {}) == 0.0
    assert ndcg_at_k(["X"], {}, k=1) == 0.0


def test_evaluate_query_bundle() -> None:
    result = evaluate_query(
        ["X", "DOC-A", "DOC-B"],
        RELEVANCE,
        k=3,
    )
    assert isinstance(result, QueryMetrics)
    assert result.hit == 1.0
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == 1.0
    assert result.reciprocal_rank == pytest.approx(0.5)
    assert 0.0 < result.ndcg <= 1.0


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(k: object) -> None:
    with pytest.raises(MetricError, match="positive integer"):
        hit_at_k(
            ["DOC-A"],
            RELEVANCE,
            k=k,  # type: ignore[arg-type]
        )


def test_duplicate_retrieved_ids_rejected() -> None:
    with pytest.raises(
        MetricError,
        match="duplicate retrieved document ID",
    ):
        hit_at_k(
            ["DOC-A", "DOC-A"],
            RELEVANCE,
            k=2,
        )


@pytest.mark.parametrize("grade", [0, -1, True, 1.5])
def test_invalid_relevance_grade(grade: object) -> None:
    with pytest.raises(
        MetricError,
        match="positive integers",
    ):
        hit_at_k(
            ["DOC-A"],
            {"DOC-A": grade},  # type: ignore[dict-item]
            k=1,
        )


@pytest.mark.parametrize("document_id", ["", "   "])
def test_invalid_retrieved_id(document_id: str) -> None:
    with pytest.raises(
        MetricError,
        match="non-empty strings",
    ):
        hit_at_k(
            [document_id],
            RELEVANCE,
            k=1,
        )


def test_query_metrics_immutable() -> None:
    result = evaluate_query(
        ["DOC-A"],
        RELEVANCE,
        k=1,
    )
    with pytest.raises(AttributeError):
        result.hit = 0.0  # type: ignore[misc]
