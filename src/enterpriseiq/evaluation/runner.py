"""Offline retrieval benchmark runner for EnterpriseIQ."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Protocol

from enterpriseiq.evaluation.benchmark import RetrievalBenchmark
from enterpriseiq.evaluation.metrics import QueryMetrics, evaluate_query
from enterpriseiq.retrieval.models import SearchResult
from enterpriseiq.retrieval.semantic import SearchFilters


class EvaluationError(ValueError):
    """Raised when retrieval evaluation cannot be completed."""


class EvaluationRetriever(Protocol):
    """Minimal retriever contract required by the benchmark runner."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        min_score: float | None = None,
    ) -> Sequence[SearchResult]:
        """Return ranked retrieval results."""


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Evaluation result for one benchmark query."""

    query_id: str
    query: str
    retrieved_document_ids: tuple[str, ...]
    metrics: QueryMetrics


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Mean benchmark metrics for one retrieval system."""

    query_count: int
    k: int
    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


@dataclass(frozen=True, slots=True)
class RetrieverEvaluation:
    """Complete benchmark result for one retriever."""

    retriever_name: str
    benchmark_name: str
    benchmark_version: str
    aggregate: AggregateMetrics
    queries: tuple[QueryEvaluation, ...]


def _relevance_map(query: object) -> dict[str, int]:
    judgments = getattr(query, "judgments", None)

    if judgments is None:
        raise EvaluationError("benchmark query has no judgments")

    return {
        judgment.document_id: judgment.relevance
        for judgment in judgments
    }


def _document_ranking(
    results: Sequence[SearchResult],
    *,
    k: int,
) -> tuple[str, ...]:
    """Convert chunk results into a unique document ranking."""

    document_ids: list[str] = []
    seen: set[str] = set()

    for result in results:
        document_id = result.document_id

        if document_id in seen:
            continue

        seen.add(document_id)
        document_ids.append(document_id)

        if len(document_ids) == k:
            break

    return tuple(document_ids)


def _aggregate(
    metrics: Sequence[QueryMetrics],
    *,
    k: int,
) -> AggregateMetrics:
    if not metrics:
        raise EvaluationError(
            "cannot aggregate an empty metric collection"
        )

    return AggregateMetrics(
        query_count=len(metrics),
        k=k,
        hit_at_k=fmean(item.hit for item in metrics),
        precision_at_k=fmean(
            item.precision for item in metrics
        ),
        recall_at_k=fmean(item.recall for item in metrics),
        mrr_at_k=fmean(
            item.reciprocal_rank for item in metrics
        ),
        ndcg_at_k=fmean(item.ndcg for item in metrics),
    )


def evaluate_retriever(
    *,
    retriever_name: str,
    retriever: EvaluationRetriever,
    benchmark: RetrievalBenchmark,
    k: int = 5,
    candidate_k: int | None = None,
) -> RetrieverEvaluation:
    """Evaluate one retriever against a versioned benchmark."""

    if not isinstance(retriever_name, str) or not retriever_name.strip():
        raise EvaluationError(
            "retriever_name must be a non-empty string"
        )

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise EvaluationError("k must be a positive integer")

    if candidate_k is None:
        candidate_k = k

    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or candidate_k < k
    ):
        raise EvaluationError(
            "candidate_k must be an integer greater than or equal to k"
        )

    query_evaluations: list[QueryEvaluation] = []

    for benchmark_query in benchmark.queries:
        results = retriever.search(
            benchmark_query.query,
            top_k=candidate_k,
        )

        document_ids = _document_ranking(
            results,
            k=k,
        )

        metrics = evaluate_query(
            document_ids,
            _relevance_map(benchmark_query),
            k=k,
        )

        query_evaluations.append(
            QueryEvaluation(
                query_id=benchmark_query.query_id,
                query=benchmark_query.query,
                retrieved_document_ids=document_ids,
                metrics=metrics,
            )
        )

    aggregate = _aggregate(
        [item.metrics for item in query_evaluations],
        k=k,
    )

    return RetrieverEvaluation(
        retriever_name=retriever_name,
        benchmark_name=benchmark.name,
        benchmark_version=benchmark.version,
        aggregate=aggregate,
        queries=tuple(query_evaluations),
    )


def compare_retrievers(
    *,
    retrievers: Mapping[str, EvaluationRetriever],
    benchmark: RetrievalBenchmark,
    k: int = 5,
    candidate_k: int | None = None,
) -> tuple[RetrieverEvaluation, ...]:
    """Evaluate multiple retrieval systems on the same benchmark."""

    if not retrievers:
        raise EvaluationError(
            "at least one retriever is required"
        )

    evaluations = [
        evaluate_retriever(
            retriever_name=name,
            retriever=retriever,
            benchmark=benchmark,
            k=k,
            candidate_k=candidate_k,
        )
        for name, retriever in retrievers.items()
    ]

    return tuple(
        sorted(
            evaluations,
            key=lambda item: (
                -item.aggregate.ndcg_at_k,
                -item.aggregate.mrr_at_k,
                -item.aggregate.recall_at_k,
                item.retriever_name,
            ),
        )
    )


def write_evaluation_report(
    evaluations: Sequence[RetrieverEvaluation],
    path: Path,
) -> None:
    """Write deterministic JSON evaluation results."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")

    if not evaluations:
        raise EvaluationError(
            "cannot write an empty evaluation report"
        )

    payload = {
        "schema_version": "1.0",
        "retrievers": [
            asdict(evaluation)
            for evaluation in evaluations
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
