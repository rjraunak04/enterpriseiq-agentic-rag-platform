"""Regression and integrity gates for advanced retrieval evaluation."""

import json
from pathlib import Path

import pytest

from enterpriseiq.evaluation import (
    compare_retrievers,
    evaluate_query,
    evaluate_retriever,
    load_benchmark,
    validate_benchmark_documents,
    write_evaluation_report,
)
from enterpriseiq.evaluation.runner import EvaluationError
from enterpriseiq.retrieval.models import SearchResult

GOLD_PATH = Path("data/evaluation/retrieval_gold_v1.json")


def _result(
    document_id: str,
    *,
    rank: int,
    score: float,
) -> SearchResult:
    return SearchResult(
        chunk_id=f"{document_id}-chunk-{rank}",
        document_id=document_id,
        text=f"Evidence from {document_id}",
        score=score,
        rank=rank,
        title=document_id,
        department="Test",
        source_path=Path(f"{document_id}.md"),
        metadata={
            "document_type": "policy",
            "classification": "internal",
        },
    )


class MappingRetriever:
    """Deterministic retriever used for regression gates."""

    def __init__(
        self,
        rankings: dict[str, list[str]],
    ) -> None:
        self.rankings = rankings

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters=None,
        min_score=None,
    ):
        document_ids = self.rankings.get(query, [])

        return [
            _result(
                document_id,
                rank=rank,
                score=1.0 / rank,
            )
            for rank, document_id in enumerate(
                document_ids[:top_k],
                start=1,
            )
        ]


def _perfect_rankings(benchmark) -> dict[str, list[str]]:
    return {
        query.query: [
            judgment.document_id
            for judgment in sorted(
                query.judgments,
                key=lambda item: (
                    -item.relevance,
                    item.document_id,
                ),
            )
        ]
        for query in benchmark.queries
    }


def test_gold_benchmark_is_versioned_and_nonempty() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    assert benchmark.name
    assert benchmark.version == "1.0.0"
    assert len(benchmark.queries) == 15


def test_gold_query_ids_are_unique() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    query_ids = [
        query.query_id
        for query in benchmark.queries
    ]

    assert len(query_ids) == len(set(query_ids))


def test_gold_query_text_is_unique() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    queries = [
        query.query
        for query in benchmark.queries
    ]

    assert len(queries) == len(set(queries))


def test_every_gold_query_has_relevance_judgments() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    assert all(
        query.judgments
        for query in benchmark.queries
    )


def test_gold_relevance_grades_are_supported() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    grades = {
        judgment.relevance
        for query in benchmark.queries
        for judgment in query.judgments
    }

    assert grades <= {1, 2}
    assert grades


def test_gold_covers_expected_departments() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    categories = {
        query.category
        for query in benchmark.queries
    }

    assert {
        "HR",
        "Finance",
        "IT",
        "Security",
        "Operations",
        "Product",
    } <= categories


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_metric_bundle_stays_within_valid_bounds(
    k: int,
) -> None:
    metrics = evaluate_query(
        ["DOC-A", "DOC-X", "DOC-B"],
        {
            "DOC-A": 2,
            "DOC-B": 1,
        },
        k=k,
    )

    assert 0.0 <= metrics.hit <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.reciprocal_rank <= 1.0
    assert 0.0 <= metrics.ndcg <= 1.0


def test_perfect_retriever_hits_quality_ceiling() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    evaluation = evaluate_retriever(
        retriever_name="perfect",
        retriever=MappingRetriever(
            _perfect_rankings(benchmark)
        ),
        benchmark=benchmark,
        k=5,
    )

    assert evaluation.aggregate.hit_at_k == 1.0
    assert evaluation.aggregate.recall_at_k == 1.0
    assert evaluation.aggregate.mrr_at_k == 1.0
    assert evaluation.aggregate.ndcg_at_k == 1.0


def test_empty_retriever_hits_quality_floor() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    evaluation = evaluate_retriever(
        retriever_name="empty",
        retriever=MappingRetriever({}),
        benchmark=benchmark,
        k=5,
    )

    assert evaluation.aggregate.hit_at_k == 0.0
    assert evaluation.aggregate.recall_at_k == 0.0
    assert evaluation.aggregate.mrr_at_k == 0.0
    assert evaluation.aggregate.ndcg_at_k == 0.0


def test_evaluation_is_deterministic() -> None:
    benchmark = load_benchmark(GOLD_PATH)
    retriever = MappingRetriever(
        _perfect_rankings(benchmark)
    )

    first = evaluate_retriever(
        retriever_name="deterministic",
        retriever=retriever,
        benchmark=benchmark,
        k=5,
    )
    second = evaluate_retriever(
        retriever_name="deterministic",
        retriever=retriever,
        benchmark=benchmark,
        k=5,
    )

    assert first == second


def test_comparison_order_is_reproducible() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    rankings = _perfect_rankings(benchmark)

    first = compare_retrievers(
        retrievers={
            "zeta": MappingRetriever(rankings),
            "alpha": MappingRetriever(rankings),
        },
        benchmark=benchmark,
        k=5,
    )

    second = compare_retrievers(
        retrievers={
            "alpha": MappingRetriever(rankings),
            "zeta": MappingRetriever(rankings),
        },
        benchmark=benchmark,
        k=5,
    )

    assert [
        item.retriever_name
        for item in first
    ] == ["alpha", "zeta"]

    assert first == second


def test_json_report_is_byte_reproducible(
    tmp_path: Path,
) -> None:
    benchmark = load_benchmark(GOLD_PATH)

    evaluation = evaluate_retriever(
        retriever_name="perfect",
        retriever=MappingRetriever(
            _perfect_rankings(benchmark)
        ),
        benchmark=benchmark,
        k=5,
    )

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_evaluation_report((evaluation,), first)
    write_evaluation_report((evaluation,), second)

    assert first.read_bytes() == second.read_bytes()


def test_json_report_schema_is_stable(
    tmp_path: Path,
) -> None:
    benchmark = load_benchmark(GOLD_PATH)

    evaluation = evaluate_retriever(
        retriever_name="perfect",
        retriever=MappingRetriever(
            _perfect_rankings(benchmark)
        ),
        benchmark=benchmark,
        k=5,
    )

    path = tmp_path / "report.json"
    write_evaluation_report((evaluation,), path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "1.0"
    assert set(payload) == {
        "schema_version",
        "retrievers",
    }

    aggregate = payload["retrievers"][0]["aggregate"]

    assert set(aggregate) == {
        "query_count",
        "k",
        "hit_at_k",
        "precision_at_k",
        "recall_at_k",
        "mrr_at_k",
        "ndcg_at_k",
    }


def test_candidate_depth_cannot_be_below_metric_cutoff() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    with pytest.raises(
        EvaluationError,
        match="greater than or equal to k",
    ):
        evaluate_retriever(
            retriever_name="invalid",
            retriever=MappingRetriever({}),
            benchmark=benchmark,
            k=5,
            candidate_k=3,
        )


def test_gold_documents_match_persistent_index() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    index_path = Path(
        "data/processed/vector_index.json"
    )
    manifest_path = Path(
        "data/processed/vector_index_manifest.json"
    )

    if not index_path.exists() or not manifest_path.exists():
        pytest.skip(
            "persistent vector index artifacts are not available"
        )

    from enterpriseiq.retrieval.vector_index import (
        PersistentVectorIndex,
    )

    index = PersistentVectorIndex.load(
        index_path=index_path,
        manifest_path=manifest_path,
    )

    validate_benchmark_documents(
        benchmark,
        index.records,
    )
