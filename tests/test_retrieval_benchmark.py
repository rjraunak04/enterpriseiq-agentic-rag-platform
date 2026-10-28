"""Tests for the EnterpriseIQ gold retrieval benchmark."""

import json
from pathlib import Path

import pytest

from enterpriseiq.evaluation.benchmark import (
    BenchmarkError,
    BenchmarkQuery,
    RelevanceJudgment,
    RetrievalBenchmark,
    load_benchmark,
    validate_benchmark_documents,
)
from enterpriseiq.retrieval.vector_index import PersistentVectorIndex

GOLD_PATH = Path("data/evaluation/retrieval_gold_v1.json")


def test_gold_benchmark_loads() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    assert benchmark.name == "enterpriseiq-enterprise-retrieval-gold"
    assert benchmark.version == "1.0.0"
    assert benchmark.query_count == 15


def test_gold_query_ids_are_unique() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    query_ids = [query.query_id for query in benchmark.queries]

    assert len(query_ids) == len(set(query_ids))


def test_gold_covers_enterprise_departments() -> None:
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


def test_gold_contains_graded_relevance() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    relevance_values = {
        judgment.relevance
        for query in benchmark.queries
        for judgment in query.judgments
    }

    assert {1, 2} <= relevance_values


def test_gold_references_only_real_corpus_documents() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    index = PersistentVectorIndex.load(
        index_path=Path("data/processed/vector_index.jsonl"),
        manifest_path=Path(
            "data/processed/vector_index_manifest.json"
        ),
    )

    document_ids = [
        record.document_id
        for record in index.records
    ]

    validate_benchmark_documents(
        benchmark,
        document_ids,
    )


def test_relevant_document_ids_property() -> None:
    query = BenchmarkQuery(
        query_id="Q1",
        query="remote work",
        category="HR",
        judgments=(
            RelevanceJudgment("HR-003", 2),
            RelevanceJudgment("HR-001", 1),
        ),
        metadata={},
    )

    assert query.relevant_document_ids == frozenset(
        {"HR-003", "HR-001"}
    )


def test_benchmark_metadata_is_immutable() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    with pytest.raises(TypeError):
        benchmark.metadata["x"] = "y"  # type: ignore[index]


def test_query_metadata_is_immutable() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    with pytest.raises(TypeError):
        benchmark.queries[0].metadata["x"] = "y"  # type: ignore[index]


def test_duplicate_query_ids_are_rejected() -> None:
    query = BenchmarkQuery(
        query_id="Q1",
        query="remote work",
        category="HR",
        judgments=(RelevanceJudgment("HR-003", 2),),
        metadata={},
    )

    with pytest.raises(
        BenchmarkError,
        match="query IDs must be unique",
    ):
        RetrievalBenchmark(
            name="benchmark",
            version="1",
            queries=(query, query),
            metadata={},
        )


def test_duplicate_judgments_are_rejected() -> None:
    with pytest.raises(
        BenchmarkError,
        match="duplicate relevance judgment",
    ):
        BenchmarkQuery(
            query_id="Q1",
            query="remote",
            category="HR",
            judgments=(
                RelevanceJudgment("HR-003", 2),
                RelevanceJudgment("HR-003", 1),
            ),
            metadata={},
        )


@pytest.mark.parametrize("relevance", [0, -1, True])
def test_invalid_relevance_is_rejected(
    relevance: object,
) -> None:
    with pytest.raises(
        BenchmarkError,
        match="positive integer",
    ):
        RelevanceJudgment(
            "HR-003",
            relevance,  # type: ignore[arg-type]
        )


def test_unknown_document_reference_is_rejected() -> None:
    benchmark = load_benchmark(GOLD_PATH)

    with pytest.raises(
        BenchmarkError,
        match="unknown document IDs",
    ):
        validate_benchmark_documents(
            benchmark,
            ["NOT-A-REAL-DOCUMENT"],
        )


def test_missing_benchmark_file_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_benchmark(tmp_path / "missing.json")


def test_invalid_json_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(
        BenchmarkError,
        match="invalid benchmark JSON",
    ):
        load_benchmark(path)


def test_empty_queries_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps(
            {
                "name": "test",
                "version": "1",
                "queries": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        BenchmarkError,
        match="non-empty queries",
    ):
        load_benchmark(path)
