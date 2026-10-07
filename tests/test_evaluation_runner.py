"""Tests for the EnterpriseIQ retrieval evaluation runner."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from enterpriseiq.evaluation.benchmark import (
    BenchmarkQuery,
    RelevanceJudgment,
    RetrievalBenchmark,
)
from enterpriseiq.evaluation.runner import (
    EvaluationError,
    compare_retrievers,
    evaluate_retriever,
    write_evaluation_report,
)
from enterpriseiq.retrieval.models import SearchResult


def _result(
    document_id: str,
    *,
    chunk_id: str | None = None,
    score: float = 1.0,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id or f"{document_id}-chunk",
        document_id=document_id,
        text=f"content for {document_id}",
        score=score,
        rank=1,
        title=document_id,
        department="Test",
        source_path=Path(f"{document_id}.md"),
        metadata={
            "document_type": "policy",
            "classification": "internal",
        },
    )

class FakeRetriever:
    def __init__(
        self,
        results_by_query: dict[str, list[SearchResult]],
    ) -> None:
        self.results_by_query = results_by_query
        self.requested_top_k: list[int] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters=None,
        min_score=None,
    ):
        self.requested_top_k.append(top_k)
        return self.results_by_query.get(query, [])[:top_k]


def _benchmark() -> RetrievalBenchmark:
    return RetrievalBenchmark(
        name="test-benchmark",
        version="1.0",
        queries=(
            BenchmarkQuery(
                query_id="Q1",
                query="remote work",
                category="HR",
                judgments=(
                    RelevanceJudgment("DOC-A", 2),
                ),
                metadata={},
            ),
            BenchmarkQuery(
                query_id="Q2",
                query="expense approval",
                category="Finance",
                judgments=(
                    RelevanceJudgment("DOC-B", 2),
                    RelevanceJudgment("DOC-C", 1),
                ),
                metadata={},
            ),
        ),
        metadata={},
    )


def test_evaluate_retriever() -> None:
    retriever = FakeRetriever(
        {
            "remote work": [_result("DOC-A")],
            "expense approval": [
                _result("DOC-X"),
                _result("DOC-B"),
                _result("DOC-C"),
            ],
        }
    )

    evaluation = evaluate_retriever(
        retriever_name="test",
        retriever=retriever,
        benchmark=_benchmark(),
        k=3,
    )

    assert evaluation.retriever_name == "test"
    assert evaluation.aggregate.query_count == 2
    assert evaluation.aggregate.k == 3
    assert len(evaluation.queries) == 2
    assert 0.0 <= evaluation.aggregate.ndcg_at_k <= 1.0


def test_chunk_results_are_deduplicated_to_documents() -> None:
    retriever = FakeRetriever(
        {
            "remote work": [
                _result("DOC-A", chunk_id="A-1"),
                _result("DOC-A", chunk_id="A-2"),
                _result("DOC-X"),
            ],
            "expense approval": [_result("DOC-B")],
        }
    )

    evaluation = evaluate_retriever(
        retriever_name="dedupe",
        retriever=retriever,
        benchmark=_benchmark(),
        k=2,
        candidate_k=3,
    )

    assert evaluation.queries[0].retrieved_document_ids == (
        "DOC-A",
        "DOC-X",
    )


def test_candidate_k_is_forwarded_to_retriever() -> None:
    retriever = FakeRetriever({})

    evaluate_retriever(
        retriever_name="test",
        retriever=retriever,
        benchmark=_benchmark(),
        k=2,
        candidate_k=7,
    )

    assert retriever.requested_top_k == [7, 7]


def test_default_candidate_k_equals_k() -> None:
    retriever = FakeRetriever({})

    evaluate_retriever(
        retriever_name="test",
        retriever=retriever,
        benchmark=_benchmark(),
        k=4,
    )

    assert retriever.requested_top_k == [4, 4]


def test_comparison_prefers_better_ndcg() -> None:
    good = FakeRetriever(
        {
            "remote work": [_result("DOC-A")],
            "expense approval": [
                _result("DOC-B"),
                _result("DOC-C"),
            ],
        }
    )
    poor = FakeRetriever(
        {
            "remote work": [_result("DOC-X")],
            "expense approval": [_result("DOC-X")],
        }
    )

    evaluations = compare_retrievers(
        retrievers={
            "poor": poor,
            "good": good,
        },
        benchmark=_benchmark(),
        k=2,
    )

    assert evaluations[0].retriever_name == "good"
    assert (
        evaluations[0].aggregate.ndcg_at_k
        > evaluations[1].aggregate.ndcg_at_k
    )


def test_comparison_is_deterministic_on_tie() -> None:
    first = FakeRetriever({})
    second = FakeRetriever({})

    evaluations = compare_retrievers(
        retrievers={
            "zeta": first,
            "alpha": second,
        },
        benchmark=_benchmark(),
        k=2,
    )

    assert [
        item.retriever_name
        for item in evaluations
    ] == ["alpha", "zeta"]


def test_report_is_written_as_json(tmp_path: Path) -> None:
    retriever = FakeRetriever(
        {
            "remote work": [_result("DOC-A")],
            "expense approval": [_result("DOC-B")],
        }
    )

    evaluation = evaluate_retriever(
        retriever_name="test",
        retriever=retriever,
        benchmark=_benchmark(),
        k=2,
    )

    path = tmp_path / "report.json"

    write_evaluation_report((evaluation,), path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "1.0"
    assert payload["retrievers"][0]["retriever_name"] == "test"
    assert (
        payload["retrievers"][0]["aggregate"]["query_count"]
        == 2
    )


@pytest.mark.parametrize("k", [0, -1, True])
def test_invalid_k_rejected(k: object) -> None:
    with pytest.raises(
        EvaluationError,
        match="positive integer",
    ):
        evaluate_retriever(
            retriever_name="test",
            retriever=FakeRetriever({}),
            benchmark=_benchmark(),
            k=k,  # type: ignore[arg-type]
        )


def test_candidate_k_cannot_be_smaller_than_k() -> None:
    with pytest.raises(
        EvaluationError,
        match="greater than or equal to k",
    ):
        evaluate_retriever(
            retriever_name="test",
            retriever=FakeRetriever({}),
            benchmark=_benchmark(),
            k=5,
            candidate_k=2,
        )


def test_empty_retriever_name_rejected() -> None:
    with pytest.raises(
        EvaluationError,
        match="non-empty string",
    ):
        evaluate_retriever(
            retriever_name=" ",
            retriever=FakeRetriever({}),
            benchmark=_benchmark(),
        )


def test_empty_comparison_rejected() -> None:
    with pytest.raises(
        EvaluationError,
        match="at least one retriever",
    ):
        compare_retrievers(
            retrievers={},
            benchmark=_benchmark(),
        )


def test_empty_report_rejected(tmp_path: Path) -> None:
    with pytest.raises(
        EvaluationError,
        match="empty evaluation report",
    ):
        write_evaluation_report(
            (),
            tmp_path / "report.json",
        )


def test_report_requires_path() -> None:
    retriever = FakeRetriever({})
    evaluation = evaluate_retriever(
        retriever_name="test",
        retriever=retriever,
        benchmark=_benchmark(),
    )

    with pytest.raises(
        TypeError,
        match="path must be a pathlib.Path",
    ):
        write_evaluation_report(
            (evaluation,),
            "report.json",  # type: ignore[arg-type]
        )


def test_query_results_are_immutable() -> None:
    retriever = FakeRetriever({})
    evaluation = evaluate_retriever(
        retriever_name="test",
        retriever=retriever,
        benchmark=_benchmark(),
    )

    with pytest.raises(AttributeError):
        evaluation.queries[0].query_id = "changed"  # type: ignore[misc]


def test_comparison_uses_secondary_metric_for_tie() -> None:
    retriever = FakeRetriever({})
    evaluation = evaluate_retriever(
        retriever_name="a",
        retriever=retriever,
        benchmark=_benchmark(),
    )

    improved = replace(
        evaluation,
        retriever_name="b",
        aggregate=replace(
            evaluation.aggregate,
            mrr_at_k=0.5,
        ),
    )

    base = replace(
        evaluation,
        retriever_name="a",
    )

    assert improved.aggregate.mrr_at_k > base.aggregate.mrr_at_k
