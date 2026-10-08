"""End-to-end tests for the grounded RAG pipeline."""

from pathlib import Path

import pytest

from enterpriseiq.rag import (
    DeterministicAnswerGenerator,
    GeneratedAnswer,
    GroundedRAGPipeline,
    GroundedResponse,
    RAGPipelineConfig,
    RAGPipelineError,
    RAGQuery,
    RAGRetriever,
)
from enterpriseiq.retrieval.models import SearchResult
from enterpriseiq.retrieval.semantic import SearchFilters


def _result(
    chunk_id: str = "chunk-1",
    *,
    rank: int = 1,
    text: str = (
        "Employees may work remotely up to three days per week."
    ),
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="HR-003",
        text=text,
        score=0.95,
        rank=rank,
        title="Remote Work Policy",
        department="HR",
        source_path=Path("hr/remote_work_policy.md"),
        metadata={"classification": "internal"},
    )


class StubRetriever:
    def __init__(
        self,
        results: list[SearchResult],
    ) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "min_score": min_score,
            }
        )
        return self.results[:top_k]


class UncitedGenerator:
    def generate(self, prompt) -> GeneratedAnswer:
        return GeneratedAnswer(
            text="This answer contains no citation."
        )


def test_stub_satisfies_retriever_protocol() -> None:
    assert isinstance(StubRetriever([]), RAGRetriever)


def test_pipeline_returns_grounded_response() -> None:
    pipeline = GroundedRAGPipeline(
        retriever=StubRetriever([_result()]),
        generator=DeterministicAnswerGenerator(),
    )

    response = pipeline.answer(
        "How many remote days are allowed?"
    )

    assert isinstance(response, GroundedResponse)
    assert response.grounded is True
    assert response.abstained is False


def test_pipeline_returns_citation_backed_answer() -> None:
    pipeline = GroundedRAGPipeline(
        retriever=StubRetriever([_result()]),
        generator=DeterministicAnswerGenerator(),
    )

    response = pipeline.answer("Remote work?")

    assert "[S1]" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].document_id == "HR-003"


def test_pipeline_preserves_exact_provenance() -> None:
    pipeline = GroundedRAGPipeline(
        retriever=StubRetriever([_result()]),
        generator=DeterministicAnswerGenerator(),
    )

    response = pipeline.answer("Remote work?")

    citation = response.citations[0]

    assert citation.chunk_id == "chunk-1"
    assert citation.title == "Remote Work Policy"
    assert citation.source_path == Path(
        "hr/remote_work_policy.md"
    )


def test_no_results_produces_safe_abstention() -> None:
    pipeline = GroundedRAGPipeline(
        retriever=StubRetriever([]),
        generator=DeterministicAnswerGenerator(),
    )

    response = pipeline.answer("Unknown policy?")

    assert response.abstained is True
    assert response.grounded is False
    assert response.citations == ()
    assert response.evidence_count == 0


def test_pipeline_accepts_rag_query() -> None:
    pipeline = GroundedRAGPipeline(
        retriever=StubRetriever([_result()]),
        generator=DeterministicAnswerGenerator(),
    )

    query = RAGQuery(
        text="Remote work?",
        query_id="query-001",
    )

    response = pipeline.answer(query)

    assert response.query == query
    assert response.query.query_id == "query-001"


def test_pipeline_normalizes_string_query() -> None:
    pipeline = GroundedRAGPipeline(
        retriever=StubRetriever([_result()]),
        generator=DeterministicAnswerGenerator(),
    )

    response = pipeline.answer("  Remote work?  ")

    assert response.query.text == "Remote work?"


def test_pipeline_passes_top_k_to_retriever() -> None:
    retriever = StubRetriever([_result()])

    pipeline = GroundedRAGPipeline(
        retriever=retriever,
        generator=DeterministicAnswerGenerator(),
        config=RAGPipelineConfig(top_k=7),
    )

    pipeline.answer("Remote work?")

    assert retriever.calls[0]["top_k"] == 7


def test_pipeline_passes_min_score_to_retriever() -> None:
    retriever = StubRetriever([_result()])

    pipeline = GroundedRAGPipeline(
        retriever=retriever,
        generator=DeterministicAnswerGenerator(),
        config=RAGPipelineConfig(
            top_k=5,
            min_score=0.4,
        ),
    )

    pipeline.answer("Remote work?")

    assert retriever.calls[0]["min_score"] == pytest.approx(0.4)


def test_pipeline_passes_filters_to_retriever() -> None:
    retriever = StubRetriever([_result()])
    filters = SearchFilters(
        departments=("HR",),
    )

    pipeline = GroundedRAGPipeline(
        retriever=retriever,
        generator=DeterministicAnswerGenerator(),
    )

    pipeline.answer(
        "Remote work?",
        filters=filters,
    )

    assert retriever.calls[0]["filters"] == filters


def test_pipeline_records_runtime_metadata() -> None:
    pipeline = GroundedRAGPipeline(
        retriever=StubRetriever([_result()]),
        generator=DeterministicAnswerGenerator(),
    )

    response = pipeline.answer("Remote work?")

    assert response.metadata["retrieved_count"] == 1
    assert response.metadata["context_count"] == 1
    assert response.metadata["context_truncated"] is False
    assert response.metadata["cited_source_ids"] == ("S1",)
    assert response.metadata["uncited_source_ids"] == ()


def test_pipeline_rejects_uncited_generation() -> None:
    pipeline = GroundedRAGPipeline(
        retriever=StubRetriever([_result()]),
        generator=UncitedGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="at least one citation",
    ):
        pipeline.answer("Remote work?")


@pytest.mark.parametrize(
    "value",
    [0, -1, True],
)
def test_config_rejects_invalid_top_k(value: object) -> None:
    with pytest.raises(
        RAGPipelineError,
        match="top_k",
    ):
        RAGPipelineConfig(top_k=value)


@pytest.mark.parametrize(
    "value",
    [True, "0.5"],
)
def test_config_rejects_invalid_min_score(
    value: object,
) -> None:
    with pytest.raises(
        RAGPipelineError,
        match="min_score",
    ):
        RAGPipelineConfig(min_score=value)


def test_config_normalizes_numeric_min_score() -> None:
    config = RAGPipelineConfig(min_score=1)

    assert config.min_score == 1.0


def test_pipeline_rejects_invalid_query_type() -> None:
    pipeline = GroundedRAGPipeline(
        retriever=StubRetriever([]),
        generator=DeterministicAnswerGenerator(),
    )

    with pytest.raises(
        RAGPipelineError,
        match="string or RAGQuery",
    ):
        pipeline.answer(123)


def test_pipeline_rejects_invalid_retriever() -> None:
    with pytest.raises(
        RAGPipelineError,
        match="retriever",
    ):
        GroundedRAGPipeline(
            retriever=object(),
            generator=DeterministicAnswerGenerator(),
        )


def test_pipeline_rejects_invalid_generator() -> None:
    with pytest.raises(
        RAGPipelineError,
        match="generator",
    ):
        GroundedRAGPipeline(
            retriever=StubRetriever([]),
            generator=object(),
        )


def test_pipeline_is_deterministic() -> None:
    pipeline = GroundedRAGPipeline(
        retriever=StubRetriever([_result()]),
        generator=DeterministicAnswerGenerator(),
    )

    first = pipeline.answer("Remote work?")
    second = pipeline.answer("Remote work?")

    assert first == second


def test_multiple_results_flow_through_context() -> None:
    retriever = StubRetriever(
        [
            _result("chunk-1", rank=1),
            _result(
                "chunk-2",
                rank=2,
                text="Managers approve remote schedules.",
            ),
        ]
    )

    pipeline = GroundedRAGPipeline(
        retriever=retriever,
        generator=DeterministicAnswerGenerator(),
    )

    response = pipeline.answer("Remote work?")

    assert response.evidence_count == 2
    assert response.metadata["retrieved_count"] == 2
    assert response.metadata["uncited_source_ids"] == ("S2",)
