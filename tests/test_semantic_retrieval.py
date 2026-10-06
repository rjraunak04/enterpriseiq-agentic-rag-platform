"""Tests for metadata-aware EnterpriseIQ semantic retrieval."""

import math

import pytest

from enterpriseiq.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingConfig,
)
from enterpriseiq.retrieval import (
    PersistentVectorIndex,
    RetrievalError,
    SearchFilters,
    SemanticRetriever,
    VectorIndexRecord,
    cosine_similarity,
)


def make_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="deterministic",
        model_name="semantic-test-model",
        dimension=12,
        batch_size=4,
        normalize=True,
    )


def make_record(
    provider: DeterministicEmbeddingProvider,
    *,
    chunk_id: str,
    document_id: str,
    text: str,
    department: str,
    classification: str = "internal",
    document_type: str = "policy",
) -> VectorIndexRecord:
    return VectorIndexRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        title=f"{department} Document",
        department=department,
        document_type=document_type,
        version="1.0",
        effective_date="2026-01-01",
        owner=f"{department} Owner",
        classification=classification,
        source_path=f"data/raw/{department.lower()}/document.md",
        metadata={"source": "test"},
        vector=provider.embed_query(text),
    )


def make_retriever() -> tuple[
    SemanticRetriever,
    DeterministicEmbeddingProvider,
]:
    config = make_config()
    provider = DeterministicEmbeddingProvider(config)

    records = [
        make_record(
            provider,
            chunk_id="HR-003::0000",
            document_id="HR-003",
            text="Employees may work remotely three days per workweek.",
            department="HR",
        ),
        make_record(
            provider,
            chunk_id="FIN-001::0000",
            document_id="FIN-001",
            text="Expense receipts are required for reimbursement.",
            department="Finance",
        ),
        make_record(
            provider,
            chunk_id="SEC-001::0000",
            document_id="SEC-001",
            text="Security incidents must be reported immediately.",
            department="Security",
            classification="confidential",
        ),
    ]

    index = PersistentVectorIndex(
        dimension=config.dimension,
        model_name=config.model_name,
        records=records,
    )

    return (
        SemanticRetriever(
            index=index,
            provider=provider,
        ),
        provider,
    )


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity(
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0),
    ) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity(
        (1.0, 0.0),
        (0.0, 1.0),
    ) == pytest.approx(0.0)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(
        RetrievalError,
        match="equal vector dimensions",
    ):
        cosine_similarity(
            (1.0, 2.0),
            (1.0,),
        )


def test_cosine_similarity_rejects_zero_vector() -> None:
    with pytest.raises(
        RetrievalError,
        match="zero vectors",
    ):
        cosine_similarity(
            (0.0, 0.0),
            (1.0, 0.0),
        )


def test_semantic_search_returns_ranked_results() -> None:
    retriever, _ = make_retriever()

    results = retriever.search(
        "Employees may work remotely three days per workweek.",
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].chunk_id == "HR-003::0000"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1.0)
    assert results[0].citation_label == (
        "HR Document (HR-003)"
    )

    assert all(
        results[index].score
        >= results[index + 1].score
        for index in range(len(results) - 1)
    )


def test_search_respects_top_k() -> None:
    retriever, _ = make_retriever()

    results = retriever.search(
        "expense reimbursement",
        top_k=2,
    )

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]


def test_department_filter_is_case_insensitive() -> None:
    retriever, _ = make_retriever()

    results = retriever.search(
        "company policy",
        filters=SearchFilters(
            departments=("finance",),
        ),
    )

    assert len(results) == 1
    assert results[0].department == "Finance"
    assert results[0].document_id == "FIN-001"


def test_classification_filter() -> None:
    retriever, _ = make_retriever()

    results = retriever.search(
        "security incident",
        filters=SearchFilters(
            classifications=("CONFIDENTIAL",),
        ),
    )

    assert len(results) == 1
    assert results[0].document_id == "SEC-001"


def test_document_id_filter() -> None:
    retriever, _ = make_retriever()

    results = retriever.search(
        "policy",
        filters=SearchFilters(
            document_ids=("HR-003",),
        ),
    )

    assert len(results) == 1
    assert results[0].document_id == "HR-003"


def test_document_type_filter() -> None:
    retriever, _ = make_retriever()

    results = retriever.search(
        "policy",
        filters=SearchFilters(
            document_types=("POLICY",),
        ),
    )

    assert len(results) == 3


def test_filter_without_matches_returns_empty() -> None:
    retriever, _ = make_retriever()

    results = retriever.search(
        "policy",
        filters=SearchFilters(
            departments=("Legal",),
        ),
    )

    assert results == []


def test_min_score_filters_results() -> None:
    retriever, _ = make_retriever()

    query = (
        "Employees may work remotely three days per workweek."
    )

    results = retriever.search(
        query,
        top_k=3,
        min_score=0.999,
    )

    assert len(results) == 1
    assert results[0].document_id == "HR-003"


def test_result_preserves_enterprise_metadata() -> None:
    retriever, _ = make_retriever()

    result = retriever.search(
        "Security incidents must be reported immediately.",
        top_k=1,
    )[0]

    assert result.document_id == "SEC-001"
    assert result.department == "Security"
    assert result.metadata["classification"] == "confidential"
    assert result.metadata["version"] == "1.0"
    assert result.metadata["owner"] == "Security Owner"


@pytest.mark.parametrize("query", ["", " ", "\n"])
def test_search_rejects_empty_query(query: str) -> None:
    retriever, _ = make_retriever()

    with pytest.raises(
        RetrievalError,
        match="query must not be empty",
    ):
        retriever.search(query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_invalid_top_k(top_k: int) -> None:
    retriever, _ = make_retriever()

    with pytest.raises(
        RetrievalError,
        match="greater than zero",
    ):
        retriever.search(
            "remote work",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "min_score",
    [-1.1, 1.1, math.inf, -math.inf],
)
def test_search_rejects_invalid_min_score(
    min_score: float,
) -> None:
    retriever, _ = make_retriever()

    with pytest.raises(RetrievalError):
        retriever.search(
            "remote work",
            min_score=min_score,
        )


def test_retriever_rejects_dimension_mismatch() -> None:
    config = make_config()
    provider = DeterministicEmbeddingProvider(config)

    incompatible_index = PersistentVectorIndex(
        dimension=6,
        model_name=config.model_name,
    )

    with pytest.raises(
        RetrievalError,
        match="dimension mismatch",
    ):
        SemanticRetriever(
            index=incompatible_index,
            provider=provider,
        )


def test_retriever_rejects_model_mismatch() -> None:
    config = make_config()
    provider = DeterministicEmbeddingProvider(config)

    incompatible_index = PersistentVectorIndex(
        dimension=config.dimension,
        model_name="different-model",
    )

    with pytest.raises(
        RetrievalError,
        match="model mismatch",
    ):
        SemanticRetriever(
            index=incompatible_index,
            provider=provider,
        )
