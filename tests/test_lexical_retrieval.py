"""Tests for BM25 lexical enterprise retrieval."""

import pytest

from enterpriseiq.retrieval.lexical import (
    BM25Config,
    BM25Retriever,
    LexicalRetrievalError,
    tokenize,
)
from enterpriseiq.retrieval.semantic import SearchFilters
from enterpriseiq.retrieval.vector_index import VectorIndexRecord


def make_record(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    department: str = "HR",
    classification: str = "internal",
    document_type: str = "policy",
) -> VectorIndexRecord:
    return VectorIndexRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        title=f"{document_id} title",
        department=department,
        document_type=document_type,
        version="1.0",
        effective_date="2026-01-01",
        owner="EnterpriseIQ",
        classification=classification,
        source_path=f"data/raw/{document_id}.md",
        metadata={
            "department": department,
            "classification": classification,
        },
        vector=(1.0, 0.0),
    )


@pytest.fixture
def records() -> list[VectorIndexRecord]:
    return [
        make_record(
            "HR-003::0000",
            "HR-003",
            (
                "Employees may work remotely up to three days "
                "per workweek under the remote work policy."
            ),
            department="HR",
        ),
        make_record(
            "FIN-001::0000",
            "FIN-001",
            (
                "A receipt is required for expense reimbursement "
                "claims of INR 1,500 or more."
            ),
            department="Finance",
        ),
        make_record(
            "PROD-001::0000",
            "PROD-001",
            (
                "The enterprise service SLA defines a P1 incident "
                "response target of 15 minutes."
            ),
            department="Product",
            document_type="sla",
        ),
        make_record(
            "IT-001::0000",
            "IT-001",
            (
                "Employees must use the corporate VPN when accessing "
                "internal systems remotely."
            ),
            department="IT",
            document_type="guide",
        ),
    ]


def test_tokenize_is_case_insensitive_and_deterministic() -> None:
    assert tokenize("Remote WORK Policy") == (
        "remote",
        "work",
        "policy",
    )


def test_tokenize_preserves_numeric_policy_terms() -> None:
    tokens = tokenize("INR 1,500 and P1 within 15 minutes")

    assert "1,500" in tokens
    assert "p1" in tokens
    assert "15" in tokens


def test_bm25_returns_exact_policy_match_first(
    records: list[VectorIndexRecord],
) -> None:
    retriever = BM25Retriever(records)

    results = retriever.search(
        "remote work three days",
        top_k=3,
    )

    assert results
    assert results[0].document_id == "HR-003"
    assert results[0].chunk_id == "HR-003::0000"
    assert results[0].score > 0
    assert results[0].rank == 1


def test_bm25_retrieves_numeric_expense_policy(
    records: list[VectorIndexRecord],
) -> None:
    retriever = BM25Retriever(records)

    results = retriever.search(
        "receipt INR 1,500 expense reimbursement",
        top_k=2,
    )

    assert results[0].document_id == "FIN-001"


def test_bm25_retrieves_sla_terms(
    records: list[VectorIndexRecord],
) -> None:
    retriever = BM25Retriever(records)

    results = retriever.search(
        "P1 response 15 minutes",
        top_k=2,
    )

    assert results[0].document_id == "PROD-001"


def test_bm25_supports_department_filter(
    records: list[VectorIndexRecord],
) -> None:
    retriever = BM25Retriever(records)

    results = retriever.search(
        "remote employees internal",
        filters=SearchFilters(departments=("IT",)),
    )

    assert results
    assert all(result.department == "IT" for result in results)


def test_bm25_supports_document_type_filter(
    records: list[VectorIndexRecord],
) -> None:
    retriever = BM25Retriever(records)

    results = retriever.search(
        "incident response",
        filters=SearchFilters(document_types=("sla",)),
    )

    assert len(results) == 1
    assert results[0].document_id == "PROD-001"


def test_bm25_unknown_terms_return_no_results(
    records: list[VectorIndexRecord],
) -> None:
    retriever = BM25Retriever(records)

    assert retriever.search("quantumfoobarxyz") == []


def test_bm25_empty_corpus_returns_no_results() -> None:
    retriever = BM25Retriever([])

    assert retriever.search("remote work") == []


@pytest.mark.parametrize(
    "query",
    ["", " ", "\n"],
)
def test_bm25_rejects_empty_query(
    records: list[VectorIndexRecord],
    query: str,
) -> None:
    retriever = BM25Retriever(records)

    with pytest.raises(
        LexicalRetrievalError,
        match="non-empty string",
    ):
        retriever.search(query)


@pytest.mark.parametrize(
    "top_k",
    [0, -1, True, 1.5],
)
def test_bm25_rejects_invalid_top_k(
    records: list[VectorIndexRecord],
    top_k: object,
) -> None:
    retriever = BM25Retriever(records)

    with pytest.raises(
        LexicalRetrievalError,
        match="positive integer",
    ):
        retriever.search("remote", top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "min_score",
    [-1.0, float("nan"), float("inf"), True],
)
def test_bm25_rejects_invalid_min_score(
    records: list[VectorIndexRecord],
    min_score: object,
) -> None:
    retriever = BM25Retriever(records)

    with pytest.raises(
        LexicalRetrievalError,
        match="finite non-negative",
    ):
        retriever.search(
            "remote",
            min_score=min_score,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("k1", "b"),
    [
        (0.0, 0.75),
        (-1.0, 0.75),
        (1.5, -0.1),
        (1.5, 1.1),
        (float("nan"), 0.75),
        (1.5, float("inf")),
    ],
)
def test_bm25_rejects_invalid_configuration(
    k1: float,
    b: float,
) -> None:
    with pytest.raises(LexicalRetrievalError):
        BM25Config(k1=k1, b=b)


def test_bm25_rejects_duplicate_chunk_ids() -> None:
    record = make_record(
        "HR-003::0000",
        "HR-003",
        "remote work policy",
    )

    with pytest.raises(
        LexicalRetrievalError,
        match="duplicate chunk_id",
    ):
        BM25Retriever([record, record])


def test_bm25_ties_are_deterministic() -> None:
    records = [
        make_record(
            "B::0000",
            "B",
            "remote policy",
        ),
        make_record(
            "A::0000",
            "A",
            "remote policy",
        ),
    ]

    results = BM25Retriever(records).search("remote")

    assert [result.chunk_id for result in results] == [
        "A::0000",
        "B::0000",
    ]


def test_bm25_respects_top_k(
    records: list[VectorIndexRecord],
) -> None:
    results = BM25Retriever(records).search(
        "employees remote",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].rank == 1
