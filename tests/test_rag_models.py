"""Tests for grounded RAG contracts."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from enterpriseiq.rag import (
    Citation,
    EvidenceContext,
    EvidenceItem,
    GeneratedAnswer,
    GroundedResponse,
    RAGContractError,
    RAGQuery,
)


def _evidence(
    source_id: str = "S1",
    *,
    rank: int = 1,
) -> EvidenceItem:
    return EvidenceItem(
        source_id=source_id,
        chunk_id=f"chunk-{source_id}",
        document_id="HR-003",
        text="Employees may work remotely.",
        title="Remote Work Policy",
        department="HR",
        source_path=Path("hr/remote_work_policy.md"),
        retrieval_score=0.91,
        retrieval_rank=rank,
        metadata={"classification": "internal"},
    )


def _citation(source_id: str = "S1") -> Citation:
    return Citation(
        source_id=source_id,
        chunk_id=f"chunk-{source_id}",
        document_id="HR-003",
        title="Remote Work Policy",
        source_path=Path("hr/remote_work_policy.md"),
    )


def test_rag_query_normalizes_text() -> None:
    query = RAGQuery(text="  What is the remote policy?  ")

    assert query.text == "What is the remote policy?"


def test_rag_query_rejects_empty_text() -> None:
    with pytest.raises(RAGContractError):
        RAGQuery(text="   ")


def test_query_metadata_is_immutable() -> None:
    query = RAGQuery(
        text="Question",
        metadata={"tenant": "demo"},
    )

    with pytest.raises(TypeError):
        query.metadata["tenant"] = "changed"


def test_evidence_item_is_valid() -> None:
    item = _evidence()

    assert item.source_id == "S1"
    assert item.retrieval_score == pytest.approx(0.91)


def test_evidence_rejects_invalid_rank() -> None:
    with pytest.raises(RAGContractError):
        _evidence(rank=0)


def test_evidence_rejects_non_path_source() -> None:
    with pytest.raises(RAGContractError):
        EvidenceItem(
            source_id="S1",
            chunk_id="chunk-1",
            document_id="HR-003",
            text="Evidence",
            title="Policy",
            department="HR",
            source_path="policy.md",
            retrieval_score=1.0,
            retrieval_rank=1,
        )


def test_evidence_metadata_is_immutable() -> None:
    item = _evidence()

    with pytest.raises(TypeError):
        item.metadata["classification"] = "public"


def test_context_accepts_unique_sources() -> None:
    context = EvidenceContext(
        items=(_evidence("S1"), _evidence("S2", rank=2)),
        rendered_text="[S1] first\n[S2] second",
    )

    assert len(context.items) == 2


def test_context_rejects_duplicate_source_ids() -> None:
    with pytest.raises(RAGContractError):
        EvidenceContext(
            items=(_evidence("S1"), _evidence("S1", rank=2)),
            rendered_text="duplicate",
        )


def test_context_can_be_empty() -> None:
    context = EvidenceContext(
        items=(),
        rendered_text="",
    )

    assert context.items == ()


def test_generated_answer_normalizes_text() -> None:
    answer = GeneratedAnswer(
        text="  Remote work is allowed [S1].  "
    )

    assert answer.text == "Remote work is allowed [S1]."


def test_generated_answer_rejects_empty_non_abstention() -> None:
    with pytest.raises(RAGContractError):
        GeneratedAnswer(text="")


def test_generated_answer_allows_empty_abstention() -> None:
    answer = GeneratedAnswer(
        text="",
        abstained=True,
    )

    assert answer.abstained is True


def test_citation_preserves_provenance() -> None:
    citation = _citation()

    assert citation.document_id == "HR-003"
    assert citation.source_path == Path(
        "hr/remote_work_policy.md"
    )


def test_grounded_response_is_valid() -> None:
    response = GroundedResponse(
        query=RAGQuery(text="Remote work?"),
        answer="Remote work is allowed [S1].",
        citations=(_citation(),),
        grounded=True,
        abstained=False,
        evidence_count=1,
    )

    assert response.grounded is True
    assert len(response.citations) == 1


def test_abstained_response_cannot_be_grounded() -> None:
    with pytest.raises(RAGContractError):
        GroundedResponse(
            query=RAGQuery(text="Unknown?"),
            answer="Insufficient evidence.",
            citations=(),
            grounded=True,
            abstained=True,
            evidence_count=0,
        )


def test_citations_cannot_exceed_evidence_count() -> None:
    with pytest.raises(RAGContractError):
        GroundedResponse(
            query=RAGQuery(text="Question"),
            answer="Answer [S1].",
            citations=(_citation(),),
            grounded=True,
            abstained=False,
            evidence_count=0,
        )


def test_models_are_frozen() -> None:
    query = RAGQuery(text="Question")

    with pytest.raises(FrozenInstanceError):
        query.text = "Changed"
