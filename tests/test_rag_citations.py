"""Tests for grounded citation parsing and validation."""

from pathlib import Path

import pytest

from enterpriseiq.rag import (
    CitationValidationError,
    CitationValidationResult,
    CitationValidator,
    EvidenceContext,
    EvidenceItem,
    GeneratedAnswer,
)


def _item(
    source_id: str,
    *,
    document_id: str,
    rank: int,
) -> EvidenceItem:
    return EvidenceItem(
        source_id=source_id,
        chunk_id=f"chunk-{source_id}",
        document_id=document_id,
        text=f"Evidence for {source_id}.",
        title=f"Policy {source_id}",
        department="HR",
        source_path=Path(f"policies/{document_id}.md"),
        retrieval_score=1.0 / rank,
        retrieval_rank=rank,
        metadata={"classification": "internal"},
    )


def _context() -> EvidenceContext:
    first = _item(
        "S1",
        document_id="HR-001",
        rank=1,
    )
    second = _item(
        "S2",
        document_id="HR-002",
        rank=2,
    )

    return EvidenceContext(
        items=(first, second),
        rendered_text=(
            "[S1]\nEvidence: Evidence for S1.\n\n"
            "[S2]\nEvidence: Evidence for S2."
        ),
    )


def test_extracts_single_citation() -> None:
    ids = CitationValidator.extract_source_ids(
        "Remote work is allowed [S1]."
    )

    assert ids == ("S1",)


def test_extracts_citations_in_occurrence_order() -> None:
    ids = CitationValidator.extract_source_ids(
        "First [S2]. Then [S1]."
    )

    assert ids == ("S2", "S1")


def test_extract_preserves_duplicate_occurrences() -> None:
    ids = CitationValidator.extract_source_ids(
        "First [S1]. Again [S1]."
    )

    assert ids == ("S1", "S1")


def test_validation_deduplicates_citation_objects() -> None:
    result = CitationValidator().validate(
        GeneratedAnswer(
            text="First [S1]. Again [S1]."
        ),
        _context(),
    )

    assert result.cited_source_ids == ("S1",)
    assert len(result.citations) == 1


def test_valid_citation_maps_to_exact_evidence() -> None:
    result = CitationValidator().validate(
        GeneratedAnswer(
            text="Supported claim [S1]."
        ),
        _context(),
    )

    citation = result.citations[0]

    assert citation.source_id == "S1"
    assert citation.chunk_id == "chunk-S1"
    assert citation.document_id == "HR-001"
    assert citation.title == "Policy S1"
    assert citation.source_path == Path(
        "policies/HR-001.md"
    )


def test_multiple_valid_citations_are_supported() -> None:
    result = CitationValidator().validate(
        GeneratedAnswer(
            text="Claim one [S1]. Claim two [S2]."
        ),
        _context(),
    )

    assert result.cited_source_ids == ("S1", "S2")
    assert len(result.citations) == 2


def test_unknown_citation_is_rejected() -> None:
    with pytest.raises(
        CitationValidationError,
        match="unknown source IDs",
    ):
        CitationValidator().validate(
            GeneratedAnswer(
                text="Unsupported source [S99]."
            ),
            _context(),
        )


@pytest.mark.parametrize(
    "citation",
    [
        "[S]",
        "[S0]",
        "[S01]",
        "[S-1]",
        "[SABC]",
        "[S1x]",
    ],
)
def test_malformed_citations_are_rejected(
    citation: str,
) -> None:
    with pytest.raises(
        CitationValidationError,
        match="malformed",
    ):
        CitationValidator().validate(
            GeneratedAnswer(
                text=f"Claim {citation}."
            ),
            _context(),
        )


def test_non_abstained_answer_requires_citation() -> None:
    with pytest.raises(
        CitationValidationError,
        match="at least one citation",
    ):
        CitationValidator().validate(
            GeneratedAnswer(
                text="An uncited factual answer."
            ),
            _context(),
        )


def test_non_abstained_answer_requires_evidence() -> None:
    with pytest.raises(
        CitationValidationError,
        match="requires evidence",
    ):
        CitationValidator().validate(
            GeneratedAnswer(
                text="Answer [S1]."
            ),
            EvidenceContext(
                items=(),
                rendered_text="",
            ),
        )


def test_abstention_without_citations_is_valid() -> None:
    result = CitationValidator().validate(
        GeneratedAnswer(
            text="Insufficient evidence.",
            abstained=True,
        ),
        _context(),
    )

    assert result.citations == ()
    assert result.cited_source_ids == ()
    assert result.uncited_source_ids == ("S1", "S2")


def test_abstention_with_citation_is_rejected() -> None:
    with pytest.raises(
        CitationValidationError,
        match="must not contain citations",
    ):
        CitationValidator().validate(
            GeneratedAnswer(
                text="Insufficient evidence [S1].",
                abstained=True,
            ),
            _context(),
        )


def test_uncited_evidence_is_reported() -> None:
    result = CitationValidator().validate(
        GeneratedAnswer(
            text="Supported [S1]."
        ),
        _context(),
    )

    assert result.uncited_source_ids == ("S2",)


def test_all_cited_evidence_leaves_no_uncited_sources() -> None:
    result = CitationValidator().validate(
        GeneratedAnswer(
            text="Combined support [S1] [S2]."
        ),
        _context(),
    )

    assert result.uncited_source_ids == ()


def test_result_is_typed_contract() -> None:
    result = CitationValidator().validate(
        GeneratedAnswer(
            text="Supported [S1]."
        ),
        _context(),
    )

    assert isinstance(result, CitationValidationResult)


def test_validation_is_deterministic() -> None:
    validator = CitationValidator()
    answer = GeneratedAnswer(
        text="Second [S2], first [S1]."
    )
    context = _context()

    first = validator.validate(answer, context)
    second = validator.validate(answer, context)

    assert first == second


def test_citation_order_follows_answer_not_context() -> None:
    result = CitationValidator().validate(
        GeneratedAnswer(
            text="Second source [S2]. First source [S1]."
        ),
        _context(),
    )

    assert result.cited_source_ids == ("S2", "S1")
    assert [
        citation.source_id
        for citation in result.citations
    ] == ["S2", "S1"]


def test_invalid_answer_type_is_rejected() -> None:
    with pytest.raises(
        CitationValidationError,
        match="GeneratedAnswer",
    ):
        CitationValidator().validate(
            "invalid",
            _context(),
        )


def test_invalid_context_type_is_rejected() -> None:
    with pytest.raises(
        CitationValidationError,
        match="EvidenceContext",
    ):
        CitationValidator().validate(
            GeneratedAnswer(
                text="Supported [S1]."
            ),
            "invalid",
        )


def test_extract_rejects_non_string_input() -> None:
    with pytest.raises(
        CitationValidationError,
        match="string",
    ):
        CitationValidator.extract_source_ids(123)


def test_plain_brackets_are_not_source_citations() -> None:
    result = CitationValidator().validate(
        GeneratedAnswer(
            text="The policy [internal] is supported [S1]."
        ),
        _context(),
    )

    assert result.cited_source_ids == ("S1",)


def test_source_provenance_is_not_taken_from_answer_text() -> None:
    result = CitationValidator().validate(
        GeneratedAnswer(
            text=(
                "Claim [S1]. "
                "Document: FAKE-999, Title: Fake Policy."
            )
        ),
        _context(),
    )

    citation = result.citations[0]

    assert citation.document_id == "HR-001"
    assert citation.title == "Policy S1"
