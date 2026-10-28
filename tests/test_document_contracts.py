"""Tests for EnterpriseIQ document domain contracts."""

from datetime import date
from pathlib import Path

import pytest

from enterpriseiq.domain import DocumentChunk, EnterpriseDocument


@pytest.fixture
def document() -> EnterpriseDocument:
    return EnterpriseDocument(
        document_id="HR-003",
        title="Remote Work Policy",
        department="HR",
        document_type="policy",
        version="2026.1",
        effective_date=date(2026, 2, 1),
        owner="People Operations",
        classification="Internal",
        content="Eligible employees may work remotely for up to three days.",
        source_path=Path("data/raw/hr/remote_work_policy.md"),
        metadata={"language": "en"},
    )


def test_enterprise_document_normalizes_values() -> None:
    document = EnterpriseDocument(
        document_id=" HR-003 ",
        title=" Remote Work Policy ",
        department=" HR ",
        document_type=" policy ",
        version=" 2026.1 ",
        effective_date=date(2026, 2, 1),
        owner=" People Operations ",
        classification=" INTERNAL ",
        content=" Policy content ",
        source_path=Path("remote_work_policy.md"),
    )

    assert document.document_id == "HR-003"
    assert document.title == "Remote Work Policy"
    assert document.classification == "internal"
    assert document.content == "Policy content"


def test_enterprise_document_rejects_empty_required_text() -> None:
    with pytest.raises(ValueError, match="title must not be empty"):
        EnterpriseDocument(
            document_id="HR-003",
            title="   ",
            department="HR",
            document_type="policy",
            version="2026.1",
            effective_date=date(2026, 2, 1),
            owner="People Operations",
            classification="internal",
            content="Content",
            source_path=Path("policy.md"),
        )


def test_enterprise_document_rejects_unknown_classification() -> None:
    with pytest.raises(ValueError, match="classification must be one of"):
        EnterpriseDocument(
            document_id="HR-003",
            title="Remote Work Policy",
            department="HR",
            document_type="policy",
            version="2026.1",
            effective_date=date(2026, 2, 1),
            owner="People Operations",
            classification="top-secret",
            content="Content",
            source_path=Path("policy.md"),
        )


def test_enterprise_document_requires_date_type() -> None:
    with pytest.raises(TypeError, match="effective_date"):
        EnterpriseDocument(
            document_id="HR-003",
            title="Remote Work Policy",
            department="HR",
            document_type="policy",
            version="2026.1",
            effective_date="2026-02-01",  # type: ignore[arg-type]
            owner="People Operations",
            classification="internal",
            content="Content",
            source_path=Path("policy.md"),
        )


def test_enterprise_document_metadata_is_defensively_copied() -> None:
    metadata = {"language": "en"}

    document = EnterpriseDocument(
        document_id="HR-003",
        title="Remote Work Policy",
        department="HR",
        document_type="policy",
        version="2026.1",
        effective_date=date(2026, 2, 1),
        owner="People Operations",
        classification="internal",
        content="Content",
        source_path=Path("policy.md"),
        metadata=metadata,
    )

    metadata["language"] = "fr"

    assert document.metadata["language"] == "en"

    with pytest.raises(TypeError):
        document.metadata["language"] = "de"  # type: ignore[index]


def test_document_citation_label(document: EnterpriseDocument) -> None:
    assert (
        document.citation_label
        == "HR-003 — Remote Work Policy (v2026.1)"
    )


def test_chunk_inherits_document_provenance(
    document: EnterpriseDocument,
) -> None:
    chunk = DocumentChunk.from_document(
        document=document,
        chunk_id="HR-003::0000",
        text=document.content,
        chunk_index=0,
        start_char=0,
        end_char=len(document.content),
        metadata={"section": "Remote Work Limit"},
    )

    assert chunk.document_id == document.document_id
    assert chunk.title == document.title
    assert chunk.department == document.department
    assert chunk.version == document.version
    assert chunk.classification == "internal"
    assert chunk.metadata["language"] == "en"
    assert chunk.metadata["section"] == "Remote Work Limit"


def test_chunk_rejects_negative_index(
    document: EnterpriseDocument,
) -> None:
    with pytest.raises(ValueError, match="chunk_index"):
        DocumentChunk.from_document(
            document=document,
            chunk_id="HR-003::0000",
            text="Example",
            chunk_index=-1,
            start_char=0,
            end_char=7,
        )


def test_chunk_rejects_invalid_character_range(
    document: EnterpriseDocument,
) -> None:
    with pytest.raises(
        ValueError,
        match="end_char must be greater than start_char",
    ):
        DocumentChunk.from_document(
            document=document,
            chunk_id="HR-003::0000",
            text="Example",
            chunk_index=0,
            start_char=10,
            end_char=5,
        )


def test_chunk_citation_label(
    document: EnterpriseDocument,
) -> None:
    chunk = DocumentChunk.from_document(
        document=document,
        chunk_id="HR-003::0000",
        text="Example",
        chunk_index=0,
        start_char=0,
        end_char=7,
    )

    assert chunk.citation_label == (
        "HR-003 — Remote Work Policy (v2026.1, chunk 0)"
    )


def test_contracts_are_immutable(
    document: EnterpriseDocument,
) -> None:
    with pytest.raises(AttributeError):
        document.title = "Changed"  # type: ignore[misc]
