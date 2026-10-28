"""Tests for EnterpriseIQ document chunking."""

from datetime import date
from pathlib import Path

import pytest

from enterpriseiq.domain import EnterpriseDocument
from enterpriseiq.ingestion import (
    ChunkingConfig,
    chunk_document,
    chunk_documents,
    load_document_directory,
)


def _make_document(
    content: str,
    *,
    document_id: str = "TEST-001",
) -> EnterpriseDocument:
    return EnterpriseDocument(
        document_id=document_id,
        title="Test Policy",
        department="Test",
        document_type="policy",
        version="2026.1",
        effective_date=date(2026, 1, 1),
        owner="Test Owner",
        classification="internal",
        content=content,
        source_path=Path("test_policy.md"),
        metadata={"language": "en"},
    )


def test_chunking_config_defaults() -> None:
    config = ChunkingConfig()

    assert config.chunk_size == 1000
    assert config.chunk_overlap == 150
    assert config.min_chunk_size == 100


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_overlap": -1}, "chunk_overlap"),
        (
            {"chunk_size": 100, "chunk_overlap": 100},
            "chunk_overlap",
        ),
        (
            {"chunk_size": 100, "chunk_overlap": 10, "min_chunk_size": 101},
            "min_chunk_size",
        ),
    ],
)
def test_chunking_config_rejects_invalid_values(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ChunkingConfig(**kwargs)


def test_short_document_produces_single_chunk() -> None:
    document = _make_document("Short enterprise policy.")

    chunks = chunk_document(
        document,
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=20,
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "TEST-001::0000"
    assert chunks[0].text == document.content
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(document.content)


def test_chunk_ids_are_deterministic() -> None:
    document = _make_document(
        "Alpha section contains policy details. "
        "Beta section contains additional requirements. "
        "Gamma section contains operational guidance. "
        "Delta section contains final controls."
    )

    config = ChunkingConfig(
        chunk_size=70,
        chunk_overlap=15,
        min_chunk_size=30,
    )

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]

    assert [chunk.text for chunk in first] == [
        chunk.text for chunk in second
    ]


def test_chunk_metadata_preserves_document_provenance() -> None:
    document = _make_document(
        "Policy requirement one. "
        "Policy requirement two. "
        "Policy requirement three."
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(
            chunk_size=40,
            chunk_overlap=10,
            min_chunk_size=20,
        ),
    )

    assert chunks

    for index, chunk in enumerate(chunks):
        assert chunk.document_id == document.document_id
        assert chunk.title == document.title
        assert chunk.department == document.department
        assert chunk.version == document.version
        assert chunk.owner == document.owner
        assert chunk.classification == document.classification
        assert chunk.chunk_index == index
        assert chunk.metadata["language"] == "en"
        assert (
            chunk.metadata["chunking_strategy"]
            == "boundary_aware_character"
        )


def test_chunk_offsets_map_back_to_source_text() -> None:
    document = _make_document(
        "First paragraph explains the policy requirement.\n\n"
        "Second paragraph explains another requirement.\n\n"
        "Third paragraph contains final guidance."
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(
            chunk_size=75,
            chunk_overlap=15,
            min_chunk_size=30,
        ),
    )

    for chunk in chunks:
        assert (
            document.content[chunk.start_char : chunk.end_char]
            == chunk.text
        )


def test_adjacent_chunks_overlap() -> None:
    document = _make_document(
        "Alpha policy requirement is important. "
        "Beta policy requirement is also important. "
        "Gamma operational requirement must be followed. "
        "Delta control requirement completes the policy."
    )

    chunks = chunk_document(
        document,
        ChunkingConfig(
            chunk_size=80,
            chunk_overlap=20,
            min_chunk_size=35,
        ),
    )

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start_char < previous.end_char


def test_chunks_do_not_exceed_configured_size() -> None:
    document = _make_document(
        " ".join(f"word{i}" for i in range(200))
    )

    config = ChunkingConfig(
        chunk_size=120,
        chunk_overlap=20,
        min_chunk_size=60,
    )

    chunks = chunk_document(document, config)

    assert chunks
    assert all(len(chunk.text) <= config.chunk_size for chunk in chunks)


def test_multiple_documents_preserve_input_order() -> None:
    first = _make_document(
        "First document content.",
        document_id="DOC-001",
    )
    second = _make_document(
        "Second document content.",
        document_id="DOC-002",
    )

    chunks = chunk_documents(
        [first, second],
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=20,
        ),
    )

    assert [chunk.document_id for chunk in chunks] == [
        "DOC-001",
        "DOC-002",
    ]


def test_real_corpus_can_be_chunked() -> None:
    documents = load_document_directory(Path("data/raw"))

    chunks = chunk_documents(
        documents,
        ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    assert len(documents) == 15
    assert len(chunks) >= 15
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    source_ids = {document.document_id for document in documents}
    chunk_source_ids = {chunk.document_id for chunk in chunks}

    assert chunk_source_ids == source_ids


def test_real_corpus_chunk_offsets_are_traceable() -> None:
    documents = load_document_directory(Path("data/raw"))

    chunks = chunk_documents(
        documents,
        ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    document_lookup = {
        document.document_id: document
        for document in documents
    }

    for chunk in chunks:
        source = document_lookup[chunk.document_id]

        assert (
            source.content[chunk.start_char : chunk.end_char]
            == chunk.text
        )


