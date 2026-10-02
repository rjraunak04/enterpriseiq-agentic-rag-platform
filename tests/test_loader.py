"""Tests for metadata-aware enterprise document loading."""

from pathlib import Path

import pytest

from enterpriseiq.ingestion import (
    DocumentLoadError,
    load_document_directory,
    load_markdown_document,
    parse_front_matter,
)


def _document_text(
    *,
    document_id: str = "HR-003",
    effective_date: str = "2026-02-01",
    extra_metadata: str = "",
    body: str = "# Remote Work Policy\n\nEmployees may work remotely.",
) -> str:
    extra = f"{extra_metadata.rstrip()}\n" if extra_metadata else ""

    return (
        "---\n"
        f"document_id: {document_id}\n"
        "title: Remote Work Policy\n"
        "department: HR\n"
        "document_type: policy\n"
        'version: "2026.1"\n'
        f'effective_date: "{effective_date}"\n'
        "owner: People Operations\n"
        "classification: internal\n"
        f"{extra}"
        "---\n\n"
        f"{body}\n"
    )


def test_parse_front_matter_separates_metadata_and_body() -> None:
    metadata, body = parse_front_matter(_document_text())

    assert metadata["document_id"] == "HR-003"
    assert metadata["version"] == "2026.1"
    assert "# Remote Work Policy" in body


def test_loader_creates_enterprise_document(tmp_path: Path) -> None:
    path = tmp_path / "remote_work_policy.md"
    path.write_text(_document_text(), encoding="utf-8")

    document = load_markdown_document(path)

    assert document.document_id == "HR-003"
    assert document.title == "Remote Work Policy"
    assert document.department == "HR"
    assert document.version == "2026.1"
    assert document.effective_date.isoformat() == "2026-02-01"
    assert document.classification == "internal"
    assert document.source_path == path
    assert document.content.startswith("# Remote Work Policy")


def test_loader_preserves_additional_metadata(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        _document_text(extra_metadata="language: en"),
        encoding="utf-8",
    )

    document = load_markdown_document(path)

    assert document.metadata["language"] == "en"


def test_loader_rejects_missing_front_matter(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\nContent", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="must begin"):
        load_markdown_document(path)


def test_loader_rejects_missing_required_metadata(
    tmp_path: Path,
) -> None:
    text = _document_text().replace(
        "owner: People Operations\n",
        "",
    )

    path = tmp_path / "policy.md"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="owner"):
        load_markdown_document(path)


def test_loader_rejects_invalid_effective_date(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        _document_text(effective_date="01-02-2026"),
        encoding="utf-8",
    )

    with pytest.raises(DocumentLoadError, match="YYYY-MM-DD"):
        load_markdown_document(path)


def test_loader_rejects_empty_body(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        _document_text(body="   "),
        encoding="utf-8",
    )

    with pytest.raises(DocumentLoadError, match="body must not be empty"):
        load_markdown_document(path)


def test_loader_rejects_non_markdown_file(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text(_document_text(), encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="unsupported"):
        load_markdown_document(path)


def test_directory_loader_is_deterministic(tmp_path: Path) -> None:
    second = tmp_path / "z_policy.md"
    first = tmp_path / "a_policy.md"

    second.write_text(
        _document_text(document_id="HR-002"),
        encoding="utf-8",
    )
    first.write_text(
        _document_text(document_id="HR-001"),
        encoding="utf-8",
    )

    documents = load_document_directory(tmp_path)

    assert [document.document_id for document in documents] == [
        "HR-001",
        "HR-002",
    ]


def test_directory_loader_rejects_duplicate_document_ids(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    first.write_text(
        _document_text(document_id="HR-003"),
        encoding="utf-8",
    )
    second.write_text(
        _document_text(document_id="HR-003"),
        encoding="utf-8",
    )

    with pytest.raises(DocumentLoadError, match="duplicate document_id"):
        load_document_directory(tmp_path)


def test_real_corpus_loads_all_documents() -> None:
    documents = load_document_directory(Path("data/raw"))

    assert len(documents) == 15
    assert len({document.document_id for document in documents}) == 15


def test_real_corpus_contains_expected_remote_work_policy() -> None:
    documents = load_document_directory(Path("data/raw"))

    policies = {
        document.document_id: document
        for document in documents
    }

    remote_policy = policies["HR-003"]

    assert remote_policy.title == "Remote Work Policy"
    assert remote_policy.department == "HR"
    assert "three days per workweek" in remote_policy.content
