"""Hardening tests for the EnterpriseIQ ingestion boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from enterpriseiq.ingestion import (
    ChunkingConfig,
    DocumentLoadError,
    calculate_sha256,
    load_document_directory,
    load_markdown_document,
    parse_front_matter,
    run_ingestion,
)


def _valid_document(
    *,
    document_id: str = "QA-001",
    title: str = "Quality Assurance Policy",
    body: str = "# Policy\n\nEnterprise controls must be followed.",
) -> str:
    return (
        "---\n"
        f"document_id: {document_id}\n"
        f"title: {title}\n"
        "department: Security\n"
        "document_type: policy\n"
        'version: "2026.1"\n'
        'effective_date: "2026-01-01"\n'
        "owner: Security Operations\n"
        "classification: internal\n"
        "---\n\n"
        f"{body}\n"
    )


def test_front_matter_requires_closing_delimiter() -> None:
    raw = (
        "---\n"
        "document_id: QA-001\n"
        "title: Broken Policy\n"
    )

    with pytest.raises(DocumentLoadError, match="closing"):
        parse_front_matter(raw)


def test_front_matter_rejects_entry_without_colon() -> None:
    raw = (
        "---\n"
        "document_id: QA-001\n"
        "invalid metadata entry\n"
        "---\n"
        "Body"
    )

    with pytest.raises(DocumentLoadError, match="invalid front-matter"):
        parse_front_matter(raw)


def test_front_matter_rejects_duplicate_key() -> None:
    raw = (
        "---\n"
        "document_id: QA-001\n"
        "document_id: QA-002\n"
        "---\n"
        "Body"
    )

    with pytest.raises(DocumentLoadError, match="duplicate metadata"):
        parse_front_matter(raw)


def test_front_matter_rejects_empty_value() -> None:
    raw = (
        "---\n"
        "document_id:\n"
        "---\n"
        "Body"
    )

    with pytest.raises(DocumentLoadError, match="must not be empty"):
        parse_front_matter(raw)


def test_loader_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_markdown_document(missing)


def test_directory_loader_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_document_directory(missing)


def test_directory_loader_rejects_file_as_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "policy.md"
    source.write_text(_valid_document(), encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="must be a directory"):
        load_document_directory(source)


def test_loader_rejects_invalid_utf8(tmp_path: Path) -> None:
    source = tmp_path / "corrupted.md"
    source.write_bytes(b"\xff\xfe\xfa\xfb")

    with pytest.raises(DocumentLoadError, match="valid UTF-8"):
        load_markdown_document(source)


def test_pipeline_rejects_empty_corpus(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.mkdir()

    with pytest.raises(ValueError, match="no Markdown documents"):
        run_ingestion(
            source_directory=source,
            output_file=tmp_path / "chunks.jsonl",
            manifest_file=tmp_path / "manifest.json",
        )


def test_pipeline_does_not_create_artifacts_for_empty_corpus(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty"
    source.mkdir()

    output = tmp_path / "chunks.jsonl"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="no Markdown documents"):
        run_ingestion(
            source_directory=source,
            output_file=output,
            manifest_file=manifest,
        )

    assert not output.exists()
    assert not manifest.exists()


def test_duplicate_ids_fail_before_artifacts_are_written(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw"
    source.mkdir()

    (source / "first.md").write_text(
        _valid_document(document_id="QA-001"),
        encoding="utf-8",
    )
    (source / "second.md").write_text(
        _valid_document(document_id="QA-001"),
        encoding="utf-8",
    )

    output = tmp_path / "chunks.jsonl"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(DocumentLoadError, match="duplicate document_id"):
        run_ingestion(
            source_directory=source,
            output_file=output,
            manifest_file=manifest,
        )

    assert not output.exists()
    assert not manifest.exists()


@pytest.mark.parametrize(
    "config",
    [
        {"chunk_size": 0},
        {"chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {
            "chunk_size": 100,
            "chunk_overlap": 10,
            "min_chunk_size": 101,
        },
    ],
)
def test_invalid_chunk_configs_fail_fast(
    config: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(**config)


def test_real_corpus_has_expected_document_identity() -> None:
    documents = load_document_directory("data/raw")

    document_ids = [document.document_id for document in documents]

    assert len(documents) == 15
    assert len(document_ids) == len(set(document_ids))


def test_real_corpus_required_fields_are_populated() -> None:
    documents = load_document_directory("data/raw")

    for document in documents:
        assert document.document_id
        assert document.title
        assert document.department
        assert document.document_type
        assert document.version
        assert document.owner
        assert document.classification
        assert document.content
        assert document.source_path.suffix == ".md"


def test_real_corpus_classifications_are_supported() -> None:
    documents = load_document_directory("data/raw")

    supported = {
        "public",
        "internal",
        "confidential",
        "restricted",
    }

    assert {
        document.classification
        for document in documents
    } <= supported


def test_pipeline_jsonl_records_have_required_schema(
    tmp_path: Path,
) -> None:
    output = tmp_path / "chunks.jsonl"
    manifest = tmp_path / "manifest.json"

    run_ingestion(
        source_directory="data/raw",
        output_file=output,
        manifest_file=manifest,
        config=ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    required_fields = {
        "chunk_id",
        "document_id",
        "chunk_index",
        "text",
        "start_char",
        "end_char",
        "title",
        "department",
        "document_type",
        "version",
        "effective_date",
        "owner",
        "classification",
        "source_path",
        "metadata",
    }

    for line in output.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)

        assert set(record) == required_fields


def test_pipeline_chunk_indexes_are_contiguous_per_document(
    tmp_path: Path,
) -> None:
    output = tmp_path / "chunks.jsonl"

    run_ingestion(
        source_directory="data/raw",
        output_file=output,
        manifest_file=tmp_path / "manifest.json",
        config=ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    grouped: dict[str, list[int]] = {}

    for record in records:
        grouped.setdefault(
            record["document_id"],
            [],
        ).append(record["chunk_index"])

    for indexes in grouped.values():
        assert indexes == list(range(len(indexes)))


def test_pipeline_chunk_ids_match_document_and_index(
    tmp_path: Path,
) -> None:
    output = tmp_path / "chunks.jsonl"

    run_ingestion(
        source_directory="data/raw",
        output_file=output,
        manifest_file=tmp_path / "manifest.json",
        config=ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    for line in output.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)

        expected = (
            f"{record['document_id']}::"
            f"{record['chunk_index']:04d}"
        )

        assert record["chunk_id"] == expected


def test_generated_artifact_hash_is_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    config = ChunkingConfig(
        chunk_size=700,
        chunk_overlap=100,
        min_chunk_size=200,
    )

    run_ingestion(
        source_directory="data/raw",
        output_file=first,
        manifest_file=tmp_path / "first.json",
        config=config,
    )
    run_ingestion(
        source_directory="data/raw",
        output_file=second,
        manifest_file=tmp_path / "second.json",
        config=config,
    )

    assert calculate_sha256(first) == calculate_sha256(second)


def test_cli_successfully_runs_real_corpus(tmp_path: Path) -> None:
    output = tmp_path / "chunks.jsonl"
    manifest = tmp_path / "manifest.json"

    command = [
        sys.executable,
        "-m",
        "enterpriseiq.ingestion.cli",
        "--source",
        "data/raw",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
        "--chunk-size",
        "700",
        "--chunk-overlap",
        "100",
        "--min-chunk-size",
        "200",
        "--log-level",
        "ERROR",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert manifest.exists()

    summary = json.loads(completed.stdout)

    assert summary["document_count"] == 15
    assert summary["chunk_count"] >= 15


def test_cli_invalid_configuration_returns_nonzero(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "enterpriseiq.ingestion.cli",
        "--source",
        "data/raw",
        "--output",
        str(tmp_path / "chunks.jsonl"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--chunk-size",
        "100",
        "--chunk-overlap",
        "100",
        "--log-level",
        "ERROR",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not (tmp_path / "chunks.jsonl").exists()
