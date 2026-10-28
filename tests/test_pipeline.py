"""Tests for the end-to-end EnterpriseIQ ingestion pipeline."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from enterpriseiq.domain import EnterpriseDocument
from enterpriseiq.ingestion import (
    ChunkingConfig,
    calculate_sha256,
    chunk_document,
    chunk_to_record,
    run_ingestion,
)


def _document() -> EnterpriseDocument:
    return EnterpriseDocument(
        document_id="TEST-001",
        title="Test Policy",
        department="Security",
        document_type="policy",
        version="2026.1",
        effective_date=date(2026, 1, 1),
        owner="Security",
        classification="internal",
        content="Enterprise policy content for testing.",
        source_path=Path("data/raw/security/test_policy.md"),
        metadata={"language": "en"},
    )


def test_chunk_to_record_is_json_serializable() -> None:
    document = _document()

    chunk = chunk_document(
        document,
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=20,
        ),
    )[0]

    record = chunk_to_record(chunk)
    encoded = json.dumps(record)

    assert encoded
    assert record["document_id"] == "TEST-001"
    assert record["effective_date"] == "2026-01-01"
    assert record["source_path"] == (
        "data/raw/security/test_policy.md"
    )


def test_end_to_end_pipeline_writes_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "chunks.jsonl"
    manifest = tmp_path / "manifest.json"

    result = run_ingestion(
        source_directory=Path("data/raw"),
        output_file=output,
        manifest_file=manifest,
        config=ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    assert result.document_count == 15
    assert result.chunk_count >= 15

    assert output.exists()
    assert manifest.exists()

    assert result.output_sha256 == calculate_sha256(output)


def test_jsonl_contains_one_valid_record_per_chunk(
    tmp_path: Path,
) -> None:
    output = tmp_path / "chunks.jsonl"
    manifest = tmp_path / "manifest.json"

    result = run_ingestion(
        source_directory="data/raw",
        output_file=output,
        manifest_file=manifest,
        config=ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    lines = output.read_text(encoding="utf-8").splitlines()

    assert len(lines) == result.chunk_count

    records = [json.loads(line) for line in lines]

    assert len({record["chunk_id"] for record in records}) == len(records)
    assert len({record["document_id"] for record in records}) == 15


def test_manifest_matches_generated_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.json"

    result = run_ingestion(
        source_directory="data/raw",
        output_file=output,
        manifest_file=manifest_path,
        config=ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["document_count"] == 15
    assert manifest["chunk_count"] == result.chunk_count
    assert len(manifest["document_ids"]) == 15
    assert manifest["output_sha256"] == result.output_sha256

    assert manifest["chunking"] == {
        "strategy": "boundary_aware_character",
        "chunk_size": 700,
        "chunk_overlap": 100,
        "min_chunk_size": 200,
    }


def test_pipeline_is_deterministic(tmp_path: Path) -> None:
    first_output = tmp_path / "first.jsonl"
    first_manifest = tmp_path / "first_manifest.json"

    second_output = tmp_path / "second.jsonl"
    second_manifest = tmp_path / "second_manifest.json"

    first = run_ingestion(
        source_directory="data/raw",
        output_file=first_output,
        manifest_file=first_manifest,
        config=ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    second = run_ingestion(
        source_directory="data/raw",
        output_file=second_output,
        manifest_file=second_manifest,
        config=ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    assert first_output.read_bytes() == second_output.read_bytes()
    assert first.output_sha256 == second.output_sha256


def test_real_pipeline_preserves_all_source_documents(
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

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    source_ids = {record["document_id"] for record in records}

    assert len(source_ids) == 15

    for record in records:
        assert record["chunk_id"]
        assert record["text"]
        assert record["title"]
        assert record["department"]
        assert record["source_path"]
        assert record["classification"]


def test_manifest_has_expected_departments(
    tmp_path: Path,
) -> None:
    output = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.json"

    run_ingestion(
        source_directory="data/raw",
        output_file=output,
        manifest_file=manifest_path,
        config=ChunkingConfig(
            chunk_size=700,
            chunk_overlap=100,
            min_chunk_size=200,
        ),
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["departments"] == [
        "Finance",
        "HR",
        "IT",
        "Operations",
        "Product",
        "Security",
    ]
