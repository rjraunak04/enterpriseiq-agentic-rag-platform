"""Tests for the deterministic EnterpriseIQ embedding pipeline."""

import hashlib
import json
from pathlib import Path

import pytest

from enterpriseiq.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingPipelineError,
    embed_chunk_records,
    load_chunk_records,
    run_embedding_pipeline,
)


def make_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="deterministic",
        model_name="enterpriseiq-test-embedding",
        dimension=12,
        batch_size=2,
        normalize=True,
    )


def make_chunk(
    *,
    chunk_id: str = "HR-003::0000",
    document_id: str = "HR-003",
    chunk_index: int = 0,
    text: str = "Employees may work remotely.",
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "chunk_index": chunk_index,
        "start_char": 0,
        "end_char": len(text),
        "title": "Remote Work Policy",
        "department": "HR",
        "document_type": "policy",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "owner": "People Operations",
        "classification": "internal",
        "source_path": "data/raw/hr/remote_work_policy.md",
        "metadata": {"classification": "internal"},
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_load_chunk_records_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    records = [
        make_chunk(),
        make_chunk(
            chunk_id="FIN-001::0000",
            document_id="FIN-001",
            text="Receipts are required.",
        ),
    ]
    write_jsonl(path, records)

    loaded = load_chunk_records(path)

    assert [record["chunk_id"] for record in loaded] == [
        "HR-003::0000",
        "FIN-001::0000",
    ]


def test_load_chunk_records_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(
        EmbeddingPipelineError,
        match="does not exist",
    ):
        load_chunk_records(tmp_path / "missing.jsonl")


def test_load_chunk_records_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text("{bad-json}\n", encoding="utf-8")

    with pytest.raises(EmbeddingPipelineError, match="invalid JSON"):
        load_chunk_records(path)


def test_load_chunk_records_rejects_empty_artifact(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(
        EmbeddingPipelineError,
        match="contains no records",
    ):
        load_chunk_records(path)


def test_load_chunk_records_rejects_duplicate_chunk_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunks.jsonl"
    record = make_chunk()
    write_jsonl(path, [record, record])

    with pytest.raises(
        EmbeddingPipelineError,
        match="duplicate chunk_id",
    ):
        load_chunk_records(path)


def test_load_chunk_records_rejects_missing_required_field(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunks.jsonl"
    record = make_chunk()
    del record["department"]
    write_jsonl(path, [record])

    with pytest.raises(
        EmbeddingPipelineError,
        match="missing required fields",
    ):
        load_chunk_records(path)


def test_embed_chunk_records_preserves_provenance() -> None:
    config = make_config()
    provider = DeterministicEmbeddingProvider(config)

    embedded = embed_chunk_records(
        [make_chunk()],
        provider,
        batch_size=1,
    )

    record = embedded[0]

    assert record["chunk_id"] == "HR-003::0000"
    assert record["document_id"] == "HR-003"
    assert record["department"] == "HR"
    assert record["classification"] == "internal"
    assert record["embedding_model"] == config.model_name
    assert record["embedding_dimension"] == 12
    assert len(record["embedding"]) == 12


def test_embedding_pipeline_writes_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "embeddings.jsonl"
    manifest_path = tmp_path / "embedding_manifest.json"

    write_jsonl(
        input_path,
        [
            make_chunk(),
            make_chunk(
                chunk_id="FIN-001::0000",
                document_id="FIN-001",
                text="Receipts are required.",
            ),
        ],
    )

    config = make_config()
    provider = DeterministicEmbeddingProvider(config)

    result = run_embedding_pipeline(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        provider=provider,
        config=config,
    )

    assert result.chunk_count == 2
    assert output_path.exists()
    assert manifest_path.exists()

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 2
    assert rows[0]["chunk_id"] == "HR-003::0000"
    assert rows[1]["chunk_id"] == "FIN-001::0000"


def test_embedding_manifest_matches_output_hash(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "embeddings.jsonl"
    manifest_path = tmp_path / "embedding_manifest.json"

    write_jsonl(input_path, [make_chunk()])

    config = make_config()
    provider = DeterministicEmbeddingProvider(config)

    run_embedding_pipeline(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        provider=provider,
        config=config,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()

    assert manifest["output_sha256"] == expected_hash


def test_embedding_pipeline_is_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    first_manifest = tmp_path / "first_manifest.json"
    second_manifest = tmp_path / "second_manifest.json"

    write_jsonl(input_path, [make_chunk()])

    config = make_config()
    provider = DeterministicEmbeddingProvider(config)

    first = run_embedding_pipeline(
        input_path=input_path,
        output_path=first_output,
        manifest_path=first_manifest,
        provider=provider,
        config=config,
    )

    second = run_embedding_pipeline(
        input_path=input_path,
        output_path=second_output,
        manifest_path=second_manifest,
        provider=provider,
        config=config,
    )

    assert first.output_sha256 == second.output_sha256
    assert first_output.read_bytes() == second_output.read_bytes()


def test_embedding_pipeline_rejects_provider_config_dimension_mismatch(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "chunks.jsonl"
    write_jsonl(input_path, [make_chunk()])

    provider_config = make_config()
    provider = DeterministicEmbeddingProvider(provider_config)

    incompatible_config = EmbeddingConfig(
        provider="deterministic",
        model_name=provider_config.model_name,
        dimension=24,
        batch_size=2,
        normalize=True,
    )

    with pytest.raises(
        EmbeddingPipelineError,
        match="provider/config dimension mismatch",
    ):
        run_embedding_pipeline(
            input_path=input_path,
            output_path=tmp_path / "output.jsonl",
            manifest_path=tmp_path / "manifest.json",
            provider=provider,
            config=incompatible_config,
        )
