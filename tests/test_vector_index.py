"""Tests for EnterpriseIQ persistent vector index."""

import hashlib
import json
from pathlib import Path

import pytest

from enterpriseiq.retrieval import (
    PersistentVectorIndex,
    VectorIndexError,
    VectorIndexRecord,
)


def make_record(
    *,
    chunk_id: str = "HR-003::0000",
    document_id: str = "HR-003",
    vector: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> VectorIndexRecord:
    return VectorIndexRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text="Employees may work remotely.",
        title="Remote Work Policy",
        department="HR",
        document_type="policy",
        version="1.0",
        effective_date="2026-01-01",
        owner="People Operations",
        classification="internal",
        source_path="data/raw/hr/remote_work_policy.md",
        metadata={"classification": "internal"},
        vector=vector,
    )


def make_embedding_record(
    *,
    chunk_id: str = "HR-003::0000",
    document_id: str = "HR-003",
    vector: list[float] | None = None,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "chunk_index": 0,
        "text": "Employees may work remotely.",
        "title": "Remote Work Policy",
        "department": "HR",
        "document_type": "policy",
        "version": "1.0",
        "effective_date": "2026-01-01",
        "owner": "People Operations",
        "classification": "internal",
        "source_path": "data/raw/hr/remote_work_policy.md",
        "metadata": {"classification": "internal"},
        "embedding": vector or [0.1, 0.2, 0.3],
        "embedding_model": "test-model",
        "embedding_dimension": 3,
    }


def write_embedding_artifact(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_vector_index_record_metadata_is_immutable() -> None:
    record = make_record()

    with pytest.raises(TypeError):
        record.metadata["classification"] = "public"  # type: ignore[index]


def test_vector_index_rejects_wrong_dimension() -> None:
    index = PersistentVectorIndex(
        dimension=3,
        model_name="test-model",
    )

    with pytest.raises(
        VectorIndexError,
        match="dimension mismatch",
    ):
        index.add(
            make_record(vector=(0.1, 0.2))
        )


def test_vector_index_rejects_duplicate_chunk_id() -> None:
    index = PersistentVectorIndex(
        dimension=3,
        model_name="test-model",
    )
    record = make_record()

    index.add(record)

    with pytest.raises(
        VectorIndexError,
        match="duplicate chunk_id",
    ):
        index.add(record)


def test_index_from_embedding_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "embeddings.jsonl"

    write_embedding_artifact(
        artifact,
        [
            make_embedding_record(),
            make_embedding_record(
                chunk_id="FIN-001::0000",
                document_id="FIN-001",
                vector=[0.4, 0.5, 0.6],
            ),
        ],
    )

    index = PersistentVectorIndex.from_embedding_artifact(
        artifact
    )

    assert len(index) == 2
    assert index.dimension == 3
    assert index.model_name == "test-model"
    assert index.records[0].chunk_id == "HR-003::0000"
    assert index.records[1].chunk_id == "FIN-001::0000"


def test_embedding_artifact_rejects_inconsistent_model(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "embeddings.jsonl"

    first = make_embedding_record()
    second = make_embedding_record(
        chunk_id="FIN-001::0000",
        document_id="FIN-001",
    )
    second["embedding_model"] = "different-model"

    write_embedding_artifact(
        artifact,
        [first, second],
    )

    with pytest.raises(
        VectorIndexError,
        match="inconsistent embedding model",
    ):
        PersistentVectorIndex.from_embedding_artifact(
            artifact
        )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = PersistentVectorIndex(
        dimension=3,
        model_name="test-model",
        records=[
            make_record(),
            make_record(
                chunk_id="FIN-001::0000",
                document_id="FIN-001",
                vector=(0.4, 0.5, 0.6),
            ),
        ],
    )

    index_path = tmp_path / "vector_index.jsonl"
    manifest_path = tmp_path / "vector_index_manifest.json"

    manifest = index.save(
        index_path=index_path,
        manifest_path=manifest_path,
    )

    loaded = PersistentVectorIndex.load(
        index_path=index_path,
        manifest_path=manifest_path,
    )

    assert manifest.record_count == 2
    assert len(loaded) == 2
    assert loaded.dimension == 3
    assert loaded.model_name == "test-model"
    assert loaded.records == index.records


def test_manifest_hash_matches_index(tmp_path: Path) -> None:
    index = PersistentVectorIndex(
        dimension=3,
        model_name="test-model",
        records=[make_record()],
    )

    index_path = tmp_path / "vector_index.jsonl"
    manifest_path = tmp_path / "manifest.json"

    manifest = index.save(
        index_path=index_path,
        manifest_path=manifest_path,
    )

    expected = hashlib.sha256(
        index_path.read_bytes()
    ).hexdigest()

    assert manifest.index_sha256 == expected


def test_load_rejects_tampered_index(tmp_path: Path) -> None:
    index = PersistentVectorIndex(
        dimension=3,
        model_name="test-model",
        records=[make_record()],
    )

    index_path = tmp_path / "vector_index.jsonl"
    manifest_path = tmp_path / "manifest.json"

    index.save(
        index_path=index_path,
        manifest_path=manifest_path,
    )

    with index_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write("tampered\n")

    with pytest.raises(
        VectorIndexError,
        match="integrity check failed",
    ):
        PersistentVectorIndex.load(
            index_path=index_path,
            manifest_path=manifest_path,
        )


def test_save_is_deterministic(tmp_path: Path) -> None:
    index = PersistentVectorIndex(
        dimension=3,
        model_name="test-model",
        records=[make_record()],
    )

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_manifest = tmp_path / "first_manifest.json"
    second_manifest = tmp_path / "second_manifest.json"

    first_result = index.save(
        index_path=first,
        manifest_path=first_manifest,
    )
    second_result = index.save(
        index_path=second,
        manifest_path=second_manifest,
    )

    assert first.read_bytes() == second.read_bytes()
    assert (
        first_result.index_sha256
        == second_result.index_sha256
    )


def test_empty_index_cannot_be_persisted(
    tmp_path: Path,
) -> None:
    index = PersistentVectorIndex(
        dimension=3,
        model_name="test-model",
    )

    with pytest.raises(
        VectorIndexError,
        match="empty vector index",
    ):
        index.save(
            index_path=tmp_path / "index.jsonl",
            manifest_path=tmp_path / "manifest.json",
        )
