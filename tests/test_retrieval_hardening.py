"""Hardening tests for the EnterpriseIQ Day-02 retrieval pipeline."""

import json
from pathlib import Path

import pytest

from enterpriseiq.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingPipelineError,
    embed_chunk_records,
)
from enterpriseiq.retrieval import (
    PersistentVectorIndex,
    RetrievalError,
    SemanticRetriever,
    VectorIndexError,
    VectorIndexRecord,
)


def make_config(
    *,
    dimension: int = 8,
    model_name: str = "hardening-test-model",
) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="deterministic",
        model_name=model_name,
        dimension=dimension,
        batch_size=2,
        normalize=True,
    )


def make_chunk(
    *,
    chunk_id: str = "HR-001::0000",
    document_id: str = "HR-001",
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
    }


def make_record(
    *,
    vector: tuple[float, ...] = (1.0, 0.0),
    chunk_id: str = "HR-001::0000",
) -> VectorIndexRecord:
    return VectorIndexRecord(
        chunk_id=chunk_id,
        document_id="HR-001",
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


class WrongCountProvider:
    dimension = 2
    model_name = "broken-provider"

    def embed_documents(
        self,
        texts: object,
    ) -> list[tuple[float, ...]]:
        return [(1.0, 0.0)]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0)


class WrongDimensionQueryProvider:
    dimension = 2
    model_name = "query-test-model"

    def embed_documents(
        self,
        texts: object,
    ) -> list[tuple[float, ...]]:
        return []

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0,)


def test_embedding_pipeline_rejects_provider_wrong_vector_count() -> None:
    records = [
        make_chunk(),
        make_chunk(
            chunk_id="FIN-001::0000",
            document_id="FIN-001",
        ),
    ]

    with pytest.raises(
        EmbeddingPipelineError,
        match="unexpected number of vectors",
    ):
        embed_chunk_records(
            records,
            WrongCountProvider(),
            batch_size=2,
        )


@pytest.mark.parametrize(
    "vector",
    [
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
    ],
)
def test_vector_record_rejects_nonfinite_values(
    vector: tuple[float, ...],
) -> None:
    with pytest.raises(
        VectorIndexError,
        match="finite numeric values",
    ):
        make_record(vector=vector)


def test_vector_index_load_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.jsonl"
    index_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        VectorIndexError,
        match="manifest does not exist",
    ):
        PersistentVectorIndex.load(
            index_path=index_path,
            manifest_path=tmp_path / "missing.json",
        )


def test_vector_index_load_rejects_invalid_manifest_json(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.jsonl"
    manifest_path = tmp_path / "manifest.json"

    index_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(
        VectorIndexError,
        match="manifest is invalid",
    ):
        PersistentVectorIndex.load(
            index_path=index_path,
            manifest_path=manifest_path,
        )


def test_vector_index_load_rejects_unsupported_schema(
    tmp_path: Path,
) -> None:
    index = PersistentVectorIndex(
        dimension=2,
        model_name="schema-test-model",
        records=[make_record()],
    )

    index_path = tmp_path / "index.jsonl"
    manifest_path = tmp_path / "manifest.json"

    index.save(
        index_path=index_path,
        manifest_path=manifest_path,
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["schema_version"] = 999

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        VectorIndexError,
        match="unsupported vector index schema version",
    ):
        PersistentVectorIndex.load(
            index_path=index_path,
            manifest_path=manifest_path,
        )


def test_vector_index_load_rejects_record_count_mismatch(
    tmp_path: Path,
) -> None:
    index = PersistentVectorIndex(
        dimension=2,
        model_name="count-test-model",
        records=[make_record()],
    )

    index_path = tmp_path / "index.jsonl"
    manifest_path = tmp_path / "manifest.json"

    index.save(
        index_path=index_path,
        manifest_path=manifest_path,
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["record_count"] = 2

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        VectorIndexError,
        match="record count does not match manifest",
    ):
        PersistentVectorIndex.load(
            index_path=index_path,
            manifest_path=manifest_path,
        )


def test_vector_index_load_rejects_duplicate_chunk_ids(
    tmp_path: Path,
) -> None:
    index = PersistentVectorIndex(
        dimension=2,
        model_name="duplicate-test-model",
        records=[make_record()],
    )

    index_path = tmp_path / "index.jsonl"
    manifest_path = tmp_path / "manifest.json"

    index.save(
        index_path=index_path,
        manifest_path=manifest_path,
    )

    line = index_path.read_text(encoding="utf-8")
    index_path.write_text(
        line + line,
        encoding="utf-8",
    )

    import hashlib

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["record_count"] = 2
    manifest["index_sha256"] = hashlib.sha256(
        index_path.read_bytes()
    ).hexdigest()

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        VectorIndexError,
        match="duplicate chunk_id",
    ):
        PersistentVectorIndex.load(
            index_path=index_path,
            manifest_path=manifest_path,
        )


def test_semantic_retriever_rejects_bad_query_vector_dimension() -> None:
    index = PersistentVectorIndex(
        dimension=2,
        model_name="query-test-model",
        records=[make_record()],
    )

    retriever = SemanticRetriever(
        index=index,
        provider=WrongDimensionQueryProvider(),
    )

    with pytest.raises(
        RetrievalError,
        match="query embedding dimension",
    ):
        retriever.search("remote work")


def test_index_records_property_cannot_mutate_internal_index() -> None:
    index = PersistentVectorIndex(
        dimension=2,
        model_name="immutable-test-model",
        records=[make_record()],
    )

    records = index.records

    assert isinstance(records, tuple)

    with pytest.raises(AttributeError):
        records.append(make_record())  # type: ignore[attr-defined]

    assert len(index) == 1


def test_persisted_index_retains_provenance(
    tmp_path: Path,
) -> None:
    original = make_record()

    index = PersistentVectorIndex(
        dimension=2,
        model_name="provenance-test-model",
        records=[original],
    )

    index_path = tmp_path / "index.jsonl"
    manifest_path = tmp_path / "manifest.json"

    index.save(
        index_path=index_path,
        manifest_path=manifest_path,
    )

    loaded = PersistentVectorIndex.load(
        index_path=index_path,
        manifest_path=manifest_path,
    )

    record = loaded.records[0]

    assert record.document_id == original.document_id
    assert record.title == original.title
    assert record.department == original.department
    assert record.source_path == original.source_path
    assert dict(record.metadata) == dict(original.metadata)


def test_deterministic_provider_produces_same_query_vector() -> None:
    config = make_config()
    provider = DeterministicEmbeddingProvider(config)

    first = provider.embed_query("enterprise retrieval")
    second = provider.embed_query("enterprise retrieval")

    assert first == second
