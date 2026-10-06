"""Batch embedding pipeline for EnterpriseIQ retrieval artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enterpriseiq.embeddings.config import EmbeddingConfig
from enterpriseiq.embeddings.contracts import EmbeddingProvider, Vector


class EmbeddingPipelineError(ValueError):
    """Raised when embedding input or generated artifacts are invalid."""


@dataclass(frozen=True, slots=True)
class EmbeddingPipelineResult:
    """Summary returned after a successful embedding pipeline run."""

    input_path: Path
    output_path: Path
    manifest_path: Path
    chunk_count: int
    dimension: int
    model_name: str
    output_sha256: str


_REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "document_id",
    "text",
    "chunk_index",
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


def _require_nonempty_string(
    record: Mapping[str, Any],
    field_name: str,
    *,
    line_number: int,
) -> str:
    value = record.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise EmbeddingPipelineError(
            f"line {line_number}: {field_name} must be a non-empty string"
        )

    return value


def _validate_chunk_record(
    record: Any,
    *,
    line_number: int,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise EmbeddingPipelineError(
            f"line {line_number}: chunk record must be a JSON object"
        )

    missing = sorted(_REQUIRED_CHUNK_FIELDS.difference(record))
    if missing:
        raise EmbeddingPipelineError(
            f"line {line_number}: missing required fields: {', '.join(missing)}"
        )

    for field_name in (
        "chunk_id",
        "document_id",
        "text",
        "title",
        "department",
        "document_type",
        "version",
        "effective_date",
        "owner",
        "classification",
        "source_path",
    ):
        _require_nonempty_string(
            record,
            field_name,
            line_number=line_number,
        )

    chunk_index = record["chunk_index"]
    if (
        isinstance(chunk_index, bool)
        or not isinstance(chunk_index, int)
        or chunk_index < 0
    ):
        raise EmbeddingPipelineError(
            f"line {line_number}: chunk_index must be a non-negative integer"
        )

    if not isinstance(record["metadata"], dict):
        raise EmbeddingPipelineError(
            f"line {line_number}: metadata must be a JSON object"
        )

    expected_prefix = f"{record['document_id']}::"
    if not record["chunk_id"].startswith(expected_prefix):
        raise EmbeddingPipelineError(
            f"line {line_number}: chunk_id does not match document_id"
        )

    return dict(record)


def load_chunk_records(path: Path) -> list[dict[str, Any]]:
    """Load and validate deterministic Day-01 chunk JSONL."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")

    if not path.exists():
        raise EmbeddingPipelineError(f"chunk artifact does not exist: {path}")

    if not path.is_file():
        raise EmbeddingPipelineError(f"chunk artifact is not a file: {path}")

    records: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EmbeddingPipelineError(
            f"chunk artifact must be valid UTF-8: {path}"
        ) from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EmbeddingPipelineError(
                f"line {line_number}: invalid JSON"
            ) from exc

        record = _validate_chunk_record(
            raw_record,
            line_number=line_number,
        )

        chunk_id = record["chunk_id"]
        if chunk_id in seen_chunk_ids:
            raise EmbeddingPipelineError(
                f"duplicate chunk_id detected: {chunk_id}"
            )

        seen_chunk_ids.add(chunk_id)
        records.append(record)

    if not records:
        raise EmbeddingPipelineError("chunk artifact contains no records")

    return records


def _validate_vector(
    vector: Vector,
    *,
    dimension: int,
    chunk_id: str,
) -> None:
    if len(vector) != dimension:
        raise EmbeddingPipelineError(
            f"{chunk_id}: embedding dimension mismatch: "
            f"expected {dimension}, got {len(vector)}"
        )

    if not all(math.isfinite(float(value)) for value in vector):
        raise EmbeddingPipelineError(
            f"{chunk_id}: embedding contains non-finite values"
        )


def _embedded_record(
    chunk: Mapping[str, Any],
    vector: Vector,
    *,
    model_name: str,
    dimension: int,
) -> dict[str, Any]:
    _validate_vector(
        vector,
        dimension=dimension,
        chunk_id=str(chunk["chunk_id"]),
    )

    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"],
        "title": chunk["title"],
        "department": chunk["department"],
        "document_type": chunk["document_type"],
        "version": chunk["version"],
        "effective_date": chunk["effective_date"],
        "owner": chunk["owner"],
        "classification": chunk["classification"],
        "source_path": chunk["source_path"],
        "metadata": chunk["metadata"],
        "embedding": [float(value) for value in vector],
        "embedding_model": model_name,
        "embedding_dimension": dimension,
    }


def _write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = "".join(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for record in records
    )

    path.write_text(payload, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def embed_chunk_records(
    records: list[dict[str, Any]],
    provider: EmbeddingProvider,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Embed validated chunk records in deterministic batches."""

    if not records:
        raise EmbeddingPipelineError("records must not be empty")

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    embedded_records: list[dict[str, Any]] = []

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [str(record["text"]) for record in batch]

        vectors = provider.embed_documents(texts)

        if len(vectors) != len(batch):
            raise EmbeddingPipelineError(
                "embedding provider returned an unexpected number of vectors"
            )

        for chunk, vector in zip(batch, vectors, strict=True):
            embedded_records.append(
                _embedded_record(
                    chunk,
                    vector,
                    model_name=provider.model_name,
                    dimension=provider.dimension,
                )
            )

    return embedded_records


def run_embedding_pipeline(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    provider: EmbeddingProvider,
    config: EmbeddingConfig,
) -> EmbeddingPipelineResult:
    """Run chunk loading, batch embedding, serialization and manifesting."""

    if provider.dimension != config.dimension:
        raise EmbeddingPipelineError(
            "provider/config dimension mismatch: "
            f"provider={provider.dimension}, config={config.dimension}"
        )

    if provider.model_name != config.model_name:
        raise EmbeddingPipelineError(
            "provider/config model mismatch: "
            f"provider={provider.model_name}, config={config.model_name}"
        )

    records = load_chunk_records(input_path)

    embedded = embed_chunk_records(
        records,
        provider,
        batch_size=config.batch_size,
    )

    _write_jsonl(output_path, embedded)
    output_hash = _sha256(output_path)

    manifest = {
        "schema_version": 1,
        "source_chunks": str(input_path),
        "embedding_output": str(output_path),
        "chunk_count": len(embedded),
        "provider": config.provider,
        "model_name": provider.model_name,
        "dimension": provider.dimension,
        "batch_size": config.batch_size,
        "normalize": config.normalize,
        "output_sha256": output_hash,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return EmbeddingPipelineResult(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        chunk_count=len(embedded),
        dimension=provider.dimension,
        model_name=provider.model_name,
        output_sha256=output_hash,
    )
