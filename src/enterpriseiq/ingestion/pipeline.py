"""End-to-end ingestion pipeline for EnterpriseIQ."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from enterpriseiq.domain import DocumentChunk
from enterpriseiq.ingestion.chunker import (
    ChunkingConfig,
    chunk_documents,
)
from enterpriseiq.ingestion.loader import load_document_directory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary returned after a successful ingestion run."""

    source_directory: Path
    output_file: Path
    manifest_file: Path
    document_count: int
    chunk_count: int
    output_sha256: str


def _json_value(value: Any) -> Any:
    """Convert supported domain values into deterministic JSON values."""
    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]

    return value


def chunk_to_record(chunk: DocumentChunk) -> dict[str, Any]:
    """Serialize one chunk into the persistent ingestion schema."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "title": chunk.title,
        "department": chunk.department,
        "document_type": chunk.document_type,
        "version": chunk.version,
        "effective_date": chunk.effective_date.isoformat(),
        "owner": chunk.owner,
        "classification": chunk.classification,
        "source_path": chunk.source_path.as_posix(),
        "metadata": _json_value(dict(chunk.metadata)),
    }


def write_chunks_jsonl(
    chunks: list[DocumentChunk],
    output_path: str | Path,
) -> Path:
    """Write chunks as deterministic UTF-8 JSON Lines."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            record = chunk_to_record(chunk)

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")

    return destination


def calculate_sha256(path: str | Path) -> str:
    """Return SHA-256 for a generated artifact."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def _build_manifest(
    *,
    source_directory: Path,
    output_file: Path,
    document_count: int,
    chunks: list[DocumentChunk],
    config: ChunkingConfig,
    output_sha256: str,
) -> dict[str, Any]:
    """Create a deterministic ingestion manifest."""
    departments = sorted({chunk.department for chunk in chunks})
    document_ids = sorted({chunk.document_id for chunk in chunks})

    return {
        "schema_version": "1.0",
        "source_directory": source_directory.as_posix(),
        "output_file": output_file.as_posix(),
        "document_count": document_count,
        "chunk_count": len(chunks),
        "document_ids": document_ids,
        "departments": departments,
        "chunking": {
            "strategy": "boundary_aware_character",
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "min_chunk_size": config.min_chunk_size,
        },
        "output_sha256": output_sha256,
    }


def write_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> Path:
    """Persist an ingestion manifest as deterministic JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return destination


def run_ingestion(
    *,
    source_directory: str | Path = "data/raw",
    output_file: str | Path = "data/processed/chunks.jsonl",
    manifest_file: str | Path = "data/processed/ingestion_manifest.json",
    config: ChunkingConfig | None = None,
) -> IngestionResult:
    """Execute the complete EnterpriseIQ document ingestion pipeline."""
    source_path = Path(source_directory)
    output_path = Path(output_file)
    manifest_path = Path(manifest_file)
    resolved_config = config or ChunkingConfig()

    logger.info(
        "Starting ingestion source=%s output=%s",
        source_path,
        output_path,
    )

    documents = load_document_directory(source_path)

    logger.info(
        "Loaded %d enterprise documents",
        len(documents),
    )

    if not documents:
        raise ValueError(
            f"no Markdown documents found in source directory: {source_path}"
        )

    chunks = chunk_documents(documents, resolved_config)

    if not chunks:
        raise ValueError("ingestion produced no chunks")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("ingestion produced duplicate chunk IDs")

    source_document_ids = {document.document_id for document in documents}
    chunk_document_ids = {chunk.document_id for chunk in chunks}

    if source_document_ids != chunk_document_ids:
        missing = sorted(source_document_ids - chunk_document_ids)

        raise ValueError(
            "one or more source documents produced no chunks: "
            + ", ".join(missing)
        )

    write_chunks_jsonl(chunks, output_path)
    output_sha256 = calculate_sha256(output_path)

    manifest = _build_manifest(
        source_directory=source_path,
        output_file=output_path,
        document_count=len(documents),
        chunks=chunks,
        config=resolved_config,
        output_sha256=output_sha256,
    )

    write_manifest(manifest, manifest_path)

    result = IngestionResult(
        source_directory=source_path,
        output_file=output_path,
        manifest_file=manifest_path,
        document_count=len(documents),
        chunk_count=len(chunks),
        output_sha256=output_sha256,
    )

    logger.info(
        "Ingestion complete documents=%d chunks=%d sha256=%s",
        result.document_count,
        result.chunk_count,
        result.output_sha256,
    )

    return result


def result_to_dict(result: IngestionResult) -> dict[str, Any]:
    """Return a JSON-friendly representation of an ingestion result."""
    return _json_value(asdict(result))

