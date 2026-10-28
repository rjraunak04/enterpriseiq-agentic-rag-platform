"""Persistent vector index for EnterpriseIQ."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


class VectorIndexError(ValueError):
    """Raised when vector index data violates its integrity contract."""


@dataclass(frozen=True, slots=True)
class VectorIndexRecord:
    """One persistent vector entry with enterprise provenance."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    title: str
    department: str
    document_type: str
    version: str
    effective_date: str
    owner: str
    classification: str
    source_path: str
    metadata: Mapping[str, Any]
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
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
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise VectorIndexError(f"{field_name} must not be empty")

        if (
            isinstance(self.chunk_index, bool)
            or not isinstance(self.chunk_index, int)
            or self.chunk_index < 0
        ):
            raise VectorIndexError(
                "chunk_index must be a non-negative integer"
            )

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        if not self.vector:
            raise VectorIndexError("vector must not be empty")

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in self.vector
        ):
            raise VectorIndexError(
                "vector must contain only finite numeric values"
            )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )
        object.__setattr__(
            self,
            "vector",
            tuple(float(value) for value in self.vector),
        )


@dataclass(frozen=True, slots=True)
class VectorIndexManifest:
    """Metadata describing a persisted EnterpriseIQ vector index."""

    schema_version: int
    record_count: int
    dimension: int
    model_name: str
    index_sha256: str


class PersistentVectorIndex:
    """Dependency-light persistent vector index.

    CP-04 owns persistence and integrity. Similarity search is implemented
    by the retrieval layer in the next checkpoint.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        dimension: int,
        model_name: str,
        records: list[VectorIndexRecord] | None = None,
    ) -> None:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise VectorIndexError(
                "dimension must be a positive integer"
            )

        if not isinstance(model_name, str) or not model_name.strip():
            raise VectorIndexError("model_name must not be empty")

        self._dimension = dimension
        self._model_name = model_name.strip()
        self._records: list[VectorIndexRecord] = []
        self._chunk_ids: set[str] = set()

        for record in records or []:
            self.add(record)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def records(self) -> tuple[VectorIndexRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: VectorIndexRecord) -> None:
        if not isinstance(record, VectorIndexRecord):
            raise TypeError("record must be a VectorIndexRecord")

        if len(record.vector) != self.dimension:
            raise VectorIndexError(
                f"{record.chunk_id}: vector dimension mismatch: "
                f"expected {self.dimension}, got {len(record.vector)}"
            )

        if record.chunk_id in self._chunk_ids:
            raise VectorIndexError(
                f"duplicate chunk_id detected: {record.chunk_id}"
            )

        self._records.append(record)
        self._chunk_ids.add(record.chunk_id)

    @classmethod
    def from_embedding_artifact(
        cls,
        path: Path,
    ) -> PersistentVectorIndex:
        """Construct an index from CP-03 embeddings JSONL."""

        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")

        if not path.exists():
            raise VectorIndexError(
                f"embedding artifact does not exist: {path}"
            )

        if not path.is_file():
            raise VectorIndexError(
                f"embedding artifact is not a file: {path}"
            )

        raw_records: list[dict[str, Any]] = []

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise VectorIndexError(
                f"embedding artifact must be valid UTF-8: {path}"
            ) from exc

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VectorIndexError(
                    f"line {line_number}: invalid JSON"
                ) from exc

            if not isinstance(record, dict):
                raise VectorIndexError(
                    f"line {line_number}: record must be a JSON object"
                )

            raw_records.append(record)

        if not raw_records:
            raise VectorIndexError(
                "embedding artifact contains no records"
            )

        first = raw_records[0]

        dimension = first.get("embedding_dimension")
        model_name = first.get("embedding_model")

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise VectorIndexError(
                "embedding_dimension must be a positive integer"
            )

        if not isinstance(model_name, str) or not model_name.strip():
            raise VectorIndexError(
                "embedding_model must be a non-empty string"
            )

        index = cls(
            dimension=dimension,
            model_name=model_name,
        )

        required = {
            "chunk_id",
            "document_id",
            "chunk_index",
            "text",
            "title",
            "department",
            "document_type",
            "version",
            "effective_date",
            "owner",
            "classification",
            "source_path",
            "metadata",
            "embedding",
            "embedding_model",
            "embedding_dimension",
        }

        for line_number, raw in enumerate(raw_records, start=1):
            missing = sorted(required.difference(raw))
            if missing:
                raise VectorIndexError(
                    f"line {line_number}: missing required fields: "
                    f"{', '.join(missing)}"
                )

            if raw["embedding_dimension"] != dimension:
                raise VectorIndexError(
                    f"line {line_number}: inconsistent embedding dimension"
                )

            if raw["embedding_model"] != model_name:
                raise VectorIndexError(
                    f"line {line_number}: inconsistent embedding model"
                )

            embedding = raw["embedding"]
            if not isinstance(embedding, list):
                raise VectorIndexError(
                    f"line {line_number}: embedding must be a JSON array"
                )

            metadata = raw["metadata"]
            if not isinstance(metadata, dict):
                raise VectorIndexError(
                    f"line {line_number}: metadata must be a JSON object"
                )

            index.add(
                VectorIndexRecord(
                    chunk_id=raw["chunk_id"],
                    document_id=raw["document_id"],
                    chunk_index=raw["chunk_index"],
                    text=raw["text"],
                    title=raw["title"],
                    department=raw["department"],
                    document_type=raw["document_type"],
                    version=raw["version"],
                    effective_date=raw["effective_date"],
                    owner=raw["owner"],
                    classification=raw["classification"],
                    source_path=raw["source_path"],
                    metadata=metadata,
                    vector=tuple(embedding),
                )
            )

        return index

    def _record_to_dict(
        self,
        record: VectorIndexRecord,
    ) -> dict[str, Any]:
        return {
            "chunk_id": record.chunk_id,
            "document_id": record.document_id,
            "chunk_index": record.chunk_index,
            "text": record.text,
            "title": record.title,
            "department": record.department,
            "document_type": record.document_type,
            "version": record.version,
            "effective_date": record.effective_date,
            "owner": record.owner,
            "classification": record.classification,
            "source_path": record.source_path,
            "metadata": dict(record.metadata),
            "vector": list(record.vector),
        }

    def save(
        self,
        *,
        index_path: Path,
        manifest_path: Path,
    ) -> VectorIndexManifest:
        """Persist deterministic index records and integrity manifest."""

        if not self._records:
            raise VectorIndexError("cannot persist an empty vector index")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        payload = "".join(
            json.dumps(
                self._record_to_dict(record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for record in self._records
        )

        index_path.write_text(payload, encoding="utf-8")

        digest = hashlib.sha256(index_path.read_bytes()).hexdigest()

        manifest_data = {
            "schema_version": self.SCHEMA_VERSION,
            "record_count": len(self),
            "dimension": self.dimension,
            "model_name": self.model_name,
            "index_sha256": digest,
        }

        manifest_path.write_text(
            json.dumps(
                manifest_data,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        return VectorIndexManifest(**manifest_data)

    @classmethod
    def load(
        cls,
        *,
        index_path: Path,
        manifest_path: Path,
    ) -> PersistentVectorIndex:
        """Load a persisted index after verifying its manifest."""

        if not index_path.exists():
            raise VectorIndexError(
                f"vector index does not exist: {index_path}"
            )

        if not manifest_path.exists():
            raise VectorIndexError(
                f"vector index manifest does not exist: {manifest_path}"
            )

        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VectorIndexError(
                "vector index manifest is invalid"
            ) from exc

        required_manifest = {
            "schema_version",
            "record_count",
            "dimension",
            "model_name",
            "index_sha256",
        }

        if (
            not isinstance(manifest, dict)
            or not required_manifest.issubset(manifest)
        ):
            raise VectorIndexError(
                "vector index manifest is missing required fields"
            )

        if manifest["schema_version"] != cls.SCHEMA_VERSION:
            raise VectorIndexError(
                "unsupported vector index schema version"
            )

        actual_hash = hashlib.sha256(
            index_path.read_bytes()
        ).hexdigest()

        if actual_hash != manifest["index_sha256"]:
            raise VectorIndexError(
                "vector index SHA-256 integrity check failed"
            )

        records: list[VectorIndexRecord] = []

        try:
            lines = index_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except UnicodeDecodeError as exc:
            raise VectorIndexError(
                "vector index must be valid UTF-8"
            ) from exc

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VectorIndexError(
                    f"line {line_number}: invalid vector index JSON"
                ) from exc

            if not isinstance(raw, dict):
                raise VectorIndexError(
                    f"line {line_number}: index record must be an object"
                )

            try:
                records.append(
                    VectorIndexRecord(
                        chunk_id=raw["chunk_id"],
                        document_id=raw["document_id"],
                        chunk_index=raw["chunk_index"],
                        text=raw["text"],
                        title=raw["title"],
                        department=raw["department"],
                        document_type=raw["document_type"],
                        version=raw["version"],
                        effective_date=raw["effective_date"],
                        owner=raw["owner"],
                        classification=raw["classification"],
                        source_path=raw["source_path"],
                        metadata=raw["metadata"],
                        vector=tuple(raw["vector"]),
                    )
                )
            except KeyError as exc:
                raise VectorIndexError(
                    f"line {line_number}: vector index field missing"
                ) from exc

        if len(records) != manifest["record_count"]:
            raise VectorIndexError(
                "vector index record count does not match manifest"
            )

        index = cls(
            dimension=manifest["dimension"],
            model_name=manifest["model_name"],
            records=records,
        )

        return index
