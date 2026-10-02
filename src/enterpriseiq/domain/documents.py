"""Domain contracts for enterprise knowledge documents and chunks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

_ALLOWED_CLASSIFICATIONS = frozenset(
    {
        "public",
        "internal",
        "confidential",
        "restricted",
    }
)


def _require_text(value: str, field_name: str) -> str:
    """Return normalized non-empty text or raise a domain validation error."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")

    return normalized


def _normalize_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Create an immutable defensive copy of additional metadata."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    return MappingProxyType(dict(metadata))


@dataclass(frozen=True, slots=True)
class EnterpriseDocument:
    """Canonical representation of one ingested enterprise document.

    The contract deliberately separates document content from business metadata.
    Downstream ingestion, retrieval, filtering, citation, and audit components
    can therefore rely on a stable representation independent of source format.
    """

    document_id: str
    title: str
    department: str
    document_type: str
    version: str
    effective_date: date
    owner: str
    classification: str
    content: str
    source_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text_fields = (
            "document_id",
            "title",
            "department",
            "document_type",
            "version",
            "owner",
            "classification",
            "content",
        )

        for field_name in text_fields:
            normalized = _require_text(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)

        classification = self.classification.lower()

        if classification not in _ALLOWED_CLASSIFICATIONS:
            allowed = ", ".join(sorted(_ALLOWED_CLASSIFICATIONS))
            raise ValueError(
                f"classification must be one of: {allowed}"
            )

        object.__setattr__(self, "classification", classification)

        if not isinstance(self.effective_date, date):
            raise TypeError("effective_date must be a datetime.date")

        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be pathlib.Path")

        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    @property
    def citation_label(self) -> str:
        """Return a stable human-readable source label."""
        return f"{self.document_id} — {self.title} (v{self.version})"


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Retrieval-ready passage derived from an EnterpriseDocument."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    title: str
    department: str
    document_type: str
    version: str
    effective_date: date
    owner: str
    classification: str
    source_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text_fields = (
            "chunk_id",
            "document_id",
            "text",
            "title",
            "department",
            "document_type",
            "version",
            "owner",
            "classification",
        )

        for field_name in text_fields:
            normalized = _require_text(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)

        classification = self.classification.lower()

        if classification not in _ALLOWED_CLASSIFICATIONS:
            allowed = ", ".join(sorted(_ALLOWED_CLASSIFICATIONS))
            raise ValueError(
                f"classification must be one of: {allowed}"
            )

        object.__setattr__(self, "classification", classification)

        if not isinstance(self.effective_date, date):
            raise TypeError("effective_date must be a datetime.date")

        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be pathlib.Path")

        if not isinstance(self.chunk_index, int) or isinstance(self.chunk_index, bool):
            raise TypeError("chunk_index must be an integer")

        if self.chunk_index < 0:
            raise ValueError("chunk_index must be greater than or equal to 0")

        for field_name in ("start_char", "end_char"):
            value = getattr(self, field_name)

            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")

        if self.start_char < 0:
            raise ValueError("start_char must be greater than or equal to 0")

        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")

        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    @classmethod
    def from_document(
        cls,
        *,
        document: EnterpriseDocument,
        chunk_id: str,
        text: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentChunk:
        """Build a chunk while preserving its parent document provenance."""
        combined_metadata = dict(document.metadata)

        if metadata:
            combined_metadata.update(metadata)

        return cls(
            chunk_id=chunk_id,
            document_id=document.document_id,
            text=text,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            title=document.title,
            department=document.department,
            document_type=document.document_type,
            version=document.version,
            effective_date=document.effective_date,
            owner=document.owner,
            classification=document.classification,
            source_path=document.source_path,
            metadata=combined_metadata,
        )

    @property
    def citation_label(self) -> str:
        """Return a citation label that identifies source and passage."""
        return (
            f"{self.document_id} — {self.title} "
            f"(v{self.version}, chunk {self.chunk_index})"
        )
