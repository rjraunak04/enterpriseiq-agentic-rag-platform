"""Retrieval-domain contracts for EnterpriseIQ."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A ranked retrieval result with source provenance."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    title: str
    department: str
    source_path: Path
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "chunk_id",
            "document_id",
            "text",
            "title",
            "department",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")

        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric")

        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer")
        if self.rank <= 0:
            raise ValueError("rank must be greater than zero")

        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a pathlib.Path")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def citation_label(self) -> str:
        """Human-readable citation label for downstream grounded RAG."""

        return f"{self.title} ({self.document_id})"
