"""Core contracts for grounded RAG responses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


class RAGContractError(ValueError):
    """Raised when a grounded RAG contract is invalid."""


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RAGContractError(
            f"{field_name} must be a non-empty string"
        )
    return value.strip()


def _immutable_metadata(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RAGContractError("metadata must be a mapping")
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class RAGQuery:
    """User query submitted to the grounded RAG pipeline."""

    text: str
    query_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _required_text(self.text, "text"),
        )

        if self.query_id is not None:
            object.__setattr__(
                self,
                "query_id",
                _required_text(self.query_id, "query_id"),
            )

        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One retrieved evidence unit available to generation."""

    source_id: str
    chunk_id: str
    document_id: str
    text: str
    title: str
    department: str
    source_path: Path
    retrieval_score: float
    retrieval_rank: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "chunk_id",
            "document_id",
            "text",
            "title",
            "department",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        if not isinstance(self.source_path, Path):
            raise RAGContractError(
                "source_path must be a pathlib.Path"
            )

        if (
            isinstance(self.retrieval_rank, bool)
            or not isinstance(self.retrieval_rank, int)
            or self.retrieval_rank <= 0
        ):
            raise RAGContractError(
                "retrieval_rank must be a positive integer"
            )

        if (
            isinstance(self.retrieval_score, bool)
            or not isinstance(self.retrieval_score, (int, float))
        ):
            raise RAGContractError(
                "retrieval_score must be numeric"
            )

        object.__setattr__(
            self,
            "retrieval_score",
            float(self.retrieval_score),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    """Bounded evidence supplied to a generator."""

    items: tuple[EvidenceItem, ...]
    rendered_text: str
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise RAGContractError("items must be a tuple")

        if not isinstance(self.rendered_text, str):
            raise RAGContractError(
                "rendered_text must be a string"
            )

        if not isinstance(self.truncated, bool):
            raise RAGContractError(
                "truncated must be a boolean"
            )

        source_ids = [item.source_id for item in self.items]
        if len(source_ids) != len(set(source_ids)):
            raise RAGContractError(
                "evidence source IDs must be unique"
            )


@dataclass(frozen=True, slots=True)
class Citation:
    """Validated mapping from an answer citation to evidence."""

    source_id: str
    chunk_id: str
    document_id: str
    title: str
    source_path: Path

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "chunk_id",
            "document_id",
            "title",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(self, field_name),
                    field_name,
                ),
            )

        if not isinstance(self.source_path, Path):
            raise RAGContractError(
                "source_path must be a pathlib.Path"
            )


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Raw answer returned by a generation provider."""

    text: str
    abstained: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.abstained, bool):
            raise RAGContractError(
                "abstained must be a boolean"
            )

        if not isinstance(self.text, str):
            raise RAGContractError("text must be a string")

        normalized = self.text.strip()

        if not self.abstained and not normalized:
            raise RAGContractError(
                "non-abstained answer text cannot be empty"
            )

        object.__setattr__(self, "text", normalized)
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class GroundedResponse:
    """Final validated response returned by grounded RAG."""

    query: RAGQuery
    answer: str
    citations: tuple[Citation, ...]
    grounded: bool
    abstained: bool
    evidence_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, RAGQuery):
            raise RAGContractError(
                "query must be a RAGQuery"
            )

        if not isinstance(self.answer, str):
            raise RAGContractError(
                "answer must be a string"
            )

        if not isinstance(self.citations, tuple):
            raise RAGContractError(
                "citations must be a tuple"
            )

        if not isinstance(self.grounded, bool):
            raise RAGContractError(
                "grounded must be a boolean"
            )

        if not isinstance(self.abstained, bool):
            raise RAGContractError(
                "abstained must be a boolean"
            )

        if (
            isinstance(self.evidence_count, bool)
            or not isinstance(self.evidence_count, int)
            or self.evidence_count < 0
        ):
            raise RAGContractError(
                "evidence_count must be a non-negative integer"
            )

        normalized = self.answer.strip()

        if not self.abstained and not normalized:
            raise RAGContractError(
                "non-abstained response cannot have an empty answer"
            )

        if self.abstained and self.grounded:
            raise RAGContractError(
                "an abstained response cannot be grounded"
            )

        if len(self.citations) > self.evidence_count:
            raise RAGContractError(
                "citation count cannot exceed evidence count"
            )

        object.__setattr__(self, "answer", normalized)
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata),
        )
