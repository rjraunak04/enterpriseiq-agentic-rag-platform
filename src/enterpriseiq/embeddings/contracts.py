"""Core embedding contracts used by EnterpriseIQ."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

Vector = tuple[float, ...]


def _validate_vector(vector: Vector, *, dimension: int) -> None:
    if len(vector) != dimension:
        raise ValueError(
            f"vector dimension mismatch: expected {dimension}, got {len(vector)}"
        )

    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("vector values must be numeric")


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A document chunk paired with its embedding vector."""

    chunk_id: str
    document_id: str
    text: str
    vector: Vector
    dimension: int
    model_name: str

    def __post_init__(self) -> None:
        for field_name in ("chunk_id", "document_id", "text", "model_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")

        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("dimension must be an integer")
        if self.dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        _validate_vector(self.vector, dimension=self.dimension)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider-neutral contract for text embedding backends."""

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""

    @property
    def model_name(self) -> str:
        """Return provider model identifier."""

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed document texts."""

    def embed_query(self, text: str) -> Vector:
        """Embed a search query."""
