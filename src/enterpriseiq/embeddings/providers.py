"""Embedding provider implementations for EnterpriseIQ."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

from enterpriseiq.embeddings.config import EmbeddingConfig
from enterpriseiq.embeddings.contracts import Vector


def _validate_text(text: str, *, field_name: str = "text") -> str:
    if not isinstance(text, str):
        raise TypeError(f"{field_name} must be a string")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")

    return cleaned


def _normalize_vector(values: Sequence[float]) -> Vector:
    vector = tuple(float(value) for value in values)

    if not vector:
        raise ValueError("embedding vector must not be empty")

    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain only finite values")

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("embedding vector must have non-zero magnitude")

    return tuple(value / norm for value in vector)


def _validate_vector(
    values: Sequence[float],
    *,
    expected_dimension: int,
    normalize: bool,
) -> Vector:
    vector = tuple(float(value) for value in values)

    if len(vector) != expected_dimension:
        raise ValueError(
            "embedding dimension mismatch: "
            f"expected {expected_dimension}, got {len(vector)}"
        )

    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding vector must contain only finite values")

    if normalize:
        return _normalize_vector(vector)

    return vector


class DeterministicEmbeddingProvider:
    """Deterministic dependency-free provider for tests and development."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def _embed(self, text: str) -> Vector:
        cleaned = _validate_text(text)

        values: list[float] = []
        counter = 0

        while len(values) < self.dimension:
            payload = f"{cleaned}|{counter}".encode()
            digest = hashlib.sha256(payload).digest()

            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) == self.dimension:
                    break

            counter += 1

        return _validate_vector(
            values,
            expected_dimension=self.dimension,
            normalize=self._config.normalize,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")

        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)


class SentenceTransformerEmbeddingProvider:
    """Lazy-loaded sentence-transformers embedding provider."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the local embedding "
                "provider. Install the embedding dependencies first."
            ) from exc

        self._model = SentenceTransformer(self.model_name)

        model_dimension = self._model.get_sentence_embedding_dimension()
        if model_dimension != self.dimension:
            raise ValueError(
                "configured embedding dimension does not match model: "
                f"configured={self.dimension}, model={model_dimension}"
            )

        return self._model

    def _encode(self, texts: Sequence[str]) -> list[Vector]:
        cleaned = [
            _validate_text(text, field_name=f"texts[{index}]")
            for index, text in enumerate(texts)
        ]

        if not cleaned:
            return []

        model = self._load_model()
        embeddings = model.encode(
            cleaned,
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        return [
            _validate_vector(
                vector,
                expected_dimension=self.dimension,
                normalize=self._config.normalize,
            )
            for vector in embeddings
        ]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")

        return self._encode(texts)

    def embed_query(self, text: str) -> Vector:
        cleaned = _validate_text(text)
        vectors = self._encode([cleaned])
        return vectors[0]
