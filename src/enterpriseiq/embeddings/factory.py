"""Embedding provider factory."""

from __future__ import annotations

from enterpriseiq.embeddings.config import EmbeddingConfig
from enterpriseiq.embeddings.contracts import EmbeddingProvider
from enterpriseiq.embeddings.providers import (
    DeterministicEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Create an embedding provider from validated configuration."""

    if config.provider in {"deterministic", "test"}:
        return DeterministicEmbeddingProvider(config)

    if config.provider in {"local", "sentence-transformers"}:
        return SentenceTransformerEmbeddingProvider(config)

    raise ValueError(f"unsupported embedding provider: {config.provider}")
