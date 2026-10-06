"""Tests for EnterpriseIQ embedding configuration and contracts."""

from pathlib import Path

import pytest

from enterpriseiq.embeddings import (
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingProvider,
)
from enterpriseiq.retrieval import SearchResult


def test_embedding_config_defaults() -> None:
    config = EmbeddingConfig()

    assert config.provider == "local"
    assert config.dimension == 384
    assert config.batch_size == 32
    assert config.normalize is True


@pytest.mark.parametrize("dimension", [0, -1])
def test_embedding_config_rejects_invalid_dimension(dimension: int) -> None:
    with pytest.raises(ValueError):
        EmbeddingConfig(dimension=dimension)


@pytest.mark.parametrize("batch_size", [0, -5])
def test_embedding_config_rejects_invalid_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError):
        EmbeddingConfig(batch_size=batch_size)


def test_embedded_chunk_validates_dimension() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        EmbeddedChunk(
            chunk_id="HR-003::0000",
            document_id="HR-003",
            text="Remote work policy",
            vector=(0.1, 0.2),
            dimension=3,
            model_name="test-model",
        )


def test_embedded_chunk_accepts_valid_vector() -> None:
    embedded = EmbeddedChunk(
        chunk_id="HR-003::0000",
        document_id="HR-003",
        text="Remote work policy",
        vector=(0.1, 0.2, 0.3),
        dimension=3,
        model_name="test-model",
    )

    assert embedded.dimension == 3
    assert embedded.vector == (0.1, 0.2, 0.3)


def test_embedding_provider_is_runtime_checkable() -> None:
    class FakeProvider:
        dimension = 2
        model_name = "fake"

        def embed_documents(self, texts: list[str]) -> list[tuple[float, ...]]:
            return [(1.0, 0.0) for _ in texts]

        def embed_query(self, text: str) -> tuple[float, ...]:
            return (1.0, 0.0)

    assert isinstance(FakeProvider(), EmbeddingProvider)


def test_search_result_preserves_provenance() -> None:
    result = SearchResult(
        chunk_id="HR-003::0000",
        document_id="HR-003",
        text="Employees may work remotely.",
        score=0.91,
        rank=1,
        title="Remote Work Policy",
        department="HR",
        source_path=Path("data/raw/hr/remote_work_policy.md"),
        metadata={"classification": "internal"},
    )

    assert result.citation_label == "Remote Work Policy (HR-003)"
    assert result.metadata["classification"] == "internal"


def test_search_result_metadata_is_immutable() -> None:
    result = SearchResult(
        chunk_id="IT-001::0000",
        document_id="IT-001",
        text="VPN access guidance",
        score=0.8,
        rank=1,
        title="VPN Access Guide",
        department="IT",
        source_path=Path("vpn.md"),
        metadata={"classification": "internal"},
    )

    with pytest.raises(TypeError):
        result.metadata["classification"] = "public"  # type: ignore[index]
