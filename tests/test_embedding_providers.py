"""Tests for EnterpriseIQ embedding providers."""

import math

import pytest

from enterpriseiq.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingConfig,
    SentenceTransformerEmbeddingProvider,
    create_embedding_provider,
)


def make_config(
    *,
    provider: str = "deterministic",
    dimension: int = 8,
    normalize: bool = True,
) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=provider,
        model_name="test-model",
        dimension=dimension,
        batch_size=4,
        normalize=normalize,
    )


def vector_norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_deterministic_provider_is_reproducible() -> None:
    provider = DeterministicEmbeddingProvider(make_config())

    first = provider.embed_query("remote work policy")
    second = provider.embed_query("remote work policy")

    assert first == second


def test_deterministic_provider_changes_with_text() -> None:
    provider = DeterministicEmbeddingProvider(make_config())

    first = provider.embed_query("remote work policy")
    second = provider.embed_query("expense reimbursement")

    assert first != second


def test_deterministic_provider_respects_dimension() -> None:
    provider = DeterministicEmbeddingProvider(make_config(dimension=17))

    vector = provider.embed_query("security incident")

    assert len(vector) == 17


def test_deterministic_provider_normalizes_vectors() -> None:
    provider = DeterministicEmbeddingProvider(make_config(normalize=True))

    vector = provider.embed_query("VPN access")

    assert vector_norm(vector) == pytest.approx(1.0)


def test_deterministic_provider_can_skip_normalization() -> None:
    provider = DeterministicEmbeddingProvider(make_config(normalize=False))

    vector = provider.embed_query("VPN access")

    assert len(vector) == provider.dimension
    assert vector_norm(vector) != pytest.approx(1.0)


def test_document_embedding_preserves_input_order() -> None:
    provider = DeterministicEmbeddingProvider(make_config())

    texts = ["alpha", "beta", "gamma"]
    vectors = provider.embed_documents(texts)

    assert vectors == [provider.embed_query(text) for text in texts]


def test_empty_document_collection_returns_empty_list() -> None:
    provider = DeterministicEmbeddingProvider(make_config())

    assert provider.embed_documents([]) == []


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_provider_rejects_empty_text(text: str) -> None:
    provider = DeterministicEmbeddingProvider(make_config())

    with pytest.raises(ValueError, match="must not be empty"):
        provider.embed_query(text)


def test_provider_rejects_string_as_document_collection() -> None:
    provider = DeterministicEmbeddingProvider(make_config())

    with pytest.raises(TypeError, match="sequence of strings"):
        provider.embed_documents("not-a-sequence")


def test_factory_builds_deterministic_provider() -> None:
    provider = create_embedding_provider(make_config())

    assert isinstance(provider, DeterministicEmbeddingProvider)


def test_factory_builds_sentence_transformer_provider() -> None:
    provider = create_embedding_provider(
        make_config(provider="sentence-transformers")
    )

    assert isinstance(provider, SentenceTransformerEmbeddingProvider)


def test_factory_rejects_unknown_provider() -> None:
    config = make_config(provider="unknown-provider")

    with pytest.raises(ValueError, match="unsupported embedding provider"):
        create_embedding_provider(config)


def test_sentence_transformer_provider_is_lazy() -> None:
    provider = SentenceTransformerEmbeddingProvider(
        make_config(provider="sentence-transformers")
    )

    assert provider._model is None
