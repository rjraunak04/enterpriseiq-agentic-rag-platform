"""EnterpriseIQ embedding interfaces."""

from enterpriseiq.embeddings.config import EmbeddingConfig
from enterpriseiq.embeddings.contracts import (
    EmbeddedChunk,
    EmbeddingProvider,
    Vector,
)
from enterpriseiq.embeddings.factory import create_embedding_provider
from enterpriseiq.embeddings.pipeline import (
    EmbeddingPipelineError,
    EmbeddingPipelineResult,
    embed_chunk_records,
    load_chunk_records,
    run_embedding_pipeline,
)
from enterpriseiq.embeddings.providers import (
    DeterministicEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)

__all__ = [
    "DeterministicEmbeddingProvider",
    "EmbeddedChunk",
    "EmbeddingConfig",
    "EmbeddingPipelineError",
    "EmbeddingPipelineResult",
    "EmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "Vector",
    "create_embedding_provider",
    "embed_chunk_records",
    "load_chunk_records",
    "run_embedding_pipeline",
]
