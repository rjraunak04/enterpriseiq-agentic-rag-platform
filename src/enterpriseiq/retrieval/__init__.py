"""EnterpriseIQ retrieval interfaces."""

from enterpriseiq.retrieval.models import SearchResult
from enterpriseiq.retrieval.semantic import (
    RetrievalError,
    SearchFilters,
    SemanticRetriever,
    cosine_similarity,
)
from enterpriseiq.retrieval.vector_index import (
    PersistentVectorIndex,
    VectorIndexError,
    VectorIndexManifest,
    VectorIndexRecord,
)

__all__ = [
    "PersistentVectorIndex",
    "RetrievalError",
    "SearchFilters",
    "SearchResult",
    "SemanticRetriever",
    "VectorIndexError",
    "VectorIndexManifest",
    "VectorIndexRecord",
    "cosine_similarity",
]
