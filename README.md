# enterpriseiq-agentic-rag-platform
Production-ready enterprise RAG and agentic AI platform for grounded knowledge retrieval, tool calling, and citation-backed decision intelligence.


---

## Day 02 - Embeddings and Vector Retrieval

EnterpriseIQ now includes pluggable embedding providers, batch embedding generation, a persistent vector index, metadata-aware cosine retrieval, provenance-rich search results, CLI workflows, artifact integrity validation, and retrieval hardening tests.

The deterministic embedding provider is used for reproducible tests and CI and is not a semantic-quality benchmark. A lazy SentenceTransformer provider supports real local semantic embeddings.

Retrieval supports department, classification, document-type, and document-ID filtering while preserving source provenance for downstream citation-backed RAG.

Architecture: docs/architecture/vector-retrieval-architecture.md

**Status:** Day 01 complete | Day 02 complete | Day 03 next: Advanced Retrieval + Evaluation
