# enterpriseiq-agentic-rag-platform
Production-ready enterprise RAG and agentic AI platform for grounded knowledge retrieval, tool calling, and citation-backed decision intelligence.


---

## Day 02 - Embeddings and Vector Retrieval

EnterpriseIQ now includes pluggable embedding providers, batch embedding generation, a persistent vector index, metadata-aware cosine retrieval, provenance-rich search results, CLI workflows, artifact integrity validation, and retrieval hardening tests.

The deterministic embedding provider is used for reproducible tests and CI and is not a semantic-quality benchmark. A lazy SentenceTransformer provider supports real local semantic embeddings.

Retrieval supports department, classification, document-type, and document-ID filtering while preserving source provenance for downstream citation-backed RAG.

Architecture: docs/architecture/vector-retrieval-architecture.md

**Status:** Day 01 complete | Day 02 complete | Day 03 next: Advanced Retrieval + Evaluation

## Advanced Retrieval & Evaluation

EnterpriseIQ includes an offline-evaluated retrieval stack with BM25 lexical retrieval, dense semantic retrieval, Reciprocal Rank Fusion, pluggable reranking, metadata filtering, a versioned 15-query enterprise benchmark, Hit@K / Precision@K / Recall@K / MRR / nDCG evaluation, deterministic system comparison, JSON reports, and regression quality gates.

Retrieval quality is evaluated independently before LLM answer generation. See docs/architecture/advanced-retrieval-evaluation.md.

## Grounded RAG & Citations

EnterpriseIQ includes an end-to-end grounded RAG layer built on top of its advanced retrieval and evaluation stack.

The pipeline converts ranked retrieval results into bounded evidence contexts, applies an evidence-only generation policy, invokes a provider-independent answer generator, validates inline `[S#]` citations, and maps accepted citations back to trusted document and chunk provenance.

Key engineering properties include:

- immutable query, evidence, citation, and response contracts;
- configurable context chunk and character budgets;
- explicit abstention when evidence is unavailable;
- rejection of malformed and hallucinated source IDs;
- citation provenance derived from retrieved evidence;
- prompt-injection boundaries for instructions embedded in documents;
- provider-independent generation interfaces;
- deterministic credential-free integration testing;
- regression gates across retrieval, generation, citations, and grounding.

The deterministic generator is a reproducibility and integration-testing component, not a semantic LLM implementation.

Architecture details: `docs/architecture/grounded-rag-citations.md`.
