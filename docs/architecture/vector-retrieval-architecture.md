# EnterpriseIQ Vector Retrieval Architecture

## Overview

Day 02 extends the EnterpriseIQ ingestion pipeline with reproducible embedding generation, persistent vector indexing, metadata-aware retrieval, and provenance-preserving search results.

## Retrieval Architecture

Enterprise documents -> Day-01 ingestion -> validated chunks -> embedding provider -> embedding artifact -> persistent vector index -> query embedding -> metadata filtering -> cosine similarity -> ranked evidence with provenance.

## Embedding Providers

EnterpriseIQ uses a pluggable embedding-provider contract. The deterministic provider supports reproducible tests, CI, and pipeline verification; it is not a semantic-quality benchmark. The SentenceTransformer provider is the local semantic embedding backend and is loaded lazily.

## Embedding Pipeline

The batch pipeline preserves chunk IDs, document IDs, text, department, classification, document type, version, owner, source path, and metadata. It validates vector count, embedding dimensions, finite values, and artifact consistency.

## Persistent Vector Index

The vector index persists vectors together with enterprise metadata and source provenance. Its manifest records schema information, record count, embedding dimension, model name, and SHA-256 integrity information. Index loading validates persisted artifacts before retrieval.

## Semantic Retrieval

SemanticRetriever performs query embedding, metadata pre-filtering, cosine similarity scoring, optional minimum-score filtering, deterministic top-k ranking, and provenance-rich SearchResult construction.

Supported filters include department, classification, document type, and document ID.

## Reliability

Day-02 tests cover malformed artifacts, duplicate chunk IDs, incompatible dimensions and models, non-finite vectors, invalid manifests, record-count mismatches, corrupted indexes, malformed query vectors, and provenance persistence.

## Milestone Boundary

Day 02 implements embedding and vector retrieval only. Hybrid retrieval and retrieval evaluation belong to Day 03; grounded LLM answer generation and citations belong to Day 04.
