# EnterpriseIQ Ingestion Architecture

## Overview

EnterpriseIQ is a production-oriented enterprise RAG and agentic AI platform for grounded knowledge retrieval, citation-backed answers, and agentic workflows.

Day 01 establishes the deterministic ingestion foundation required before embeddings, retrieval, generation, and agent orchestration are introduced.

## Day-01 Pipeline

```text
Enterprise Markdown Corpus
        |
        v
Metadata-Aware Loader
        |
        v
Front-Matter Validation
        |
        v
Markdown-Preserving Cleaner
        |
        v
EnterpriseDocument
        |
        v
Boundary-Aware Chunker
        |
        v
DocumentChunk
        |
        +------------------------+
        |                        |
        v                        v
chunks.jsonl          ingestion_manifest.json
                               |
                               v
                       SHA-256 Integrity