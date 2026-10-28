# Grounded RAG and Citation Architecture

## Overview

EnterpriseIQ implements a grounded retrieval-augmented generation (RAG) pipeline for enterprise knowledge retrieval and citation-backed answer generation.

Day-04 extends the advanced retrieval and evaluation foundation completed in Day-03 by separating retrieval, evidence assembly, prompt construction, generation, citation validation, and final response construction into explicit contracts.

## Architecture

Enterprise Query
-> Advanced Retrieval
-> Ranked Search Results
-> Evidence Context Builder
-> Bounded Evidence Context
-> Grounded Prompt Builder
-> Pluggable Answer Generator
-> Citation Validator
-> GroundedResponse

## RAG Contracts

The RAG layer defines immutable contracts for queries, retrieved evidence, bounded evidence contexts, generated answers, citations, and final grounded responses.

Evidence retains trusted provenance including source ID, chunk ID, document ID, title, department, source path, retrieval score, retrieval rank, and metadata.

## Evidence Context Assembly

Retrieved results are converted into deterministic evidence blocks with prompt-local identifiers such as [S1], [S2], and [S3].

Context assembly enforces configurable chunk-count and character limits. Complete evidence blocks are omitted when the context budget is exceeded instead of partially truncating evidence.

## Grounding Policy

The generation policy requires answers to use only supplied evidence, avoid unsupported claims, cite supplied source IDs, never invent source identifiers, and abstain when evidence is insufficient.

Instructions contained inside retrieved documents are treated as evidence data rather than authoritative instructions.

## Pluggable Generation

Answer generation is provider-independent through the AnswerGenerator protocol.

The deterministic generator is dependency-free and is used for testing, reproducibility, and end-to-end integration verification. It is not a semantic LLM implementation.

This abstraction allows external LLM providers to be integrated without changing retrieval, citation, or response contracts.

## Citation Validation

Generated answers reference evidence through inline identifiers such as [S1].

The citation layer:

- extracts source identifiers;
- rejects malformed citations;
- rejects unknown or hallucinated source IDs;
- requires citations for non-abstained answers;
- rejects citations in abstained answers;
- deduplicates repeated citations deterministically;
- preserves citation order;
- maps accepted citations back to trusted retrieved evidence.

Citation provenance is constructed from the evidence context rather than metadata written by the generator.

## Grounded RAG Orchestration

GroundedRAGPipeline coordinates:

RAGQuery
-> Retriever
-> EvidenceContextBuilder
-> GroundedPromptBuilder
-> AnswerGenerator
-> CitationValidator
-> GroundedResponse

The final response records the normalized query, generated answer, validated citations, grounding status, abstention status, evidence count, retrieval count, context count, context truncation state, cited source IDs, and uncited source IDs.

## Abstention Behavior

When retrieval returns no usable evidence, the pipeline produces an explicit no-evidence state and abstains instead of fabricating an answer.

An abstained response is not marked grounded and contains no citations.

## Prompt-Injection Boundary

Retrieved enterprise documents are treated as untrusted generation input.

Evidence may contain text attempting to override instructions or disable grounding requirements. The prompt contract explicitly states that instructions found inside retrieved evidence must not be followed.

Regression tests verify that malicious evidence remains evidence and does not replace the grounding policy.

## Failure and Regression Gates

Day-04 regression coverage includes hallucinated citations, malformed citations, uncited answers, invalid abstention citations, empty retrieval, duplicate evidence, bounded contexts, prompt-injection content, provenance spoofing, generator failures, retrieval failures, blank queries, filter propagation, and deterministic execution.

Infrastructure failures are not silently converted into fabricated answers.

## Security and Trust Boundary

The citation trust path is:

Retrieved Evidence
-> Prompt-local Source ID
-> Generated [S#] Reference
-> Citation Validation
-> Original Evidence Provenance

Only source identifiers already present in the evidence context can produce valid citations.

## Determinism

The deterministic test path provides stable evidence ordering, source-ID assignment, prompt construction, generation, citation extraction, citation deduplication, provenance mapping, and final grounded responses.

## Day-04 Scope Boundary

Day-04 provides grounded RAG and citation-backed answer generation.

Autonomous agents, tool selection, tool execution, multi-agent orchestration, and external side effects are intentionally deferred to Day-05.

## Production Extension Path

The architecture supports future hosted or local LLM generators, retry and timeout policies, structured generation, richer citation verification, answer-quality evaluation, observability, tenant-aware authorization, and agent tool execution while preserving the grounded evidence and citation contracts.
