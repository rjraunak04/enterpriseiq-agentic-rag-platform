# Advanced Retrieval and Evaluation Architecture

## Overview

EnterpriseIQ Day-03 extends the vector retrieval foundation with lexical retrieval, hybrid rank fusion, reranking, offline benchmarking, retrieval metrics, system comparison, and regression testing.

The objective is to measure retrieval quality before introducing LLM answer generation.

## Retrieval Architecture

Enterprise queries are processed through complementary dense and lexical retrieval paths. Dense retrieval provides semantic matching while BM25 provides deterministic keyword-oriented retrieval. Their ranked candidate lists can be combined using Reciprocal Rank Fusion (RRF), followed by an optional reranking layer.

The resulting evidence is evaluated independently from answer generation.

## Retrieval Components

### BM25 Lexical Retrieval

The lexical retriever provides BM25 Okapi ranking, deterministic tokenization, metadata filtering, duplicate-ID protection, deterministic tie breaking, and SearchResult provenance.

### Hybrid Retrieval

Dense and lexical rankings are combined with Reciprocal Rank Fusion. RRF operates on rank positions rather than directly mixing cosine similarity and BM25 scores with incompatible score distributions.

### Reranking

The pluggable reranking interface operates on retrieved candidates. The current deterministic implementation combines normalized candidate score and lexical query overlap. It provides a reproducible fallback and testing baseline and is not presented as a semantic cross-encoder.

## Gold Retrieval Benchmark

The versioned benchmark at data/evaluation/retrieval_gold_v1.json contains 15 enterprise queries across HR, Finance, IT, Security, Operations, and Product. Relevance judgments are document-level and use graded relevance: 1 for relevant and 2 for highly relevant.

## Retrieval Metrics

EnterpriseIQ evaluates retrieval using Hit@K, Precision@K, Recall@K, Reciprocal Rank, DCG@K, and nDCG@K. Mean Reciprocal Rank is produced by aggregating per-query reciprocal rank values across the benchmark.

## Evaluation Pipeline

The evaluation runner executes each gold query against a retriever, converts chunk-level results into unique document rankings, compares those rankings with gold relevance judgments, computes per-query metrics, and aggregates results across the benchmark.

Multiple retrieval systems can be compared deterministically using nDCG@K, MRR@K, Recall@K, and retriever name as a final tie breaker.

Evaluation results can also be persisted as deterministic JSON reports for reproducibility and regression testing.

## Regression Hardening

Day-03 regression gates cover benchmark integrity, relevance grades, department coverage, metric bounds, perfect and empty retrieval boundaries, deterministic repeated evaluation, reproducible system comparison, stable JSON reports, candidate-depth validation, and benchmark/index consistency when persistent index artifacts are available.

## Day-03 Boundary

Day-03 evaluates evidence retrieval only. LLM answer generation is intentionally excluded so retrieval quality can be measured independently from generation quality. Grounded generation and citation-backed answers begin in Day-04.

