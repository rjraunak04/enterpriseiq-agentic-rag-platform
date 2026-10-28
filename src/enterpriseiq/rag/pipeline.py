"""End-to-end grounded retrieval-augmented generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from enterpriseiq.rag.citations import CitationValidator
from enterpriseiq.rag.context import EvidenceContextBuilder
from enterpriseiq.rag.generation import AnswerGenerator
from enterpriseiq.rag.models import (
    GroundedResponse,
    RAGContractError,
    RAGQuery,
)
from enterpriseiq.rag.prompt import GroundedPromptBuilder
from enterpriseiq.retrieval.semantic import SearchFilters


class RAGPipelineError(RAGContractError):
    """Raised when grounded RAG orchestration cannot be completed."""


@runtime_checkable
class RAGRetriever(Protocol):
    """Retriever contract required by the grounded RAG pipeline."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        min_score: float | None = None,
    ) -> list:
        """Return ranked retrieval results."""
        ...


@dataclass(frozen=True, slots=True)
class RAGPipelineConfig:
    """Runtime configuration for grounded RAG orchestration."""

    top_k: int = 5
    min_score: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise RAGPipelineError(
                "top_k must be a positive integer"
            )

        if self.min_score is not None:
            if (
                isinstance(self.min_score, bool)
                or not isinstance(self.min_score, (int, float))
            ):
                raise RAGPipelineError(
                    "min_score must be numeric or None"
                )

            object.__setattr__(
                self,
                "min_score",
                float(self.min_score),
            )


class GroundedRAGPipeline:
    """Coordinate retrieval, grounding, generation, and citations."""

    def __init__(
        self,
        *,
        retriever: RAGRetriever,
        generator: AnswerGenerator,
        context_builder: EvidenceContextBuilder | None = None,
        prompt_builder: GroundedPromptBuilder | None = None,
        citation_validator: CitationValidator | None = None,
        config: RAGPipelineConfig | None = None,
    ) -> None:
        if not isinstance(retriever, RAGRetriever):
            raise RAGPipelineError(
                "retriever must satisfy the RAGRetriever protocol"
            )

        if not isinstance(generator, AnswerGenerator):
            raise RAGPipelineError(
                "generator must satisfy the AnswerGenerator protocol"
            )

        self.retriever = retriever
        self.generator = generator
        self.context_builder = (
            context_builder or EvidenceContextBuilder()
        )
        self.prompt_builder = (
            prompt_builder or GroundedPromptBuilder()
        )
        self.citation_validator = (
            citation_validator or CitationValidator()
        )
        self.config = config or RAGPipelineConfig()

    def answer(
        self,
        query: str | RAGQuery,
        *,
        filters: SearchFilters | None = None,
    ) -> GroundedResponse:
        """Answer a query using only retrieved and validated evidence."""

        rag_query = self._normalize_query(query)

        results = self.retriever.search(
            rag_query.text,
            top_k=self.config.top_k,
            filters=filters,
            min_score=self.config.min_score,
        )

        context = self.context_builder.build(results)
        prompt = self.prompt_builder.build(rag_query, context)
        generated = self.generator.generate(prompt)

        validation = self.citation_validator.validate(
            generated,
            context,
        )

        grounded = (
            not generated.abstained
            and bool(validation.citations)
        )

        return GroundedResponse(
            query=rag_query,
            answer=generated.text,
            citations=validation.citations,
            grounded=grounded,
            abstained=generated.abstained,
            evidence_count=len(context.items),
            metadata={
                "retrieved_count": len(results),
                "context_count": len(context.items),
                "context_truncated": context.truncated,
                "cited_source_ids": validation.cited_source_ids,
                "uncited_source_ids": validation.uncited_source_ids,
            },
        )

    @staticmethod
    def _normalize_query(
        query: str | RAGQuery,
    ) -> RAGQuery:
        if isinstance(query, RAGQuery):
            return query

        if isinstance(query, str):
            return RAGQuery(text=query)

        raise RAGPipelineError(
            "query must be a string or RAGQuery"
        )
