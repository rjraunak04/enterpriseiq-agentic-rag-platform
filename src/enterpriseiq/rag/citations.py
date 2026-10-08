"""Citation parsing and validation for grounded RAG answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from enterpriseiq.rag.models import (
    Citation,
    EvidenceContext,
    GeneratedAnswer,
    RAGContractError,
)


class CitationValidationError(RAGContractError):
    """Raised when generated citations violate grounding contracts."""


_CITATION_PATTERN = re.compile(r"\[(S[1-9][0-9]*)\]")
_CITATION_LIKE_PATTERN = re.compile(r"\[S[^\]]*\]")


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """Validated citations extracted from a generated answer."""

    citations: tuple[Citation, ...]
    cited_source_ids: tuple[str, ...]
    uncited_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.citations, tuple):
            raise CitationValidationError(
                "citations must be a tuple"
            )

        if not isinstance(self.cited_source_ids, tuple):
            raise CitationValidationError(
                "cited_source_ids must be a tuple"
            )

        if not isinstance(self.uncited_source_ids, tuple):
            raise CitationValidationError(
                "uncited_source_ids must be a tuple"
            )


class CitationValidator:
    """Parse citations and map them to retrieved evidence."""

    def validate(
        self,
        answer: GeneratedAnswer,
        context: EvidenceContext,
    ) -> CitationValidationResult:
        """Validate all answer citations against supplied evidence."""

        if not isinstance(answer, GeneratedAnswer):
            raise CitationValidationError(
                "answer must be a GeneratedAnswer"
            )

        if not isinstance(context, EvidenceContext):
            raise CitationValidationError(
                "context must be an EvidenceContext"
            )

        if answer.abstained:
            return self._validate_abstention(answer, context)

        if not context.items:
            raise CitationValidationError(
                "non-abstained answer requires evidence"
            )

        self._reject_malformed_citations(answer.text)

        extracted = self.extract_source_ids(answer.text)

        if not extracted:
            raise CitationValidationError(
                "non-abstained answer must contain at least one citation"
            )

        evidence_by_id = {
            item.source_id: item
            for item in context.items
        }

        unknown = tuple(
            source_id
            for source_id in extracted
            if source_id not in evidence_by_id
        )

        if unknown:
            joined = ", ".join(unknown)
            raise CitationValidationError(
                f"answer contains unknown source IDs: {joined}"
            )

        unique_ids = tuple(dict.fromkeys(extracted))

        citations = tuple(
            Citation(
                source_id=source_id,
                chunk_id=evidence_by_id[source_id].chunk_id,
                document_id=evidence_by_id[source_id].document_id,
                title=evidence_by_id[source_id].title,
                source_path=evidence_by_id[source_id].source_path,
            )
            for source_id in unique_ids
        )

        cited_set = set(unique_ids)
        uncited = tuple(
            item.source_id
            for item in context.items
            if item.source_id not in cited_set
        )

        return CitationValidationResult(
            citations=citations,
            cited_source_ids=unique_ids,
            uncited_source_ids=uncited,
        )

    @staticmethod
    def extract_source_ids(text: str) -> tuple[str, ...]:
        """Extract valid source IDs in textual occurrence order."""

        if not isinstance(text, str):
            raise CitationValidationError(
                "text must be a string"
            )

        return tuple(_CITATION_PATTERN.findall(text))

    @staticmethod
    def _reject_malformed_citations(text: str) -> None:
        citation_like = _CITATION_LIKE_PATTERN.findall(text)

        malformed = tuple(
            token
            for token in citation_like
            if _CITATION_PATTERN.fullmatch(token) is None
        )

        if malformed:
            joined = ", ".join(malformed)
            raise CitationValidationError(
                f"malformed source citations: {joined}"
            )

    def _validate_abstention(
        self,
        answer: GeneratedAnswer,
        context: EvidenceContext,
    ) -> CitationValidationResult:
        self._reject_malformed_citations(answer.text)

        cited = self.extract_source_ids(answer.text)

        if cited:
            raise CitationValidationError(
                "abstained answer must not contain citations"
            )

        return CitationValidationResult(
            citations=(),
            cited_source_ids=(),
            uncited_source_ids=tuple(
                item.source_id for item in context.items
            ),
        )
