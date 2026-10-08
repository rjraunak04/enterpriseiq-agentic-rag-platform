"""Deterministic evidence context assembly for grounded RAG."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from enterpriseiq.rag.models import (
    EvidenceContext,
    EvidenceItem,
    RAGContractError,
)
from enterpriseiq.retrieval.models import SearchResult


class ContextAssemblyError(RAGContractError):
    """Raised when retrieved evidence cannot be assembled safely."""


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Configuration for bounded evidence context assembly."""

    max_chunks: int = 8
    max_chars: int = 12000

    def __post_init__(self) -> None:
        for name in ("max_chunks", "max_chars"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ContextAssemblyError(
                    f"{name} must be a positive integer"
                )


class EvidenceContextBuilder:
    """Convert ranked retrieval results into citation-ready context."""

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()

    def build(
        self,
        results: Sequence[SearchResult],
    ) -> EvidenceContext:
        """Build deterministic bounded context from retrieval results."""

        if isinstance(results, (str, bytes)) or not isinstance(
            results, Sequence
        ):
            raise ContextAssemblyError(
                "results must be a sequence of SearchResult objects"
            )

        if not results:
            return EvidenceContext(
                items=(),
                rendered_text="",
                truncated=False,
            )

        self._validate_results(results)

        ordered = sorted(
            results,
            key=lambda result: (result.rank, result.chunk_id),
        )

        items: list[EvidenceItem] = []
        rendered_blocks: list[str] = []
        used_chars = 0
        truncated = False

        for result in ordered:
            if len(items) >= self.config.max_chunks:
                truncated = True
                break

            source_id = f"S{len(items) + 1}"

            item = EvidenceItem(
                source_id=source_id,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                title=result.title,
                department=result.department,
                source_path=result.source_path,
                retrieval_score=result.score,
                retrieval_rank=result.rank,
                metadata=result.metadata,
            )

            block = self._render_item(item)

            separator_size = 2 if rendered_blocks else 0
            required_chars = len(block) + separator_size

            if used_chars + required_chars > self.config.max_chars:
                truncated = True
                break

            items.append(item)
            rendered_blocks.append(block)
            used_chars += required_chars

        if len(items) < len(ordered):
            truncated = True

        return EvidenceContext(
            items=tuple(items),
            rendered_text="\n\n".join(rendered_blocks),
            truncated=truncated,
        )

    @staticmethod
    def _validate_results(
        results: Sequence[SearchResult],
    ) -> None:
        seen_chunk_ids: set[str] = set()

        for result in results:
            if not isinstance(result, SearchResult):
                raise ContextAssemblyError(
                    "all results must be SearchResult objects"
                )

            if result.chunk_id in seen_chunk_ids:
                raise ContextAssemblyError(
                    f"duplicate chunk_id: {result.chunk_id}"
                )

            seen_chunk_ids.add(result.chunk_id)

    @staticmethod
    def _render_item(item: EvidenceItem) -> str:
        return "\n".join(
            (
                f"[{item.source_id}]",
                f"Title: {item.title}",
                f"Document: {item.document_id}",
                f"Department: {item.department}",
                f"Chunk: {item.chunk_id}",
                f"Evidence: {item.text}",
            )
        )
