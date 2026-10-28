"""Configurable, deterministic chunking for enterprise documents."""

from __future__ import annotations

from dataclasses import dataclass

from enterpriseiq.domain import DocumentChunk, EnterpriseDocument


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration controlling deterministic character-based chunking."""

    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_size: int = 100

    def __post_init__(self) -> None:
        for field_name in ("chunk_size", "chunk_overlap", "min_chunk_size"):
            value = getattr(self, field_name)

            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be greater than 0")

        if self.min_chunk_size > self.chunk_size:
            raise ValueError(
                "min_chunk_size must be smaller than or equal to chunk_size"
            )


def _find_boundary(
    text: str,
    start: int,
    target_end: int,
    minimum_end: int,
) -> int:
    """Prefer a semantic text boundary near the target chunk end."""
    if target_end >= len(text):
        return len(text)

    boundary_markers = (
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
    )

    search_region = text[minimum_end:target_end]

    for marker in boundary_markers:
        position = search_region.rfind(marker)

        if position != -1:
            return minimum_end + position + len(marker)

    return target_end


def _trim_span(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Trim boundary whitespace while preserving source character offsets."""
    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    return start, end


def chunk_document(
    document: EnterpriseDocument,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Split one enterprise document into retrieval-ready chunks."""
    if not isinstance(document, EnterpriseDocument):
        raise TypeError("document must be an EnterpriseDocument")

    config = config or ChunkingConfig()
    text = document.content

    if len(text) <= config.chunk_size:
        return [
            DocumentChunk.from_document(
                document=document,
                chunk_id=f"{document.document_id}::0000",
                text=text,
                chunk_index=0,
                start_char=0,
                end_char=len(text),
                metadata={"chunking_strategy": "boundary_aware_character"},
            )
        ]

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        target_end = min(start + config.chunk_size, len(text))

        if target_end == len(text):
            end = len(text)
        else:
            minimum_boundary = min(
                start + config.min_chunk_size,
                target_end,
            )

            end = _find_boundary(
                text=text,
                start=start,
                target_end=target_end,
                minimum_end=minimum_boundary,
            )

        span_start, span_end = _trim_span(text, start, end)

        if span_end <= span_start:
            break

        chunk_text = text[span_start:span_end]

        chunks.append(
            DocumentChunk.from_document(
                document=document,
                chunk_id=f"{document.document_id}::{chunk_index:04d}",
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=span_start,
                end_char=span_end,
                metadata={
                    "chunking_strategy": "boundary_aware_character",
                    "configured_chunk_size": config.chunk_size,
                    "configured_chunk_overlap": config.chunk_overlap,
                },
            )
        )

        if end >= len(text):
            break

        next_start = max(end - config.chunk_overlap, start + 1)

        # Avoid beginning the next passage in whitespace.
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1

        start = next_start
        chunk_index += 1

    return chunks


def chunk_documents(
    documents: list[EnterpriseDocument],
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Chunk a collection while preserving deterministic document order."""
    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    resolved_config = config or ChunkingConfig()
    chunks: list[DocumentChunk] = []

    for document in documents:
        chunks.extend(chunk_document(document, resolved_config))

    return chunks
