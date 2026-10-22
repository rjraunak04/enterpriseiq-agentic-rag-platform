"""Pluggable answer generation contracts for grounded RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from enterpriseiq.rag.models import GeneratedAnswer, RAGContractError
from enterpriseiq.rag.prompt import (
    DEFAULT_ABSTENTION_TEXT,
    GroundedPrompt,
)


class GenerationError(RAGContractError):
    """Raised when grounded answer generation fails."""


@runtime_checkable
class AnswerGenerator(Protocol):
    """Provider-independent grounded answer generator."""

    def generate(self, prompt: GroundedPrompt) -> GeneratedAnswer:
        """Generate an answer from a structured grounded prompt."""
        ...


@dataclass(frozen=True, slots=True)
class DeterministicGeneratorConfig:
    """Configuration for the deterministic test generator."""

    abstention_text: str = DEFAULT_ABSTENTION_TEXT
    answer_prefix: str = "Grounded evidence:"

    def __post_init__(self) -> None:
        for name in ("abstention_text", "answer_prefix"):
            value = getattr(self, name)

            if not isinstance(value, str) or not value.strip():
                raise GenerationError(
                    f"{name} must be a non-empty string"
                )

            object.__setattr__(self, name, value.strip())


class DeterministicAnswerGenerator:
    """Dependency-free deterministic generator for tests and QA.

    This implementation is not a semantic LLM. It produces a stable,
    citation-bearing response from the first available evidence block.
    """

    def __init__(
        self,
        config: DeterministicGeneratorConfig | None = None,
    ) -> None:
        self.config = config or DeterministicGeneratorConfig()

    def generate(
        self,
        prompt: GroundedPrompt,
    ) -> GeneratedAnswer:
        """Generate a reproducible answer without external API calls."""

        if not isinstance(prompt, GroundedPrompt):
            raise GenerationError(
                "prompt must be a GroundedPrompt"
            )

        if prompt.requires_abstention:
            return GeneratedAnswer(
                text=self.config.abstention_text,
                abstained=True,
                metadata={
                    "provider": "deterministic",
                    "mode": "abstention",
                },
            )

        if not prompt.source_ids:
            raise GenerationError(
                "non-abstention prompt must contain source IDs"
            )

        evidence_text = self._first_evidence_text(prompt.user)

        if not evidence_text:
            raise GenerationError(
                "grounded prompt does not contain usable evidence"
            )

        source_id = prompt.source_ids[0]

        answer = (
            f"{self.config.answer_prefix} "
            f"{evidence_text} [{source_id}]"
        )

        return GeneratedAnswer(
            text=answer,
            abstained=False,
            metadata={
                "provider": "deterministic",
                "mode": "grounded",
                "source_id": source_id,
            },
        )

    @staticmethod
    def _first_evidence_text(user_prompt: str) -> str:
        """Extract the complete first evidence block, including multiline text.

        Evidence documents commonly contain Markdown headings and paragraphs.
        Returning only the line containing the Evidence label silently
        discarded everything after the first newline (often just the title).
        Stop at the next source marker or the response requirements section so
        that additional sources and prompt instructions are not included.
        """

        evidence_lines: list[str] = []
        collecting = False

        for line in user_prompt.splitlines():
            stripped = line.strip()

            if not collecting:
                if line.startswith("Evidence:"):
                    evidence_lines.append(
                        line.removeprefix("Evidence:").strip()
                    )
                    collecting = True
                continue

            if stripped == "RESPONSE REQUIREMENTS:":
                break

            if (
                stripped.startswith("[S")
                and stripped.endswith("]")
                and stripped[2:-1].isdigit()
            ):
                break

            evidence_lines.append(line)

        return "\n".join(evidence_lines).strip()
