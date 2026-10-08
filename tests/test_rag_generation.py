"""Tests for provider-independent grounded answer generation."""

from pathlib import Path

import pytest

from enterpriseiq.rag import (
    AnswerGenerator,
    DeterministicAnswerGenerator,
    DeterministicGeneratorConfig,
    EvidenceContext,
    EvidenceItem,
    GeneratedAnswer,
    GenerationError,
    GroundedPrompt,
    GroundedPromptBuilder,
    RAGQuery,
)


def _context() -> EvidenceContext:
    item = EvidenceItem(
        source_id="S1",
        chunk_id="chunk-1",
        document_id="HR-003",
        text="Employees may work remotely up to three days per week.",
        title="Remote Work Policy",
        department="HR",
        source_path=Path("hr/remote_work_policy.md"),
        retrieval_score=0.95,
        retrieval_rank=1,
        metadata={"classification": "internal"},
    )

    rendered = "\n".join(
        (
            "[S1]",
            "Title: Remote Work Policy",
            "Document: HR-003",
            "Department: HR",
            "Chunk: chunk-1",
            (
                "Evidence: Employees may work remotely "
                "up to three days per week."
            ),
        )
    )

    return EvidenceContext(
        items=(item,),
        rendered_text=rendered,
    )


def _grounded_prompt() -> GroundedPrompt:
    return GroundedPromptBuilder().build(
        RAGQuery(
            text="How many days may employees work remotely?"
        ),
        _context(),
    )


def _abstention_prompt() -> GroundedPrompt:
    return GroundedPromptBuilder().build(
        RAGQuery(text="What is the unknown policy?"),
        EvidenceContext(
            items=(),
            rendered_text="",
        ),
    )


def test_protocol_accepts_deterministic_generator() -> None:
    generator = DeterministicAnswerGenerator()

    assert isinstance(generator, AnswerGenerator)


def test_deterministic_generator_returns_generated_answer() -> None:
    answer = DeterministicAnswerGenerator().generate(
        _grounded_prompt()
    )

    assert isinstance(answer, GeneratedAnswer)


def test_grounded_generation_is_not_abstained() -> None:
    answer = DeterministicAnswerGenerator().generate(
        _grounded_prompt()
    )

    assert answer.abstained is False


def test_grounded_generation_contains_evidence_text() -> None:
    answer = DeterministicAnswerGenerator().generate(
        _grounded_prompt()
    )

    assert (
        "Employees may work remotely up to three days per week."
        in answer.text
    )


def test_grounded_generation_contains_source_citation() -> None:
    answer = DeterministicAnswerGenerator().generate(
        _grounded_prompt()
    )

    assert "[S1]" in answer.text


def test_generation_metadata_identifies_provider() -> None:
    answer = DeterministicAnswerGenerator().generate(
        _grounded_prompt()
    )

    assert answer.metadata["provider"] == "deterministic"
    assert answer.metadata["mode"] == "grounded"
    assert answer.metadata["source_id"] == "S1"


def test_abstention_prompt_produces_abstention() -> None:
    generator = DeterministicAnswerGenerator()

    answer = generator.generate(_abstention_prompt())

    assert answer.abstained is True
    assert answer.text == (
        "I cannot answer this question from the available evidence."
    )


def test_abstention_metadata_is_recorded() -> None:
    answer = DeterministicAnswerGenerator().generate(
        _abstention_prompt()
    )

    assert answer.metadata["provider"] == "deterministic"
    assert answer.metadata["mode"] == "abstention"


def test_custom_config_is_applied() -> None:
    generator = DeterministicAnswerGenerator(
        DeterministicGeneratorConfig(
            abstention_text="NO_SUPPORTED_EVIDENCE",
            answer_prefix="Evidence says:",
        )
    )

    answer = generator.generate(_grounded_prompt())

    assert answer.text.startswith("Evidence says:")
    assert answer.text.endswith("[S1]")


def test_custom_abstention_text_is_applied() -> None:
    generator = DeterministicAnswerGenerator(
        DeterministicGeneratorConfig(
            abstention_text="NO_SUPPORTED_EVIDENCE",
        )
    )

    answer = generator.generate(_abstention_prompt())

    assert answer.text == "NO_SUPPORTED_EVIDENCE"


@pytest.mark.parametrize(
    "field",
    ["abstention_text", "answer_prefix"],
)
def test_config_rejects_empty_strings(field: str) -> None:
    kwargs = {field: "   "}

    with pytest.raises(GenerationError):
        DeterministicGeneratorConfig(**kwargs)


def test_generator_rejects_invalid_prompt() -> None:
    with pytest.raises(
        GenerationError,
        match="GroundedPrompt",
    ):
        DeterministicAnswerGenerator().generate("invalid")


def test_non_abstention_prompt_requires_source_ids() -> None:
    prompt = GroundedPrompt(
        system="System",
        user="QUESTION:\nQuestion\n\nEVIDENCE:\nEvidence: fact",
        source_ids=(),
        requires_abstention=False,
    )

    with pytest.raises(
        GenerationError,
        match="source IDs",
    ):
        DeterministicAnswerGenerator().generate(prompt)


def test_missing_evidence_field_is_rejected() -> None:
    prompt = GroundedPrompt(
        system="System",
        user="QUESTION:\nQuestion\n\nEVIDENCE:\nNo usable field",
        source_ids=("S1",),
        requires_abstention=False,
    )

    with pytest.raises(
        GenerationError,
        match="usable evidence",
    ):
        DeterministicAnswerGenerator().generate(prompt)


def test_repeated_generation_is_deterministic() -> None:
    generator = DeterministicAnswerGenerator()
    prompt = _grounded_prompt()

    first = generator.generate(prompt)
    second = generator.generate(prompt)

    assert first == second


def test_generator_uses_first_available_source() -> None:
    prompt = GroundedPrompt(
        system="System",
        user="\n".join(
            (
                "QUESTION:",
                "Question",
                "",
                "EVIDENCE:",
                "[S2]",
                "Evidence: First ranked evidence.",
                "[S8]",
                "Evidence: Second ranked evidence.",
            )
        ),
        source_ids=("S2", "S8"),
        requires_abstention=False,
    )

    answer = DeterministicAnswerGenerator().generate(prompt)

    assert "First ranked evidence." in answer.text
    assert "[S2]" in answer.text
    assert "[S8]" not in answer.text


def test_generator_does_not_require_external_credentials() -> None:
    generator = DeterministicAnswerGenerator()

    answer = generator.generate(_grounded_prompt())

    assert answer.metadata["provider"] == "deterministic"
