"""Grounded retrieval-augmented generation contracts."""

from enterpriseiq.rag.citations import (
    CitationValidationError,
    CitationValidationResult,
    CitationValidator,
)
from enterpriseiq.rag.context import (
    ContextAssemblyError,
    ContextConfig,
    EvidenceContextBuilder,
)
from enterpriseiq.rag.generation import (
    AnswerGenerator,
    DeterministicAnswerGenerator,
    DeterministicGeneratorConfig,
    GenerationError,
)
from enterpriseiq.rag.models import (
    Citation,
    EvidenceContext,
    EvidenceItem,
    GeneratedAnswer,
    GroundedResponse,
    RAGContractError,
    RAGQuery,
)
from enterpriseiq.rag.prompt import (
    DEFAULT_ABSTENTION_TEXT,
    DEFAULT_SYSTEM_POLICY,
    GroundedPrompt,
    GroundedPromptBuilder,
    PromptConfig,
    PromptConstructionError,
)

__all__ = [
    "AnswerGenerator",
    "Citation",
    "CitationValidationError",
    "CitationValidationResult",
    "CitationValidator",
    "ContextAssemblyError",
    "ContextConfig",
    "DEFAULT_ABSTENTION_TEXT",
    "DEFAULT_SYSTEM_POLICY",
    "DeterministicAnswerGenerator",
    "DeterministicGeneratorConfig",
    "EvidenceContext",
    "EvidenceContextBuilder",
    "EvidenceItem",
    "GeneratedAnswer",
    "GenerationError",
    "GroundedPrompt",
    "GroundedPromptBuilder",
    "GroundedResponse",
    "PromptConfig",
    "PromptConstructionError",
    "RAGContractError",
    "RAGQuery",
]
