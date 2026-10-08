"""Grounded retrieval-augmented generation contracts."""

from enterpriseiq.rag.context import (
    ContextAssemblyError,
    ContextConfig,
    EvidenceContextBuilder,
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
    "Citation",
    "ContextAssemblyError",
    "ContextConfig",
    "DEFAULT_ABSTENTION_TEXT",
    "DEFAULT_SYSTEM_POLICY",
    "EvidenceContext",
    "EvidenceContextBuilder",
    "EvidenceItem",
    "GeneratedAnswer",
    "GroundedPrompt",
    "GroundedPromptBuilder",
    "GroundedResponse",
    "PromptConfig",
    "PromptConstructionError",
    "RAGContractError",
    "RAGQuery",
]
