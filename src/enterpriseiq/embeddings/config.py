"""Embedding configuration for EnterpriseIQ."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration shared by EnterpriseIQ embedding providers."""

    provider: str = "local"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        model_name = self.model_name.strip()

        if not provider:
            raise ValueError("provider must not be empty")
        if not model_name:
            raise ValueError("model_name must not be empty")

        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("dimension must be an integer")
        if self.dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be a boolean")

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model_name", model_name)
