"""Runtime dependency container for the EnterpriseIQ API service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ServiceDependencyError(RuntimeError):
    """Raised when a required service dependency is unavailable."""


@dataclass(slots=True)
class ServiceContainer:
    """Mutable runtime container for application-level dependencies.

    Expensive runtime components such as the agent orchestrator are attached
    during service startup rather than being created at module import time.
    """

    agent: Any | None = None

    @property
    def agent_ready(self) -> bool:
        """Return whether an agent runtime has been attached."""

        return self.agent is not None

    def require_agent(self) -> Any:
        """Return the configured agent or fail with a controlled error."""

        if self.agent is None:
            raise ServiceDependencyError(
                "EnterpriseIQ agent runtime is not configured."
            )

        return self.agent