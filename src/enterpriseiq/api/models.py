"""Transport models for the EnterpriseIQ API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictAPIModel(BaseModel):
    """Base model rejecting unexpected transport fields."""

    model_config = ConfigDict(extra="forbid")


class AgentQueryRequest(StrictAPIModel):
    """HTTP request contract for an EnterpriseIQ agent query."""

    query: str = Field(
        min_length=1,
        max_length=4000,
    )
    query_id: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class CitationResponse(StrictAPIModel):
    """Citation exposed through the HTTP boundary."""

    source_id: str
    document_id: str | None = None
    title: str | None = None


class ToolResultResponse(StrictAPIModel):
    """Safe transport representation of a tool result."""

    tool_name: str
    status: str
    call_id: str | None = None
    output: Any = None
    error: str | None = None


class AgentQueryResponse(StrictAPIModel):
    """HTTP representation of an agent execution result."""

    query_id: str | None = None
    status: str
    decision: str
    answer: str
    citations: list[CitationResponse] = Field(
        default_factory=list
    )
    tool_results: list[ToolResultResponse] = Field(
        default_factory=list
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class HealthResponse(StrictAPIModel):
    """Liveness response."""

    status: str
    service: str
    version: str


class ReadinessResponse(StrictAPIModel):
    """Readiness response."""

    status: str
    ready: bool
    checks: dict[str, bool] = Field(
        default_factory=dict
    )


class ErrorResponse(StrictAPIModel):
    """Sanitized API error response."""

    error: str
    request_id: str | None = None