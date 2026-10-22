"""HTTP boundary for EnterpriseIQ agent execution."""

from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from enterpriseiq.api.dependencies import (
    ServiceContainer,
    ServiceDependencyError,
)
from enterpriseiq.api.middleware import get_request_id, log_event
from enterpriseiq.api.models import (
    AgentQueryRequest,
    AgentQueryResponse,
    CitationResponse,
    ErrorResponse,
    ToolResultResponse,
)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


def _enum_value(value: Any) -> Any:
    """Return a stable primitive representation for enum values."""

    if isinstance(value, Enum):
        return value.value

    return value


def _string_value(
    value: Any,
    *,
    default: str = "",
) -> str:
    """Convert safe primitive values to strings."""

    value = _enum_value(value)

    if value is None:
        return default

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)):
        return str(value)

    return default


def _mapping(value: Any) -> dict[str, Any]:
    """Convert supported structured objects into a shallow mapping."""

    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if is_dataclass(value) and not isinstance(value, type):
        converted = asdict(value)

        if isinstance(converted, dict):
            return converted

    model_dump = getattr(value, "model_dump", None)

    if callable(model_dump):
        converted = model_dump()

        if isinstance(converted, dict):
            return converted

    return {}


def _attribute(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    """Read a field from either a mapping or structured object."""

    if isinstance(value, dict):
        return value.get(name, default)

    return getattr(value, name, default)


def _safe_json_value(value: Any) -> Any:
    """Convert values into JSON-safe data without arbitrary repr leakage."""

    value = _enum_value(value)

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): _safe_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _safe_json_value(item)
            for item in value
        ]

    structured = _mapping(value)

    if structured:
        return _safe_json_value(structured)

    return None


def _serialize_citation(value: Any) -> CitationResponse | None:
    """Serialize one citation without exposing internal objects."""

    source_id = _string_value(
        _attribute(value, "source_id")
        or _attribute(value, "source")
        or _attribute(value, "evidence_id")
    )

    if not source_id:
        return None

    document_id = _string_value(
        _attribute(value, "document_id"),
        default="",
    )
    title = _string_value(
        _attribute(value, "title"),
        default="",
    )

    return CitationResponse(
        source_id=source_id,
        document_id=document_id or None,
        title=title or None,
    )


def _serialize_tool_result(
    value: Any,
) -> ToolResultResponse | None:
    """Serialize one controlled tool execution result."""

    tool_name = _string_value(
        _attribute(value, "tool_name")
        or _attribute(value, "name")
    )

    if not tool_name:
        return None

    status = _string_value(
        _attribute(value, "status"),
        default="unknown",
    )
    call_id = _string_value(
        _attribute(value, "call_id"),
        default="",
    )
    error = _string_value(
        _attribute(value, "error"),
        default="",
    )

    return ToolResultResponse(
        tool_name=tool_name,
        status=status,
        call_id=call_id or None,
        output=_safe_json_value(
            _attribute(value, "output")
        ),
        error=error or None,
    )


def _extract_decision(response: Any) -> str:
    """Extract a stable agent routing decision."""

    decision = _attribute(
        response,
        "decision",
    )

    if decision is None:
        return "unknown"

    nested_type = _attribute(
        decision,
        "decision_type",
    )

    if nested_type is not None:
        return _string_value(
            nested_type,
            default="unknown",
        )

    return _string_value(
        decision,
        default="unknown",
    )


def _extract_answer(response: Any) -> str:
    """Extract final answer text from supported agent response shapes."""

    for field_name in (
        "answer",
        "final_answer",
        "text",
    ):
        value = _attribute(
            response,
            field_name,
        )

        text = _string_value(value)

        if text:
            return text

    grounded_response = _attribute(
        response,
        "grounded_response",
    )

    if grounded_response is not None:
        for field_name in (
            "answer",
            "text",
        ):
            value = _attribute(
                grounded_response,
                field_name,
            )
            text = _string_value(value)

            if text:
                return text

    return ""


def _extract_citations(
    response: Any,
) -> list[CitationResponse]:
    """Extract public citation contracts."""

    candidates = _attribute(
        response,
        "citations",
        None,
    )

    if candidates is None:
        grounded_response = _attribute(
            response,
            "grounded_response",
        )
        candidates = _attribute(
            grounded_response,
            "citations",
            None,
        )

    if not candidates:
        metadata = _attribute(response, "metadata", {})
        if isinstance(metadata, dict):
            candidates = metadata.get("rag_citations", candidates)

    if not isinstance(
        candidates,
        (list, tuple),
    ):
        return []

    citations: list[CitationResponse] = []

    for candidate in candidates:
        citation = _serialize_citation(candidate)

        if citation is not None:
            citations.append(citation)

    return citations


def _extract_tool_results(
    response: Any,
) -> list[ToolResultResponse]:
    """Extract safe tool execution summaries."""

    candidates = _attribute(
        response,
        "tool_results",
        [],
    )

    if not isinstance(
        candidates,
        (list, tuple),
    ):
        return []

    results: list[ToolResultResponse] = []

    for candidate in candidates:
        result = _serialize_tool_result(candidate)

        if result is not None:
            results.append(result)

    return results


def serialize_agent_response(
    response: Any,
    *,
    query_id: str | None,
) -> AgentQueryResponse:
    """Map the internal agent response onto the HTTP contract."""

    status = _string_value(
        _attribute(response, "status"),
        default="unknown",
    )

    metadata = _safe_json_value(
        _attribute(response, "metadata", {})
    )

    if not isinstance(metadata, dict):
        metadata = {}

    return AgentQueryResponse(
        query_id=query_id,
        status=status,
        decision=_extract_decision(response),
        answer=_extract_answer(response),
        citations=_extract_citations(response),
        tool_results=_extract_tool_results(response),
        metadata=metadata,
    )


def _get_container(request: Request) -> ServiceContainer:
    """Resolve the runtime service container."""

    container = getattr(
        request.app.state,
        "service_container",
        None,
    )

    if not isinstance(container, ServiceContainer):
        raise RuntimeError(
            "EnterpriseIQ service container is unavailable."
        )

    return container


async def _run_agent(
    agent: Any,
    query: str,
) -> Any:
    """Execute sync or async compatible agent runtimes."""

    run = agent.run

    if inspect.iscoroutinefunction(run):
        return await run(query)

    result = await run_in_threadpool(run, query)

    if inspect.isawaitable(result):
        return await result

    return result


@router.post(
    "",
    response_model=AgentQueryResponse,
    responses={
        503: {
            "model": ErrorResponse,
            "description": "Agent runtime is unavailable.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Agent execution failed.",
        },
    },
    summary="Execute an EnterpriseIQ agent query",
)
async def query_agent(
    payload: AgentQueryRequest,
    request: Request,
) -> AgentQueryResponse | JSONResponse:
    """Execute a grounded EnterpriseIQ agent request."""

    container = _get_container(request)

    try:
        agent = container.require_agent()
    except ServiceDependencyError:
        error = ErrorResponse(
            error="Agent runtime is unavailable.",
            request_id=get_request_id(),
        )

        return JSONResponse(
            status_code=503,
            content=error.model_dump(),
        )

    query = payload.query.strip()
    config = request.app.state.api_config

    if not query:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Query must contain non-whitespace characters.",
                "request_id": get_request_id(),
            },
        )

    if len(query) > config.max_query_length:
        return JSONResponse(
            status_code=422,
            content={
                "error": f"Query exceeds maximum length of {config.max_query_length}.",
                "request_id": get_request_id(),
            },
        )

    try:
        response = await _run_agent(
            agent,
            query,
        )
    except Exception as exc:
        log_event(
            "agent_execution_failed",
            exception_type=type(exc).__name__,
        )
        error = ErrorResponse(
            error="Agent execution failed.",
            request_id=get_request_id(),
        )

        return JSONResponse(
            status_code=500,
            content=error.model_dump(),
        )

    return serialize_agent_response(
        response,
        query_id=payload.query_id,
    )