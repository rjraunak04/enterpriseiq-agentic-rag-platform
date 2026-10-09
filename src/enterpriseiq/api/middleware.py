"""Request correlation and structured service logging for EnterpriseIQ."""

from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from enterpriseiq.api.config import APIConfig

_request_id: ContextVar[str | None] = ContextVar("enterpriseiq_request_id", default=None)
logger = logging.getLogger("enterpriseiq.api")


def get_request_id() -> str | None:
    """Return the correlation ID for the active request, when available."""
    return _request_id.get()


def _clean_request_id(value: str | None) -> str:
    """Return a safe request ID or generate a new one."""
    if value is not None:
        candidate = value.strip()
        if candidate and len(candidate) <= 128 and all(
            char.isalnum() or char in "-_.:" for char in candidate
        ):
            return candidate
    return str(uuid4())


def log_event(event: str, **fields: object) -> None:
    """Emit one structured JSON service event without request payloads."""
    payload = {"event": event, "request_id": get_request_id(), **fields}
    logger.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach correlation IDs and emit payload-safe request logs."""

    def __init__(self, app: object, *, config: APIConfig) -> None:
        super().__init__(app)
        self.config = config

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = _clean_request_id(
            request.headers.get(self.config.request_id_header)
        )
        token = _request_id.set(request_id)
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log_event("request_failed", method=request.method, path=request.url.path)
            raise
        else:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            response.headers[self.config.request_id_header] = request_id
            log_event(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            return response
        finally:
            _request_id.reset(token)
