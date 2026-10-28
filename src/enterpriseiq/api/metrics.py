"""Dependency-free operational metrics for the EnterpriseIQ API."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse


@dataclass(slots=True)
class MetricsRegistry:
    """Thread-safe in-process request metrics registry."""

    _requests: dict[tuple[str, str, int], int] = field(default_factory=dict)
    _errors: int = 0
    _duration_seconds: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def observe(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record one completed HTTP request."""

        key = (method.upper(), path, status_code)
        with self._lock:
            self._requests[key] = self._requests.get(key, 0) + 1
            if status_code >= 500:
                self._errors += 1
            self._duration_seconds += max(duration_seconds, 0.0)

    def render_prometheus(self) -> str:
        """Render the current registry using Prometheus text exposition."""

        with self._lock:
            requests = sorted(self._requests.items())
            errors = self._errors
            duration = self._duration_seconds

        total = sum(count for _, count in requests)
        lines = [
            "# HELP enterpriseiq_http_requests_total Total HTTP requests.",
            "# TYPE enterpriseiq_http_requests_total counter",
        ]
        for (method, path, status), count in requests:
            safe_path = path.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(
                "enterpriseiq_http_requests_total"
                f'{{method="{method}",path="{safe_path}",status="{status}"}} {count}'
            )

        lines.extend(
            [
                "# HELP enterpriseiq_http_errors_total Total HTTP 5xx responses.",
                "# TYPE enterpriseiq_http_errors_total counter",
                f"enterpriseiq_http_errors_total {errors}",
                "# HELP enterpriseiq_http_request_duration_seconds_sum "
                "Cumulative HTTP request duration.",
                "# TYPE enterpriseiq_http_request_duration_seconds_sum counter",
                f"enterpriseiq_http_request_duration_seconds_sum {duration:.6f}",
                "# HELP enterpriseiq_http_request_duration_seconds_count "
                "Observed HTTP request durations.",
                "# TYPE enterpriseiq_http_request_duration_seconds_count counter",
                f"enterpriseiq_http_request_duration_seconds_count {total}",
            ]
        )
        return "\n".join(lines) + "\n"


router = APIRouter(tags=["observability"])


def _registry(request: Request) -> MetricsRegistry:
    registry = getattr(request.app.state, "metrics_registry", None)
    if not isinstance(registry, MetricsRegistry):
        raise RuntimeError("EnterpriseIQ metrics registry is unavailable.")
    return registry


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Operational metrics",
    include_in_schema=True,
)
async def metrics(request: Request) -> PlainTextResponse:
    """Expose service metrics in Prometheus-compatible text format."""

    return PlainTextResponse(
        _registry(request).render_prometheus(),
        media_type="text/plain; version=0.0.4",
    )
