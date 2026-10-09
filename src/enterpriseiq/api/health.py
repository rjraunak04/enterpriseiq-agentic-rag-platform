"""Health and readiness endpoints for the EnterpriseIQ service."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from enterpriseiq.api.config import APIConfig
from enterpriseiq.api.dependencies import ServiceContainer
from enterpriseiq.api.models import HealthResponse, ReadinessResponse

router = APIRouter(
    tags=["health"],
)


def _get_config(request: Request) -> APIConfig:
    """Resolve the immutable API configuration."""

    config = getattr(
        request.app.state,
        "api_config",
        None,
    )

    if not isinstance(config, APIConfig):
        raise RuntimeError(
            "EnterpriseIQ API configuration is unavailable."
        )

    return config


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


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service liveness",
    description=(
        "Reports whether the EnterpriseIQ API process is alive. "
        "This endpoint does not require the agent runtime to be ready."
    ),
)
async def health(
    request: Request,
) -> HealthResponse:
    """Return process-level liveness information."""

    config = _get_config(request)

    return HealthResponse(
        status="ok",
        service=config.service_name,
        version=config.service_version,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        503: {
            "model": ReadinessResponse,
            "description": "Service dependencies are not ready.",
        }
    },
    summary="Service readiness",
    description=(
        "Reports whether EnterpriseIQ has the runtime dependencies "
        "required to serve agent requests."
    ),
)
async def readiness(
    request: Request,
) -> ReadinessResponse | JSONResponse:
    """Return dependency-level readiness information."""

    container = _get_container(request)

    checks = {
        "agent": container.agent_ready,
    }
    ready = all(checks.values())

    payload = ReadinessResponse(
        status="ready" if ready else "not_ready",
        ready=ready,
        checks=checks,
    )

    if not ready:
        return JSONResponse(
            status_code=503,
            content=payload.model_dump(),
        )

    return payload