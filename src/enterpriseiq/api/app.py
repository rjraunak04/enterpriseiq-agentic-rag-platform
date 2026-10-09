"""FastAPI application foundation for EnterpriseIQ."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from enterpriseiq.api.agent import router as agent_router
from enterpriseiq.api.config import APIConfig
from enterpriseiq.api.dependencies import ServiceContainer
from enterpriseiq.api.health import router as health_router
from enterpriseiq.api.models import ErrorResponse


def _build_lifespan(
    container: ServiceContainer,
):
    """Create the application lifespan handler."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service_container = container
        yield

    return lifespan


def get_service_container(request: Request) -> ServiceContainer:
    """Return the service dependency container attached to the application."""

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


def create_app(
    *,
    config: APIConfig | None = None,
    container: ServiceContainer | None = None,
) -> FastAPI:
    """Create a configured EnterpriseIQ FastAPI application."""

    resolved_config = config or APIConfig()
    resolved_container = container or ServiceContainer()

    docs_url = "/docs" if resolved_config.expose_docs else None
    redoc_url = "/redoc" if resolved_config.expose_docs else None
    openapi_url = "/openapi.json" if resolved_config.expose_docs else None

    app = FastAPI(
        title=resolved_config.service_name,
        version=resolved_config.service_version,
        description=(
            "Enterprise-grade agentic RAG service for grounded "
            "knowledge retrieval and controlled tool execution."
        ),
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=_build_lifespan(resolved_container),
    )

    app.state.api_config = resolved_config
    app.state.service_container = resolved_container

    # Service health and readiness endpoints.
    app.include_router(
        health_router,
        prefix=resolved_config.api_prefix,
    )

    # Agent execution endpoint.
    app.include_router(
        agent_router,
        prefix=resolved_config.api_prefix,
    )

    @app.get(
        "/",
        tags=["service"],
        summary="Service metadata",
    )
    async def service_root() -> dict[str, Any]:
        """Return public service metadata."""

        return {
            "service": resolved_config.service_name,
            "version": resolved_config.service_version,
            "environment": resolved_config.environment,
            "api_prefix": resolved_config.api_prefix,
            "docs_enabled": resolved_config.expose_docs,
        }

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Return a sanitized response for unexpected service failures."""

        del request, exc

        payload = ErrorResponse(
            error="Internal service error."
        )

        return JSONResponse(
            status_code=500,
            content=payload.model_dump(),
        )

    return app


app = create_app()