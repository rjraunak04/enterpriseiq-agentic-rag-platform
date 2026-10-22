"""Production ASGI entry point for EnterpriseIQ."""

import logging

from enterpriseiq.api.agent_runtime import build_agent_runtime
from enterpriseiq.api.app import create_app
from enterpriseiq.api.dependencies import ServiceContainer
from enterpriseiq.api.runtime import config_from_env

logger = logging.getLogger(__name__)
container = ServiceContainer()

try:
    container.agent = build_agent_runtime()
except Exception:
    # Keep liveness available, but readiness remains 503 until startup is fixed.
    logger.exception("enterpriseiq_agent_runtime_initialization_failed")

app = create_app(config=config_from_env(), container=container)
