"""Production ASGI entry point for EnterpriseIQ."""

from enterpriseiq.api.app import create_app
from enterpriseiq.api.runtime import config_from_env

app = create_app(config=config_from_env())
