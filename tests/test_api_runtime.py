"""Tests for environment-backed production configuration."""

import pytest

from enterpriseiq.api.config import APIConfigError
from enterpriseiq.api.runtime import config_from_env


def test_production_defaults_disable_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ENTERPRISEIQ_ENVIRONMENT",
        "ENTERPRISEIQ_EXPOSE_DOCS",
        "ENTERPRISEIQ_MAX_QUERY_LENGTH",
    ):
        monkeypatch.delenv(name, raising=False)

    config = config_from_env()

    assert config.environment == "production"
    assert config.expose_docs is False
    assert config.max_query_length == 4000
    assert config.service_version == "0.7.0"


def test_runtime_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTERPRISEIQ_ENVIRONMENT", "staging")
    monkeypatch.setenv("ENTERPRISEIQ_EXPOSE_DOCS", "true")
    monkeypatch.setenv("ENTERPRISEIQ_MAX_QUERY_LENGTH", "2048")

    config = config_from_env()

    assert config.environment == "staging"
    assert config.expose_docs is True
    assert config.max_query_length == 2048


def test_runtime_config_rejects_invalid_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTERPRISEIQ_EXPOSE_DOCS", "sometimes")

    with pytest.raises(APIConfigError, match="must be a boolean"):
        config_from_env()
