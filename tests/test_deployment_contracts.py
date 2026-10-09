"""Static deployment-contract tests for the production image."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_non_root_runtime() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert "USER enterpriseiq" in dockerfile
    assert "uvicorn enterpriseiq.api.app:app" in dockerfile


def test_dockerfile_has_service_healthcheck() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "HEALTHCHECK" in dockerfile
    assert "/v1/health" in dockerfile


def test_dockerignore_excludes_local_and_test_artifacts() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".git" in ignored
    assert ".venv" in ignored
    assert "tests" in ignored
