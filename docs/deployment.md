# Production Deployment

EnterpriseIQ ships as a non-root Docker image with an explicit production ASGI
entry point and health check.

## Build and run

```bash
docker build -t enterpriseiq:0.7.0 .
docker run --rm -p 8000:8000 enterpriseiq:0.7.0
```

Verify `/v1/health`, `/v1/ready`, and `/v1/metrics`. Production defaults
disable OpenAPI documentation. Set `ENTERPRISEIQ_EXPOSE_DOCS=true` only when
interactive documentation is intentionally required.

Configuration is environment-backed; see `.env.example`. Do not commit
provider credentials or enterprise secrets.

## Railway

The repository includes `railway.toml` and a Dockerfile. Railway can build the
container directly and uses `/v1/health` as its deployment health check. The
platform-provided `PORT` is consumed by the container start command.

## Release gates

GitHub Actions runs Ruff, the complete pytest suite, package compilation, a
Docker build, a live container health probe, and a production docs-policy
check. A release should not be promoted when any gate fails.

## Runtime boundaries

The HTTP service provides deployment infrastructure and dependency injection.
External LLM, embedding, or enterprise data credentials must be supplied by a
deployment-specific composition layer; secrets are never embedded in the
container image.
