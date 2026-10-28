# Deployment Guide

EnterpriseIQ is packaged as a Docker web service and currently has a Render deployment. This guide covers local container verification, hosted endpoints, configuration, and release checks.

## Live service

- **Service:** [EnterpriseIQ on Render](https://enterpriseiq-agentic-rag-platform.onrender.com/)
- **Browser demo:** [`/demo`](https://enterpriseiq-agentic-rag-platform.onrender.com/demo)
- **Readiness:** [`/v1/ready`](https://enterpriseiq-agentic-rag-platform.onrender.com/v1/ready)
- **Liveness:** [`/v1/health`](https://enterpriseiq-agentic-rag-platform.onrender.com/v1/health)
- **Metrics:** [`/v1/metrics`](https://enterpriseiq-agentic-rag-platform.onrender.com/v1/metrics)
- **Render dashboard:** [Service dashboard](https://dashboard.render.com/web/srv-dbgsqsfb9u1s7396aml0)

Free-tier services can sleep when idle and may have a cold start. Treat hosted availability as a platform property, not an application guarantee.

## Build and run with Docker

From the repository root:

```bash
docker build -t enterpriseiq:local .
docker run --rm -p 8000:8000 enterpriseiq:local
```

Verify the running container:

```bash
curl --fail http://127.0.0.1:8000/v1/health
curl --fail http://127.0.0.1:8000/v1/ready
curl --fail http://127.0.0.1:8000/v1/metrics
```

Ask a sample question:

```bash
curl --fail http://127.0.0.1:8000/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the remote work policy?"}'
```

The container runs as a non-root user and defines a readiness-based Docker health check. The container start command respects the platform-provided `PORT` variable.

## Render configuration

The deployed service uses the GitHub repository `rjraunak04/enterpriseiq-agentic-rag-platform`, branch `main`, and the repository Dockerfile.

When recreating the service, configure:

- **Runtime:** Docker
- **Dockerfile:** `./Dockerfile`
- **Docker context:** repository root
- **Health check path:** `/v1/ready`
- **Port:** use the platform-provided `PORT`; the Docker start command defaults to 8000 locally
- **Environment:** set only non-secret service configuration in the dashboard; never commit credentials

The checked-in `.env.example` documents supported default settings. Interactive OpenAPI docs are disabled by default. Set `ENTERPRISEIQ_EXPOSE_DOCS=true` only when deliberately required in a controlled environment.

## CI/CD release gates

The GitHub Actions workflow at [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs:

1. Ruff lint checks, the complete pytest suite, and package compilation.
2. A Docker build and runtime smoke test, including readiness, production documentation policy, and an agent request.
3. Publishing the verified image to GitHub Container Registry on push events.

Inspect the workflow status before considering a code change release-ready:
[GitHub Actions](https://github.com/rjraunak04/enterpriseiq-agentic-rag-platform/actions/workflows/ci.yml).

## Configuration and secrets

Configuration is environment-backed. Use `.env.example` as the reference for local development and the hosting dashboard for deployment-specific settings. Do not store API keys, tokens, private documents, or production credentials in Git, Docker layers, logs, or issue threads.

## Runtime and data boundaries

The current default answer generator and embedding implementation are deterministic and intended for reproducible demonstration and testing. Before using this platform with real enterprise content, configure and evaluate the appropriate model/embedding providers, rebuild the index as needed, and implement identity-aware document authorization and governance. The bundled policy corpus is illustrative, not authoritative company policy.

## Alternate deployment configuration

A `railway.toml` file is retained for compatibility with an alternate Docker deployment path. The live deployment documented above is Render. Keep platform-specific settings explicit rather than assuming configuration files are interchangeable.
