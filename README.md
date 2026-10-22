# EnterpriseIQ — Agentic RAG Platform

A production-oriented enterprise knowledge assistant built with Python, FastAPI, retrieval-augmented generation (RAG), citation validation, and controlled tool execution.

EnterpriseIQ is designed around a simple rule: **answers must be grounded in retrieved enterprise evidence, and tools must operate inside an explicit safety boundary.**

## What it does

- **Grounded knowledge retrieval:** loads a curated policy corpus, chunks documents, embeds them, and retrieves ranked evidence.
- **Hybrid retrieval components:** provides semantic and lexical retrieval with reciprocal-rank fusion for combining rankings.
- **Citation-backed answers:** attaches source IDs to evidence and validates citations before returning a grounded response.
- **Controlled agent routing:** separates direct RAG, allowlisted tool calls, and abstention paths.
- **Read-only enterprise tools:** supports knowledge search, exact document retrieval, document discovery, and department discovery.
- **Integrity-checked vector index:** validates persisted index bytes against a SHA-256 manifest before loading.
- **Service operations:** exposes health/readiness endpoints, Prometheus-compatible metrics, request correlation IDs, structured logging, and sanitized errors.
- **Container deployment:** includes a non-root Docker image, container health check, production documentation defaults, and Railway configuration.

## Architecture

```text
Client / Demo UI
      |
      v
FastAPI boundary
(request validation, request IDs, security headers, metrics)
      |
      v
Agent router
  |            |             |
  v            v             v
Direct RAG   Safe tool     Abstain
  |          registry        |
  v            |             |
Retrieve       v             |
evidence   Typed validation  |
  |            |             |
  v            v             |
Grounded    Read-only        |
context     handlers         |
  |            |             |
  v            +-------------+
Answer generation
  |
  v
Citation validation
  |
  v
Structured response + audit metadata
```

Detailed design: [Agentic RAG architecture](docs/agentic-rag-architecture.md) · [Service architecture](docs/service-architecture.md)

## Repository map

| Path | Purpose |
|---|---|
| `src/enterpriseiq/agent/` | Agent routing and controlled execution contracts |
| `src/enterpriseiq/rag/` | Evidence context, prompt policy, generation contract, and citation validation |
| `src/enterpriseiq/retrieval/` | Semantic/lexical retrieval, hybrid fusion, and persistent vector index |
| `src/enterpriseiq/ingestion/` | Enterprise document loading and chunking |
| `src/enterpriseiq/api/` | FastAPI service, request models, health, metrics, and demo UI |
| `data/raw/` | Sample enterprise policy corpus |
| `tests/` | Unit, integration, security, API, and retrieval benchmark tests |
| `docs/` | Architecture, operations, and deployment documentation |

## Run locally

Requires Python 3.11 or newer.

### Install

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Start the API

```bash
uvicorn enterpriseiq.api.asgi:app --host 127.0.0.1 --port 8000
```

Production configuration is environment-backed. See [.env.example](.env.example). Interactive OpenAPI documentation is disabled by default; set `ENTERPRISEIQ_EXPOSE_DOCS=true` when it is intentionally needed in a local or controlled environment.

## API quickstart

### Readiness

```bash
curl --fail http://127.0.0.1:8000/v1/ready
```

### Ask a policy question

```bash
curl --fail http://127.0.0.1:8000/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the expense reimbursement policy?"}'
```

PowerShell equivalent:

```powershell
$body = @{ query = "What is the expense reimbursement policy?" } |
    ConvertTo-Json -Compress

Invoke-RestMethod -Uri "http://localhost:8000/v1/agent" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Useful service endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /v1/health` | Liveness/health |
| `GET /v1/ready` | Readiness probe |
| `POST /v1/agent` | Execute an agent query |
| `GET /v1/metrics` | Prometheus-compatible operational metrics |
| `GET /demo` | Lightweight browser demo |
| `GET /docs` | OpenAPI UI, when explicitly enabled |

## Run tests and quality checks

```bash
ruff check .
pytest -q
python -m compileall -q src
```

The GitHub Actions workflow runs linting, the full test suite, package compilation, a Docker build, a live readiness probe, and a production documentation-policy check.

## Build and run with Docker

```bash
docker build -t enterpriseiq:local .
docker run --rm -p 8000:8000 enterpriseiq:local
```

Then verify:

```bash
curl --fail http://127.0.0.1:8000/v1/ready
```

The image runs as a non-root user and includes a container health check. Deployment instructions, including Railway configuration, are in [docs/deployment.md](docs/deployment.md).

## Grounding and safety boundaries

- Retrieved document content is treated as evidence, not as instructions to override system policy.
- Unsupported questions should abstain rather than fabricate a policy answer.
- Only registered tools with typed parameters are exposed to the agent.
- Destructive shell, Python, filesystem-mutation, and database operations are not exposed as agent tools.
- Citation validation checks that returned source IDs belong to the retrieved evidence.
- The persisted vector index is verified against its SHA-256 manifest on load.
- Secrets and provider credentials must be supplied through deployment configuration; they should not be committed to the repository.

## Current implementation scope

The default answer generator is **deterministic and dependency-free**. It is useful for reproducible tests, local demonstrations, and verifying evidence/citation flow; it preserves the selected evidence block rather than producing semantic LLM-style summaries. A production LLM provider can be added behind the existing `AnswerGenerator` contract, while retaining citation validation and abstention controls.

The default runtime uses a deterministic embedding implementation for reproducible local execution. Replace it with a deployment-specific embedding provider and evaluated index before relying on the service for production enterprise search at scale.

## Deployment

- [Deployment guide](docs/deployment.md)
- [Agentic RAG architecture](docs/agentic-rag-architecture.md)
- [Service architecture](docs/service-architecture.md)
- [Environment template](.env.example)

## License

No license file is currently defined in this repository. Contact the repository owner before reusing or redistributing this project.
