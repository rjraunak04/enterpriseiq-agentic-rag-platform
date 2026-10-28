# EnterpriseIQ — Agentic RAG Platform

> **A deployable enterprise knowledge assistant built around grounded retrieval, citation checks, and explicitly bounded agent tools.**

[![CI](https://github.com/rjraunak04/enterpriseiq-agentic-rag-platform/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rjraunak04/enterpriseiq-agentic-rag-platform/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Deployed on Render](https://img.shields.io/badge/Deployment-Render-46E3B7?logo=render&logoColor=111)](https://enterpriseiq-agentic-rag-platform.onrender.com)

**Live demo:** [enterpriseiq-agentic-rag-platform.onrender.com](https://enterpriseiq-agentic-rag-platform.onrender.com) · **Browser demo:** [Open /demo](https://enterpriseiq-agentic-rag-platform.onrender.com/demo) · **Readiness:** [Check /v1/ready](https://enterpriseiq-agentic-rag-platform.onrender.com/v1/ready)

---

## The problem

Enterprise teams often have policies and operating procedures spread across many documents. A plain chatbot can return a confident answer without showing which document supports it, while an unrestricted agent can introduce unnecessary execution risk.

**EnterpriseIQ demonstrates a more controlled approach:** retrieve relevant document evidence, keep agent actions behind an explicit allowlist, validate citations against retrieved sources, and abstain when the system cannot support an answer.

## What it does

- **Ingests document knowledge** — loads a curated Markdown policy corpus, normalizes content, and creates configurable overlapping chunks with metadata.
- **Retrieves evidence** — supports lexical and semantic retrieval, hybrid reciprocal-rank fusion, reranking, and a persisted vector index.
- **Keeps answers grounded** — builds context from retrieved passages, checks returned source IDs against the retrieved evidence, and provides a safe abstention path.
- **Routes agent requests explicitly** — separates direct RAG, approved tool calls, and abstention instead of granting the agent unrestricted execution.
- **Restricts tool execution** — exposes typed, allowlisted, read-only knowledge tools rather than arbitrary shell, Python, filesystem-write, or database access.
- **Checks index integrity** — validates persisted vector-index data against a SHA-256 manifest before loading.
- **Exposes an observable API** — FastAPI endpoints, request correlation IDs, structured and sanitized logs, security headers, readiness checks, and Prometheus-compatible metrics.
- **Ships in a container** — Docker image runs as a non-root user and includes a readiness health check.
- **Tests the whole path** — automated checks cover retrieval, grounding, citation validation, agent routing, security boundaries, API contracts, and container smoke behavior.

## Try the deployed app

The service is deployed on Render. Start with these links:

| Link | What to look for |
|---|---|
| [Home / service metadata](https://enterpriseiq-agentic-rag-platform.onrender.com/) | Service response |
| [Interactive demo](https://enterpriseiq-agentic-rag-platform.onrender.com/demo) | Submit a sample policy question |
| [Readiness probe](https://enterpriseiq-agentic-rag-platform.onrender.com/v1/ready) | Whether the service is ready to handle requests |
| [Liveness probe](https://enterpriseiq-agentic-rag-platform.onrender.com/v1/health) | Basic service health |
| [Metrics](https://enterpriseiq-agentic-rag-platform.onrender.com/v1/metrics) | Prometheus-compatible metrics endpoint |

The hosted instance may sleep or take time to wake on a free hosting plan. OpenAPI docs are disabled by default in production.

### Example question

> What is the expense reimbursement policy?

The API returns a structured response from the current implementation. Try questions grounded in the sample corpus (HR, Finance, IT, Security, Operations, and Product); unsupported questions should not be treated as verified company policy.

## How a request flows

```text
User / Demo UI
      |
      v
FastAPI boundary
(validation, request IDs, security headers, metrics)
      |
      v
Agent routing policy
   |             |             |
   v             v             v
Direct RAG    Approved tool   Abstain
   |             call           |
   v               |            |
Hybrid retrieval    v            |
   |           Typed validation |
   v               |            |
Retrieved evidence v            |
   |          Read-only handler |
   +---------------+------------+
                   |
                   v
          Grounded response path
                   |
                   v
          Citation/source checks
                   |
                   v
       Response + operational trace
```

**Key design decision:** retrieval evidence and tool execution are separate paths. A tool call must be routed to a registered tool and pass typed argument validation; the model is not given arbitrary machine access.

More detail: [Agentic RAG architecture](docs/agentic-rag-architecture.md) · [Service architecture](docs/service-architecture.md) · [Ingestion architecture](docs/architecture/ingestion-architecture.md) · [Vector retrieval architecture](docs/architecture/vector-retrieval-architecture.md) · [Grounded RAG and citations](docs/architecture/grounded-rag-citations.md)

## Technology stack

| Area | Implementation |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI, Pydantic, Uvicorn |
| Ingestion | Markdown loading, cleaning, metadata-aware chunking |
| Retrieval | Lexical + semantic retrieval, hybrid rank fusion, reranking |
| RAG / agent | Typed contracts, explicit routing, evidence context, citation validation, abstention |
| Observability | Structured logs, request IDs, health/readiness endpoints, Prometheus text metrics |
| Quality | pytest, Ruff, compile checks, regression and security tests |
| Delivery | Docker, GitHub Actions, GitHub Container Registry, Render |

## Repository guide

```text
.
├── .github/workflows/ci.yml       # Quality gates, container smoke test, image publishing
├── data/
│   ├── raw/                       # Curated sample enterprise policy documents
│   ├── evaluation/                # Retrieval evaluation gold set
│   └── processed/                 # Reproducible, tracked index artifacts
├── docs/
│   ├── architecture/              # Ingestion, retrieval, grounding, evaluation design
│   ├── decisions/                 # Architecture decision records (reserved)
│   ├── agentic-rag-architecture.md
│   ├── service-architecture.md
│   └── deployment.md
├── src/enterpriseiq/
│   ├── agent/                     # Routing, orchestrator, safe tool execution
│   ├── api/                       # FastAPI app, endpoints, middleware, demo
│   ├── domain/                    # Core document contracts
│   ├── embeddings/                # Embedding contracts, providers, pipeline
│   ├── evaluation/                # Retrieval benchmark and evaluation metrics
│   ├── ingestion/                 # Loading, cleaning, chunking, pipeline
│   ├── rag/                       # Context, generation contract, citations, prompts
│   └── retrieval/                 # Lexical, semantic, hybrid search, index, reranking
├── tests/                         # Unit, integration, regression, API and security tests
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Run locally

### 1. Requirements

- Python 3.11 or newer
- Git
- Docker only if you want to run the container

### 2. Install

**Windows PowerShell**

```powershell
git clone https://github.com/rjraunak04/enterpriseiq-agentic-rag-platform.git
cd enterpriseiq-agentic-rag-platform
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks environment activation, you can run the environment's Python directly with `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` and `.\.venv\Scripts\python.exe -m uvicorn enterpriseiq.api.asgi:app --host 127.0.0.1 --port 8000`.

**Linux / macOS**

```bash
git clone https://github.com/rjraunak04/enterpriseiq-agentic-rag-platform.git
cd enterpriseiq-agentic-rag-platform
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 3. Configure and start

Copy `.env.example` to `.env` if you want to customize local settings. The checked-in example contains non-secret defaults; do not commit real credentials.

```bash
uvicorn enterpriseiq.api.asgi:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/demo](http://127.0.0.1:8000/demo). OpenAPI docs are disabled by default; set `ENTERPRISEIQ_EXPOSE_DOCS=true` only when you intentionally need them in a controlled environment.

### 4. Ask a question

**Linux / macOS**

```bash
curl --fail http://127.0.0.1:8000/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the expense reimbursement policy?"}'
```

**PowerShell**

```powershell
$body = @{ query = "What is the expense reimbursement policy?" } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/agent" -Method Post -ContentType "application/json" -Body $body
```

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service metadata |
| GET | `/v1/health` | Liveness |
| GET | `/v1/ready` | Readiness |
| POST | `/v1/agent` | Agent query |
| GET | `/v1/metrics` | Prometheus-compatible metrics |
| GET | `/demo` | Browser demo |
| GET | `/docs` | OpenAPI UI, when explicitly enabled |

The default API prefix is `/v1` and can be configured. See [.env.example](.env.example).

## Quality checks

Run these from the repository root:

```bash
ruff check .
pytest -q
python -m compileall -q src
```

The GitHub Actions pipeline runs three gates:

1. **Python quality:** Ruff, full pytest suite, and package compilation.
2. **Container smoke test:** builds and starts the image, checks readiness, verifies production docs remain disabled, and sends an agent query.
3. **Image publishing:** publishes a verified container image to GitHub Container Registry on push events.

[View the GitHub Actions workflow](https://github.com/rjraunak04/enterpriseiq-agentic-rag-platform/actions/workflows/ci.yml).

## Docker

```bash
docker build -t enterpriseiq:local .
docker run --rm -p 8000:8000 enterpriseiq:local
```

Then open `http://127.0.0.1:8000/demo` or check readiness:

```bash
curl --fail http://127.0.0.1:8000/v1/ready
```

The container uses a non-root account and exposes a health check. See [deployment notes](docs/deployment.md).

## Evaluation and testing

The repository includes a retrieval gold set at `data/evaluation/retrieval_gold_v1.json` and a benchmark/evaluation implementation under `src/enterpriseiq/evaluation/`. Tests cover retrieval behavior and regressions; run the suite to reproduce the current checks. Do not interpret a test pass as proof that the system is accurate for every enterprise corpus—evaluate on representative, permission-aware documents before real deployment.

## Security and responsible use

- Treat retrieved documents as evidence, not as instructions that can override system policy.
- Do not present unsupported answers as authoritative policy.
- Validate citation IDs against the evidence returned for the current request.
- Keep tool access allowlisted, typed, and read-only.
- Do not expose arbitrary shell, Python, filesystem-write, or database operations as agent tools.
- Supply secrets via environment/deployment configuration; never commit them.
- Review document permissions and access controls before indexing real enterprise data.

## Important implementation scope

This repository is a **production-oriented engineering demonstration**, not a claim that a fully connected enterprise AI service is ready for sensitive business use without further integration.

- The default answer generator is deterministic and dependency-free. It supports reproducible demonstrations and testing of the evidence/citation path; it does **not** provide the quality of an LLM-generated semantic summary.
- The default runtime includes a deterministic embedding implementation for reproducible local execution. For real semantic search, select and configure an appropriate embedding provider, rebuild the index, and evaluate retrieval quality on representative data.
- The included policies are a sample corpus for demonstrating system behavior; they are not real company policies.
- Enterprise identity, per-user document authorization, external model-provider integration, and production data governance must be designed and verified before use with confidential data.
- Hosting availability and cold-start latency depend on the Render plan and platform state.

These boundaries are documented intentionally: a recruiter should be able to distinguish what is implemented from what would be the next production integration step.

## Documentation

- [Agentic RAG architecture](docs/agentic-rag-architecture.md)
- [Service architecture](docs/service-architecture.md)
- [Ingestion architecture](docs/architecture/ingestion-architecture.md)
- [Vector retrieval architecture](docs/architecture/vector-retrieval-architecture.md)
- [Grounded RAG and citations](docs/architecture/grounded-rag-citations.md)
- [Advanced retrieval evaluation](docs/architecture/advanced-retrieval-evaluation.md)
- [Deployment guide](docs/deployment.md)
- [Environment template](.env.example)

## License

No license file is currently defined. All rights remain with the repository owner unless a license is added; contact the owner before reuse or redistribution.
