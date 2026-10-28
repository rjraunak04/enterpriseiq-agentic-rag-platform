# EnterpriseIQ Service Architecture

Day 06 turns the retrieval and agent layers into an observable HTTP service.

## Request flow

```text
Browser / API Client
        |
        v
FastAPI service
  |-- /v1/health + /v1/ready
  |-- /v1/agent
  |-- /v1/metrics
  |-- /demo
        |
        v
AgenticRAGOrchestrator
  |-- DIRECT_RAG -> grounded RAG -> citations
  |-- TOOL_CALL  -> allowlisted read-only tools
  '-- ABSTAIN    -> safe refusal
```

## Production-oriented controls

- strict Pydantic transport contracts and sanitized error responses
- correlation IDs propagated through responses and structured JSON logs
- payload-safe logging that avoids recording query text
- thread-safe request, error, and latency metrics in Prometheus text format
- configurable API prefix and query-length boundary
- synchronous agent work offloaded from the async event loop
- CSP, frame, MIME-sniffing, and referrer security headers
- no permissive CORS policy by default
- dependency-free recruiter demo at `/demo`

The service container keeps the agent runtime injectable, so tests do not
depend on external model providers. Each application instance owns its own
metrics registry, preventing state leakage between service instances.

## Operational endpoints

| Endpoint | Purpose |
| --- | --- |
| `/` | service metadata |
| `/v1/health` | liveness |
| `/v1/ready` | dependency readiness |
| `/v1/agent` | grounded agent execution |
| `/v1/metrics` | Prometheus-compatible operational metrics |
| `/demo` | browser demo |

The API prefix is configurable; versioned endpoints follow that configured
prefix while the recruiter demo remains at `/demo`.

## Observability boundary

Metrics use bounded service metadata rather than request IDs or query text.
Structured logs include correlation IDs, method, path, status, latency, and
sanitized exception types. Agent failures return public error contracts
without exposing internal exception messages.

## Local service

Install the project with development dependencies and run:

```bash
uvicorn enterpriseiq.api.app:app --reload
```

Then open `/demo` for the browser interface, `/docs` for OpenAPI docs,
or `/v1/metrics` for operational metrics.
