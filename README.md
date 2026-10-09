## Agentic RAG and Controlled Tool Execution

EnterpriseIQ includes a single-agent execution layer on top of the
citation-backed grounded RAG pipeline.

The agent routes enterprise requests through three explicit paths:

- **DIRECT_RAG** — answer using grounded enterprise evidence
- **TOOL_CALL** — invoke an explicitly allowlisted enterprise tool
- **ABSTAIN** — safely reject unsupported or unsafe execution

The tool layer provides read-only enterprise operations for knowledge
search, exact document retrieval, document discovery, and department
discovery.

Tool execution uses defense-in-depth controls:

- explicit tool allowlisting
- typed argument validation
- read-only enforcement by default
- destructive-request detection
- controlled handler execution
- sanitized backend failures
- immutable execution contracts
- structured audit traces

Unsafe shell, Python, file-mutation, and destructive database requests are
not exposed as agent capabilities.

The deterministic router supports reproducible local execution and
security testing. A future semantic or LLM router can implement the same
routing contract without bypassing the tool registry or safe execution
boundary.

See [`docs/agentic-rag-architecture.md`](docs/agentic-rag-architecture.md)
for the complete agentic RAG architecture.

## FastAPI Service, Demo, and Observability

EnterpriseIQ exposes the agentic RAG runtime through a production-oriented
FastAPI boundary with strict request/response contracts, correlation IDs,
structured logging, operational metrics, and sanitized failures.

Key service surfaces:

- `/v1/health` and `/v1/ready` for service health
- `/v1/agent` for grounded agent execution
- `/v1/metrics` for Prometheus-compatible request/error/latency metrics
- `/demo` for a lightweight recruiter-facing browser interface
- `/docs` for interactive OpenAPI documentation when enabled

The HTTP layer applies configurable query limits, offloads synchronous agent
execution from the async event loop, adds defensive browser security headers,
and does not enable permissive CORS by default. Logs and metrics avoid query
payloads and use request correlation IDs for operational tracing.

Run locally:

```bash
uvicorn enterpriseiq.api.app:app --reload
```

See [`docs/service-architecture.md`](docs/service-architecture.md) for the
Day-06 service architecture and operational boundaries.


## Production Engineering and Deployment

EnterpriseIQ includes a non-root Docker runtime, environment-backed production
configuration, container health checks, Railway deployment metadata, and CI
container smoke tests.

Production defaults intentionally disable interactive API documentation.
Runtime configuration is documented in `.env.example`, while deployment and
release procedures are described in
[`docs/deployment.md`](docs/deployment.md).

Build locally:

```bash
docker build -t enterpriseiq:0.7.0 .
docker run --rm -p 8000:8000 enterpriseiq:0.7.0
```

The CI pipeline gates releases on linting, the full automated test suite,
package compilation, container construction, live health probing, and the
production documentation policy.
