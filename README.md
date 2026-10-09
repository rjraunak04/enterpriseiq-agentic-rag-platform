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