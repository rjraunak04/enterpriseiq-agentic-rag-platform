# EnterpriseIQ Agentic RAG Architecture

## Overview

EnterpriseIQ extends a grounded enterprise retrieval-augmented generation
pipeline with a controlled agent execution layer.

The agent does not receive unrestricted execution access. Every tool call
must pass through deterministic routing, an explicit tool allowlist, typed
argument validation, and a controlled execution boundary.

The design prioritizes:

- grounded enterprise answers
- explicit tool boundaries
- read-only execution by default
- deterministic audit traces
- citation-backed RAG
- failure isolation
- secret-safe error handling
- testable agent behavior

The current implementation intentionally uses a single-agent architecture.
Multi-agent orchestration is deferred until the single-agent execution
boundary is fully observable and production hardened.

---

## High-Level Architecture

```text
                         Enterprise Query
                                |
                                v
                         +--------------+
                         | Agent Query  |
                         +--------------+
                                |
                                v
                    +-----------------------+
                    | Agent Routing Policy  |
                    +-----------------------+
                       /        |         \
                      /         |          \
                     v          v           v
              DIRECT_RAG    TOOL_CALL     ABSTAIN
                  |              |             |
                  v              v             |
        +----------------+  +-------------+    |
        | Grounded RAG   |  | Tool        |    |
        | Pipeline       |  | Registry    |    |
        +----------------+  +-------------+    |
                  |              |             |
                  |              v             |
                  |       +---------------+     |
                  |       | Typed Input   |     |
                  |       | Validation    |     |
                  |       +---------------+     |
                  |              |             |
                  |              v             |
                  |       +---------------+     |
                  |       | Safe Tool     |     |
                  |       | Executor      |     |
                  |       +---------------+     |
                  |              |             |
                  |              v             |
                  |       +---------------+     |
                  |       | Enterprise    |     |
                  |       | Read Tools    |     |
                  |       +---------------+     |
                  |              |             |
                  +--------------+-------------+
                                 |
                                 v
                        +----------------+
                        | Agent Response |
                        +----------------+
                                 |
                                 v
                        +----------------+
                        | Audit Trace    |
                        +----------------+