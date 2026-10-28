# Contributing to EnterpriseIQ

Thanks for helping improve EnterpriseIQ. Keep changes focused, reproducible, tested, and explicit about the difference between implemented behavior and future integrations.

## Development setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
# Activate the environment using your operating system's normal command.
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before opening a pull request

Run the same core checks used by CI:

```bash
ruff check .
pytest -q
python -m compileall -q src
```

For changes to the container or API boundary, also run:

```bash
docker build -t enterpriseiq:local .
docker run --rm -p 8000:8000 enterpriseiq:local
```

Verify `/v1/ready`, `/v1/agent`, and the production documentation policy before considering a container change complete.

## Design principles

- Keep retrieval evidence separate from agent instructions and tool execution.
- Route tools through the explicit registry and validate arguments with typed contracts.
- Prefer read-only, allowlisted tool behavior; do not add arbitrary shell or filesystem mutation.
- Validate citations against evidence actually retrieved for the request.
- Preserve abstention when the corpus does not support an answer.
- Keep logs sanitized and avoid recording raw user queries or secrets.
- Add regression tests for security boundaries and failure paths.
- Do not claim LLM-quality generation or production enterprise readiness unless the relevant provider, authorization, and evaluation work is implemented and tested.

## Pull request checklist

- [ ] Change has a clear, narrow purpose.
- [ ] Tests cover new behavior and relevant edge cases.
- [ ] Ruff, pytest, and compile checks pass.
- [ ] Documentation and examples match the implementation.
- [ ] No credentials, personal data, or confidential enterprise content are included.
