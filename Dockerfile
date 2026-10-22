FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN addgroup --system enterpriseiq \
    && adduser --system --ingroup enterpriseiq enterpriseiq

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/raw ./data/raw
RUN python -m pip install --upgrade pip \
    && python -m pip install .

USER enterpriseiq
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=2)"

CMD ["sh", "-c", "uvicorn enterpriseiq.api.asgi:app --host 0.0.0.0 --port ${PORT:-8000}"]
