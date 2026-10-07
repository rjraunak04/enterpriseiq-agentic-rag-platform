"""Versioned retrieval benchmark contracts and loading utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


class BenchmarkError(ValueError):
    """Raised when a retrieval benchmark is invalid."""


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """Document-level relevance judgment for one benchmark query."""

    document_id: str
    relevance: int

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise BenchmarkError("document_id must be a non-empty string")

        if (
            isinstance(self.relevance, bool)
            or not isinstance(self.relevance, int)
            or self.relevance <= 0
        ):
            raise BenchmarkError("relevance must be a positive integer")


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    """One enterprise retrieval evaluation query."""

    query_id: str
    query: str
    category: str
    judgments: tuple[RelevanceJudgment, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, value in (
            ("query_id", self.query_id),
            ("query", self.query),
            ("category", self.category),
        ):
            if not isinstance(value, str) or not value.strip():
                raise BenchmarkError(f"{name} must be a non-empty string")

        if not self.judgments:
            raise BenchmarkError(
                "benchmark query must contain at least one relevance judgment"
            )

        document_ids = [
            judgment.document_id
            for judgment in self.judgments
        ]
        if len(document_ids) != len(set(document_ids)):
            raise BenchmarkError(
                f"duplicate relevance judgment in query {self.query_id}"
            )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def relevant_document_ids(self) -> frozenset[str]:
        """Return all documents considered relevant to this query."""

        return frozenset(
            judgment.document_id
            for judgment in self.judgments
        )


@dataclass(frozen=True, slots=True)
class RetrievalBenchmark:
    """Versioned collection of enterprise retrieval queries."""

    name: str
    version: str
    queries: tuple[BenchmarkQuery, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise BenchmarkError("benchmark name must be a non-empty string")

        if not isinstance(self.version, str) or not self.version.strip():
            raise BenchmarkError(
                "benchmark version must be a non-empty string"
            )

        if not self.queries:
            raise BenchmarkError(
                "benchmark must contain at least one query"
            )

        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise BenchmarkError("benchmark query IDs must be unique")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @property
    def query_count(self) -> int:
        return len(self.queries)


def _parse_judgments(
    raw: object,
    *,
    query_id: str,
) -> tuple[RelevanceJudgment, ...]:
    if not isinstance(raw, list) or not raw:
        raise BenchmarkError(
            f"query {query_id} must define non-empty judgments"
        )

    judgments: list[RelevanceJudgment] = []

    for item in raw:
        if not isinstance(item, dict):
            raise BenchmarkError(
                f"invalid relevance judgment in query {query_id}"
            )

        judgments.append(
            RelevanceJudgment(
                document_id=item.get("document_id", ""),
                relevance=item.get("relevance", 0),
            )
        )

    return tuple(judgments)


def load_benchmark(path: Path) -> RetrievalBenchmark:
    """Load and validate a retrieval benchmark JSON artifact."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(
            f"invalid benchmark JSON: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise BenchmarkError(
            "benchmark root must be a JSON object"
        )

    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise BenchmarkError(
            "benchmark must define a non-empty queries list"
        )

    queries: list[BenchmarkQuery] = []

    for raw_query in raw_queries:
        if not isinstance(raw_query, dict):
            raise BenchmarkError(
                "each benchmark query must be a JSON object"
            )

        query_id = raw_query.get("query_id", "")

        queries.append(
            BenchmarkQuery(
                query_id=query_id,
                query=raw_query.get("query", ""),
                category=raw_query.get("category", ""),
                judgments=_parse_judgments(
                    raw_query.get("judgments"),
                    query_id=query_id,
                ),
                metadata=raw_query.get("metadata", {}),
            )
        )

    return RetrievalBenchmark(
        name=payload.get("name", ""),
        version=payload.get("version", ""),
        queries=tuple(queries),
        metadata=payload.get("metadata", {}),
    )


def validate_benchmark_documents(
    benchmark: RetrievalBenchmark,
    available_document_ids: Sequence[str],
) -> None:
    """Ensure every relevance judgment exists in the corpus."""

    available = set(available_document_ids)

    missing = sorted(
        {
            judgment.document_id
            for query in benchmark.queries
            for judgment in query.judgments
            if judgment.document_id not in available
        }
    )

    if missing:
        raise BenchmarkError(
            "benchmark references unknown document IDs: "
            + ", ".join(missing)
        )
