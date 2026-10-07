"""EnterpriseIQ retrieval evaluation interfaces."""

from enterpriseiq.evaluation.benchmark import (
    BenchmarkError,
    BenchmarkQuery,
    RelevanceJudgment,
    RetrievalBenchmark,
    load_benchmark,
    validate_benchmark_documents,
)
from enterpriseiq.evaluation.metrics import (
    MetricError,
    QueryMetrics,
    dcg_at_k,
    evaluate_query,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "BenchmarkError",
    "BenchmarkQuery",
    "MetricError",
    "QueryMetrics",
    "RelevanceJudgment",
    "RetrievalBenchmark",
    "dcg_at_k",
    "evaluate_query",
    "hit_at_k",
    "load_benchmark",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "validate_benchmark_documents",
]
