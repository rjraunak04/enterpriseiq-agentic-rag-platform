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
from enterpriseiq.evaluation.runner import (
    AggregateMetrics,
    EvaluationError,
    EvaluationRetriever,
    QueryEvaluation,
    RetrieverEvaluation,
    compare_retrievers,
    evaluate_retriever,
    write_evaluation_report,
)

__all__ = [
    "AggregateMetrics",
    "BenchmarkError",
    "BenchmarkQuery",
    "EvaluationError",
    "EvaluationRetriever",
    "MetricError",
    "QueryEvaluation",
    "QueryMetrics",
    "RelevanceJudgment",
    "RetrievalBenchmark",
    "RetrieverEvaluation",
    "compare_retrievers",
    "dcg_at_k",
    "evaluate_query",
    "evaluate_retriever",
    "hit_at_k",
    "load_benchmark",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "validate_benchmark_documents",
    "write_evaluation_report",
]
