"""Build the dependency-free, production-safe EnterpriseIQ agent runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from enterpriseiq.agent.execution import SafeToolExecutor
from enterpriseiq.agent.orchestrator import AgenticRAGOrchestrator
from enterpriseiq.agent.routing import DeterministicAgentRouter
from enterpriseiq.agent.tools import (
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
)
from enterpriseiq.embeddings.config import EmbeddingConfig
from enterpriseiq.embeddings.providers import DeterministicEmbeddingProvider
from enterpriseiq.ingestion.chunker import chunk_documents
from enterpriseiq.ingestion.loader import load_document_directory
from enterpriseiq.rag.generation import DeterministicAnswerGenerator
from enterpriseiq.rag.pipeline import GroundedRAGPipeline
from enterpriseiq.retrieval.hybrid import HybridRetriever
from enterpriseiq.retrieval.lexical import BM25Retriever
from enterpriseiq.retrieval.semantic import SemanticRetriever
from enterpriseiq.retrieval.vector_index import (
    PersistentVectorIndex,
    VectorIndexRecord,
)


def build_agent_runtime(
    corpus_dir: str | Path = "data/raw",
) -> AgenticRAGOrchestrator:
    """Load the bundled policy corpus and compose retrieval, RAG and safe tools."""
    documents = load_document_directory(corpus_dir)
    if not documents:
        raise RuntimeError(f"No EnterpriseIQ documents found in {corpus_dir!s}")

    chunks = chunk_documents(documents)
    embedding_config = EmbeddingConfig(
        provider="deterministic",
        model_name="enterpriseiq-deterministic-v1",
        dimension=384,
    )
    provider = DeterministicEmbeddingProvider(embedding_config)
    vectors = provider.embed_documents([chunk.text for chunk in chunks])
    records = [
        VectorIndexRecord(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            title=chunk.title,
            department=chunk.department,
            document_type=chunk.document_type,
            version=chunk.version,
            effective_date=chunk.effective_date.isoformat(),
            owner=chunk.owner,
            classification=chunk.classification,
            source_path=chunk.source_path.as_posix(),
            metadata=dict(chunk.metadata),
            vector=vector,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    index = PersistentVectorIndex(
        dimension=provider.dimension,
        model_name=provider.model_name,
        records=records,
    )
    dense_retriever = SemanticRetriever(index=index, provider=provider)
    lexical_retriever = BM25Retriever(index.records)
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        lexical_retriever=lexical_retriever,
    )
    rag_pipeline = GroundedRAGPipeline(
        retriever=retriever,
        generator=DeterministicAnswerGenerator(),
    )

    def search_knowledge(
        *,
        context: ToolContext,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search the indexed enterprise knowledge."""
        del context
        matches = retriever.search(query, top_k=top_k)
        return {
            "matches": [
                {
                    "document_id": item.document_id,
                    "title": item.title,
                    "text": item.text,
                }
                for item in matches
            ]
        }

    def list_documents(
        *,
        context: ToolContext,
        department: str | None = None,
    ) -> dict[str, Any]:
        """List documents, optionally filtering by department."""
        del context
        filtered_documents = [
            doc
            for doc in documents
            if department is None
            or doc.department.casefold() == department.casefold()
        ]
        return {
            "department": department,
            "count": len(filtered_documents),
            "documents": [
                {
                    "document_id": doc.document_id,
                    "title": doc.title,
                }
                for doc in filtered_documents
            ],
        }

    def list_departments(*, context: ToolContext) -> dict[str, Any]:
        """Return the departments represented in the corpus."""
        del context
        return {
            "departments": sorted(
                {doc.department for doc in documents}
            )
        }

    def get_document(
        *,
        context: ToolContext,
        document_id: str,
    ) -> dict[str, Any]:
        """Retrieve a document by its ID."""
        del context
        for doc in documents:
            if doc.document_id.casefold() == document_id.casefold():
                return {
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "department": doc.department,
                    "content": doc.content,
                }
        return {
            "error": "Document not found.",
            "document_id": document_id,
        }

    registry = ToolRegistry(
        (
            ToolDefinition(
                name="search_knowledge",
                description="Search approved enterprise policy knowledge.",
                handler=search_knowledge,
                parameters=(
                    ToolParameter(
                        name="query",
                        value_type=str,
                        description="Search query.",
                    ),
                    ToolParameter(
                        name="top_k",
                        value_type=int,
                        description="Maximum number of search results.",
                        required=False,
                        default=5,
                    ),
                ),
            ),
            ToolDefinition(
                name="list_documents",
                description=(
                    "List available enterprise documents, optionally "
                    "filtered by department."
                ),
                handler=list_documents,
                parameters=(
                    ToolParameter(
                        name="department",
                        value_type=str,
                        description="Optional department filter.",
                        required=False,
                        default=None,
                    ),
                ),
            ),
            ToolDefinition(
                name="list_departments",
                description="List departments represented in the corpus.",
                handler=list_departments,
            ),
            ToolDefinition(
                name="get_document",
                description="Retrieve an approved enterprise document by ID.",
                handler=get_document,
                parameters=(
                    ToolParameter(
                        name="document_id",
                        value_type=str,
                        description="Document ID.",
                    ),
                ),
            ),
        )
    )
    executor = SafeToolExecutor(registry)
    router = DeterministicAgentRouter(registry)
    return AgenticRAGOrchestrator(
        router=router,
        rag_pipeline=rag_pipeline,
        tool_executor=executor,
    )
