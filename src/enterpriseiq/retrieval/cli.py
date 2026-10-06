"""Command-line interface for EnterpriseIQ vector indexing and retrieval."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from enterpriseiq.embeddings import (
    EmbeddingConfig,
    create_embedding_provider,
    run_embedding_pipeline,
)
from enterpriseiq.retrieval.semantic import (
    SearchFilters,
    SemanticRetriever,
)
from enterpriseiq.retrieval.vector_index import PersistentVectorIndex

DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_EMBEDDINGS_PATH = Path("data/processed/embeddings.jsonl")
DEFAULT_EMBEDDING_MANIFEST_PATH = Path(
    "data/processed/embedding_manifest.json"
)
DEFAULT_INDEX_PATH = Path("data/processed/vector_index.jsonl")
DEFAULT_INDEX_MANIFEST_PATH = Path(
    "data/processed/vector_index_manifest.json"
)


def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        default="deterministic",
        choices=(
            "deterministic",
            "test",
            "local",
            "sentence-transformers",
        ),
    )
    parser.add_argument(
        "--model",
        default="enterpriseiq-deterministic-v1",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
    )


def _embedding_config(args: argparse.Namespace) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=args.provider,
        model_name=args.model,
        dimension=args.dimension,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
    )


def build_embeddings(args: argparse.Namespace) -> int:
    """Build deterministic embedding artifacts from Day-01 chunks."""

    config = _embedding_config(args)
    provider = create_embedding_provider(config)

    result = run_embedding_pipeline(
        input_path=args.input,
        output_path=args.output,
        manifest_path=args.manifest,
        provider=provider,
        config=config,
    )

    print("Embedding artifact built successfully")
    print(f"chunks: {result.chunk_count}")
    print(f"dimension: {result.dimension}")
    print(f"model: {result.model_name}")
    print(f"output: {result.output_path}")
    print(f"manifest: {result.manifest_path}")
    print(f"sha256: {result.output_sha256}")

    return 0


def build_index(args: argparse.Namespace) -> int:
    """Build and persist an EnterpriseIQ vector index."""

    index = PersistentVectorIndex.from_embedding_artifact(
        args.embeddings
    )

    manifest = index.save(
        index_path=args.index,
        manifest_path=args.manifest,
    )

    print("Vector index built successfully")
    print(f"records: {manifest.record_count}")
    print(f"dimension: {manifest.dimension}")
    print(f"model: {manifest.model_name}")
    print(f"index: {args.index}")
    print(f"manifest: {args.manifest}")
    print(f"sha256: {manifest.index_sha256}")

    return 0


def _filters_from_args(
    args: argparse.Namespace,
) -> SearchFilters:
    return SearchFilters(
        departments=tuple(args.department or ()),
        classifications=tuple(args.classification or ()),
        document_types=tuple(args.document_type or ()),
        document_ids=tuple(args.document_id or ()),
    )


def _result_to_dict(result: object) -> dict[str, object]:
    return {
        "rank": result.rank,
        "score": result.score,
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "title": result.title,
        "department": result.department,
        "text": result.text,
        "source_path": str(result.source_path),
        "metadata": dict(result.metadata),
        "citation": result.citation_label,
    }


def search_index(args: argparse.Namespace) -> int:
    """Search a persisted EnterpriseIQ vector index."""

    index = PersistentVectorIndex.load(
        index_path=args.index,
        manifest_path=args.manifest,
    )

    config = EmbeddingConfig(
        provider=args.provider,
        model_name=index.model_name,
        dimension=index.dimension,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
    )

    provider = create_embedding_provider(config)

    retriever = SemanticRetriever(
        index=index,
        provider=provider,
    )

    results = retriever.search(
        args.query,
        top_k=args.top_k,
        filters=_filters_from_args(args),
        min_score=args.min_score,
    )

    if args.json:
        print(
            json.dumps(
                [_result_to_dict(result) for result in results],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")

    if not results:
        print("No matching chunks found.")
        return 0

    for result in results:
        print()
        print(
            f"[{result.rank}] "
            f"{result.citation_label} "
            f"score={result.score:.6f}"
        )
        print(f"department: {result.department}")
        print(f"chunk_id: {result.chunk_id}")
        print(f"source: {result.source_path}")
        print(f"text: {result.text}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the EnterpriseIQ retrieval CLI parser."""

    parser = argparse.ArgumentParser(
        prog="enterpriseiq-retrieval",
        description=(
            "Build embeddings, persist vector indexes, "
            "and search enterprise knowledge."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    embeddings_parser = subparsers.add_parser(
        "embed",
        help="Build embeddings from Day-01 chunks.",
    )
    embeddings_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
    )
    embeddings_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
    )
    embeddings_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_EMBEDDING_MANIFEST_PATH,
    )
    _add_embedding_arguments(embeddings_parser)
    embeddings_parser.set_defaults(func=build_embeddings)

    index_parser = subparsers.add_parser(
        "index",
        help="Build a persistent vector index.",
    )
    index_parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
    )
    index_parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
    )
    index_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_INDEX_MANIFEST_PATH,
    )
    index_parser.set_defaults(func=build_index)

    search_parser = subparsers.add_parser(
        "search",
        help="Search a persisted vector index.",
    )
    search_parser.add_argument(
        "query",
        type=str,
    )
    search_parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
    )
    search_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_INDEX_MANIFEST_PATH,
    )
    search_parser.add_argument(
        "--provider",
        default="deterministic",
        choices=(
            "deterministic",
            "test",
            "local",
            "sentence-transformers",
        ),
    )
    search_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    search_parser.add_argument(
        "--no-normalize",
        action="store_true",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    search_parser.add_argument(
        "--min-score",
        type=float,
        default=None,
    )
    search_parser.add_argument(
        "--department",
        action="append",
    )
    search_parser.add_argument(
        "--classification",
        action="append",
    )
    search_parser.add_argument(
        "--document-type",
        action="append",
    )
    search_parser.add_argument(
        "--document-id",
        action="append",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        dest="json",
    )
    search_parser.set_defaults(func=search_index)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the retrieval CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
