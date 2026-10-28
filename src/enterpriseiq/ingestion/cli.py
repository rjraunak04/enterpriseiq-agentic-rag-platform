"""Command-line interface for EnterpriseIQ document ingestion."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from enterpriseiq.ingestion.chunker import ChunkingConfig
from enterpriseiq.ingestion.pipeline import (
    result_to_dict,
    run_ingestion,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the ingestion CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Load, validate, clean, and chunk the EnterpriseIQ "
            "enterprise knowledge corpus."
        )
    )

    parser.add_argument(
        "--source",
        default="data/raw",
        help="Directory containing source Markdown documents.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/chunks.jsonl",
        help="Destination JSONL chunk artifact.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/ingestion_manifest.json",
        help="Destination ingestion manifest.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Requested overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=100,
        help="Minimum boundary-search size.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the ingestion CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    config = ChunkingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
    )

    result = run_ingestion(
        source_directory=args.source,
        output_file=args.output,
        manifest_file=args.manifest,
        config=config,
    )

    print(
        json.dumps(
            result_to_dict(result),
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
