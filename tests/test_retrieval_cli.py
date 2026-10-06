"""Tests for the EnterpriseIQ retrieval CLI."""

import json
from pathlib import Path

from enterpriseiq.retrieval.cli import main


def write_chunks(path: Path) -> None:
    records = [
        {
            "chunk_id": "HR-003::0000",
            "document_id": "HR-003",
            "chunk_index": 0,
            "text": "Employees may work remotely three days per workweek.",
            "title": "Remote Work Policy",
            "department": "HR",
            "document_type": "policy",
            "version": "1.0",
            "effective_date": "2026-01-01",
            "owner": "People Operations",
            "classification": "internal",
            "source_path": "data/raw/hr/remote_work_policy.md",
            "metadata": {"classification": "internal"},
        },
        {
            "chunk_id": "FIN-001::0000",
            "document_id": "FIN-001",
            "chunk_index": 0,
            "text": "Receipts are required for expense reimbursement.",
            "title": "Expense Reimbursement Policy",
            "department": "Finance",
            "document_type": "policy",
            "version": "1.0",
            "effective_date": "2026-01-01",
            "owner": "Finance",
            "classification": "internal",
            "source_path": "data/raw/finance/expense_policy.md",
            "metadata": {"classification": "internal"},
        },
    ]

    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_cli_end_to_end_build_and_search(
    tmp_path: Path,
    capsys: object,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.jsonl"
    embedding_manifest = tmp_path / "embedding_manifest.json"
    index = tmp_path / "vector_index.jsonl"
    index_manifest = tmp_path / "vector_index_manifest.json"

    write_chunks(chunks)

    assert (
        main(
            [
                "embed",
                "--input",
                str(chunks),
                "--output",
                str(embeddings),
                "--manifest",
                str(embedding_manifest),
                "--provider",
                "deterministic",
                "--model",
                "cli-test-model",
                "--dimension",
                "12",
                "--batch-size",
                "1",
            ]
        )
        == 0
    )

    assert embeddings.exists()
    assert embedding_manifest.exists()

    assert (
        main(
            [
                "index",
                "--embeddings",
                str(embeddings),
                "--index",
                str(index),
                "--manifest",
                str(index_manifest),
            ]
        )
        == 0
    )

    assert index.exists()
    assert index_manifest.exists()

    assert (
        main(
            [
                "search",
                "Employees may work remotely three days per workweek.",
                "--index",
                str(index),
                "--manifest",
                str(index_manifest),
                "--provider",
                "deterministic",
                "--top-k",
                "1",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out

    assert "Remote Work Policy" in output
    assert "HR-003" in output
    assert "score=1.000000" in output


def test_cli_search_supports_metadata_filter(
    tmp_path: Path,
    capsys: object,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.jsonl"
    embedding_manifest = tmp_path / "embedding_manifest.json"
    index = tmp_path / "vector_index.jsonl"
    index_manifest = tmp_path / "vector_index_manifest.json"

    write_chunks(chunks)

    main(
        [
            "embed",
            "--input",
            str(chunks),
            "--output",
            str(embeddings),
            "--manifest",
            str(embedding_manifest),
            "--provider",
            "deterministic",
            "--model",
            "cli-filter-model",
            "--dimension",
            "12",
        ]
    )

    main(
        [
            "index",
            "--embeddings",
            str(embeddings),
            "--index",
            str(index),
            "--manifest",
            str(index_manifest),
        ]
    )

    main(
        [
            "search",
            "company policy",
            "--index",
            str(index),
            "--manifest",
            str(index_manifest),
            "--provider",
            "deterministic",
            "--department",
            "Finance",
            "--top-k",
            "5",
        ]
    )

    output = capsys.readouterr().out

    assert "Expense Reimbursement Policy" in output
    assert "FIN-001" in output


def test_cli_json_search_output(
    tmp_path: Path,
    capsys: object,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.jsonl"
    embedding_manifest = tmp_path / "embedding_manifest.json"
    index = tmp_path / "vector_index.jsonl"
    index_manifest = tmp_path / "vector_index_manifest.json"

    write_chunks(chunks)

    main(
        [
            "embed",
            "--input",
            str(chunks),
            "--output",
            str(embeddings),
            "--manifest",
            str(embedding_manifest),
            "--provider",
            "deterministic",
            "--model",
            "cli-json-model",
            "--dimension",
            "12",
        ]
    )

    main(
        [
            "index",
            "--embeddings",
            str(embeddings),
            "--index",
            str(index),
            "--manifest",
            str(index_manifest),
        ]
    )

    # Discard build output so only JSON search output remains.
    capsys.readouterr()

    main(
        [
            "search",
            "remote work",
            "--index",
            str(index),
            "--manifest",
            str(index_manifest),
            "--provider",
            "deterministic",
            "--top-k",
            "1",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert len(payload) == 1
    assert payload[0]["rank"] == 1
    assert "citation" in payload[0]
    assert "source_path" in payload[0]


def test_cli_search_with_unmatched_filter_returns_cleanly(
    tmp_path: Path,
    capsys: object,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.jsonl"
    embedding_manifest = tmp_path / "embedding_manifest.json"
    index = tmp_path / "vector_index.jsonl"
    index_manifest = tmp_path / "vector_index_manifest.json"

    write_chunks(chunks)

    main(
        [
            "embed",
            "--input",
            str(chunks),
            "--output",
            str(embeddings),
            "--manifest",
            str(embedding_manifest),
            "--provider",
            "deterministic",
            "--model",
            "cli-empty-model",
            "--dimension",
            "12",
        ]
    )

    main(
        [
            "index",
            "--embeddings",
            str(embeddings),
            "--index",
            str(index),
            "--manifest",
            str(index_manifest),
        ]
    )

    capsys.readouterr()

    assert (
        main(
            [
                "search",
                "policy",
                "--index",
                str(index),
                "--manifest",
                str(index_manifest),
                "--provider",
                "deterministic",
                "--department",
                "Legal",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "No matching chunks found." in output

