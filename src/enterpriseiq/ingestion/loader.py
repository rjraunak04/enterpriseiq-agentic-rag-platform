"""Metadata-aware loader for EnterpriseIQ Markdown documents."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from enterpriseiq.domain import EnterpriseDocument
from enterpriseiq.ingestion.cleaner import clean_text

_REQUIRED_METADATA_FIELDS = frozenset(
    {
        "document_id",
        "title",
        "department",
        "document_type",
        "version",
        "effective_date",
        "owner",
        "classification",
    }
)


class DocumentLoadError(ValueError):
    """Raised when a source document cannot satisfy the ingestion contract."""


def _parse_scalar(value: str) -> str:
    """Normalize a simple front-matter scalar value."""
    value = value.strip()

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {'"', "'"}
    ):
        value = value[1:-1]

    return value.strip()


def parse_front_matter(raw_text: str) -> tuple[dict[str, str], str]:
    """Parse simple key/value front matter and return metadata plus body."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # UTF-8 BOM may be present in files written by Windows tooling.
    normalized = normalized.lstrip("\ufeff")

    lines = normalized.split("\n")

    if not lines or lines[0].strip() != "---":
        raise DocumentLoadError(
            "document must begin with a '---' front-matter delimiter"
        )

    closing_index: int | None = None

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        raise DocumentLoadError(
            "document front matter is missing the closing '---' delimiter"
        )

    metadata: dict[str, str] = {}

    for line_number, line in enumerate(lines[1:closing_index], start=2):
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if ":" not in stripped:
            raise DocumentLoadError(
                f"invalid front-matter entry on line {line_number}: {line!r}"
            )

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = _parse_scalar(value)

        if not key:
            raise DocumentLoadError(
                f"empty metadata key on line {line_number}"
            )

        if not value:
            raise DocumentLoadError(
                f"metadata field {key!r} must not be empty"
            )

        if key in metadata:
            raise DocumentLoadError(
                f"duplicate metadata field: {key}"
            )

        metadata[key] = value

    body = "\n".join(lines[closing_index + 1 :])

    return metadata, body


def _validate_required_metadata(metadata: dict[str, str]) -> None:
    """Ensure metadata required by the domain contract is present."""
    missing = sorted(_REQUIRED_METADATA_FIELDS - metadata.keys())

    if missing:
        raise DocumentLoadError(
            "missing required metadata fields: " + ", ".join(missing)
        )


def _parse_effective_date(value: str) -> date:
    """Parse an ISO-8601 effective date."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DocumentLoadError(
            f"effective_date must use YYYY-MM-DD format: {value!r}"
        ) from exc


def load_markdown_document(path: str | Path) -> EnterpriseDocument:
    """Load one metadata-rich Markdown file into the canonical contract."""
    source_path = Path(path)

    if not source_path.exists():
        raise FileNotFoundError(f"document does not exist: {source_path}")

    if not source_path.is_file():
        raise DocumentLoadError(
            f"document path must reference a file: {source_path}"
        )

    if source_path.suffix.lower() != ".md":
        raise DocumentLoadError(
            f"unsupported document type: {source_path.suffix or '<none>'}"
        )

    try:
        raw_text = source_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"document must be valid UTF-8: {source_path}"
        ) from exc

    metadata, body = parse_front_matter(raw_text)

    _validate_required_metadata(metadata)

    content = clean_text(body)

    if not content:
        raise DocumentLoadError(
            f"document body must not be empty: {source_path}"
        )

    effective_date = _parse_effective_date(metadata["effective_date"])

    required_keys = _REQUIRED_METADATA_FIELDS
    extra_metadata: dict[str, Any] = {
        key: value
        for key, value in metadata.items()
        if key not in required_keys
    }

    return EnterpriseDocument(
        document_id=metadata["document_id"],
        title=metadata["title"],
        department=metadata["department"],
        document_type=metadata["document_type"],
        version=metadata["version"],
        effective_date=effective_date,
        owner=metadata["owner"],
        classification=metadata["classification"],
        content=content,
        source_path=source_path,
        metadata=extra_metadata,
    )


def load_document_directory(
    root: str | Path,
) -> list[EnterpriseDocument]:
    """Recursively load all Markdown documents below a corpus directory."""
    root_path = Path(root)

    if not root_path.exists():
        raise FileNotFoundError(
            f"document directory does not exist: {root_path}"
        )

    if not root_path.is_dir():
        raise DocumentLoadError(
            f"document root must be a directory: {root_path}"
        )

    paths = sorted(
        root_path.rglob("*.md"),
        key=lambda path: path.as_posix().lower(),
    )

    documents = [load_markdown_document(path) for path in paths]

    seen_ids: dict[str, Path] = {}

    for document in documents:
        previous_path = seen_ids.get(document.document_id)

        if previous_path is not None:
            raise DocumentLoadError(
                "duplicate document_id "
                f"{document.document_id!r}: "
                f"{previous_path} and {document.source_path}"
            )

        seen_ids[document.document_id] = document.source_path

    return documents

