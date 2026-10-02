"""Tests for conservative enterprise text cleaning."""

import pytest

from enterpriseiq.ingestion import clean_text


def test_clean_text_normalizes_line_endings_and_whitespace() -> None:
    raw = (
        "# Policy\r\n"
        "\r\n"
        "Employees   may\twork remotely.   \r\n"
        "\r\n"
        "\r\n"
        "## Rules\r\n"
    )

    assert clean_text(raw) == (
        "# Policy\n\n"
        "Employees may work remotely.\n\n"
        "## Rules"
    )


def test_clean_text_preserves_markdown_structure() -> None:
    raw = "# Heading\n\nParagraph one.\n\n## Section\n\nParagraph two."

    assert clean_text(raw) == raw


def test_clean_text_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        clean_text(123)  # type: ignore[arg-type]
