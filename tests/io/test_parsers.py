"""Stage 7 golden tests: text extraction per format + structured errors."""
from pathlib import Path

import pytest

from docproc.io.detect import (
    DOCX,
    HTML,
    MARKDOWN,
    PDF_SCANNED,
    PDF_TEXT,
    IMAGE,
    ParseError,
    UnsupportedFormatError,
)
from docproc.io.parsers import extract_text

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "io"

GOLDEN_CASES = [
    ("text_pdf.pdf", PDF_TEXT),
    ("embedded_pdf.pdf", PDF_TEXT),
    ("invoice.docx", DOCX),
    ("notes.md", MARKDOWN),
    ("page.html", HTML),
]


def _golden(name: str) -> str:
    return (FIX / f"{name}.expected.txt").read_text(encoding="utf-8")


def _norm(s: str) -> str:
    """Collapse layout-only line wrapping; word sequence must match exactly."""
    return " ".join(s.split())


@pytest.mark.parametrize("name,file_type", GOLDEN_CASES)
def test_golden_extraction_exact(name, file_type):
    assert _norm(extract_text(FIX / name)) == _norm(_golden(name))


def test_explicit_type_override_skips_detection(tmp_path):
    src = FIX / "notes.md"
    dst = tmp_path / "renamed.noext"
    dst.write_bytes(src.read_bytes())
    assert extract_text(dst, file_type=MARKDOWN) == _golden("notes.md")


def test_extract_text_on_scanned_pdf_raises_unsupported():
    with pytest.raises(UnsupportedFormatError) as ei:
        extract_text(FIX / "scanned_pdf.pdf")
    assert ei.value.file_type == PDF_SCANNED


def test_extract_text_on_image_raises_unsupported():
    with pytest.raises(UnsupportedFormatError):
        extract_text(FIX / "photo.png")


def test_corrupt_docx_structured_error():
    with pytest.raises(ParseError) as ei:
        extract_text(FIX / "corrupt.docx")
    assert ei.value.file_type == DOCX
    assert ei.value.path.endswith("corrupt.docx")


def test_truncated_pdf_parse_error_has_reason():
    with pytest.raises(ParseError) as ei:
        extract_text(FIX / "truncated.pdf", file_type=PDF_TEXT)
    assert "failed" in ei.value.reason.lower() or "cannot" in ei.value.reason.lower()


def test_errors_are_document_io_error_subclasses():
    from docproc.io.detect import DocumentIOError

    assert issubclass(ParseError, DocumentIOError)
    assert issubclass(UnsupportedFormatError, DocumentIOError)


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_text(FIX / "nope.md")