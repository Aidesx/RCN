"""Stage 7 tests: file type detection + scanned-PDF probing."""
from pathlib import Path

import pytest

from docproc.io.detect import (
    DOCX,
    HTML,
    IMAGE,
    MARKDOWN,
    PDF_SCANNED,
    PDF_TEXT,
    UNKNOWN,
    detect_file_type,
)

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "io"


class TestDetection:
    def test_text_pdf(self):
        d = detect_file_type(FIX / "text_pdf.pdf")
        assert d.file_type == PDF_TEXT
        assert "text_chars=" in d.detail

    def test_embedded_pdf_is_pdf_text(self):
        assert detect_file_type(FIX / "embedded_pdf.pdf").file_type == PDF_TEXT

    def test_scanned_pdf_detected_by_threshold(self):
        d = detect_file_type(FIX / "scanned_pdf.pdf")
        assert d.file_type == PDF_SCANNED

    def test_docx(self):
        d = detect_file_type(FIX / "invoice.docx")
        assert d.file_type == DOCX and ".docx" in d.detail

    def test_markdown_by_extension(self):
        assert detect_file_type(FIX / "notes.md").file_type == MARKDOWN

    def test_html_by_extension(self):
        assert detect_file_type(FIX / "page.html").file_type == HTML

    def test_image_png_magic(self):
        d = detect_file_type(FIX / "photo.png")
        assert d.file_type == IMAGE

    def test_unknown_binary(self):
        assert detect_file_type(FIX / "unknown.bin").file_type == UNKNOWN

    def test_content_sniff_html_without_extension(self, tmp_path):
        p = tmp_path / "page.noext"
        p.write_bytes(b"<html><body><p>hello</p></body></html>")
        assert detect_file_type(p).file_type == HTML

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            detect_file_type(FIX / "missing.md")


class TestScannedThreshold:
    def test_threshold_loaded_from_pipeline_yaml(self):
        from docproc.io.detect import _pipeline_threshold

        assert _pipeline_threshold() == 100

    def test_scanned_detail_mentions_chars(self):
        d = detect_file_type(FIX / "scanned_pdf.pdf")
        assert "<100" in d.detail