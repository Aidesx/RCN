"""io package: deterministic document detection, parsing, rendering."""
from docproc.io.detect import (
    DOCX,
    HTML,
    IMAGE,
    MARKDOWN,
    PDF_SCANNED,
    PDF_TEXT,
    UNKNOWN,
    Detection,
    ParseError,
    UnsupportedFormatError,
    detect_file_type,
)
from docproc.io.parsers import extract_text
from docproc.io.render import extract_embedded_images_pdf, render_pdf_pages

__all__ = [
    "DOCX", "HTML", "IMAGE", "MARKDOWN", "PDF_SCANNED", "PDF_TEXT", "UNKNOWN",
    "Detection", "ParseError", "UnsupportedFormatError",
    "detect_file_type", "extract_text", "render_pdf_pages",
    "extract_embedded_images_pdf",
]