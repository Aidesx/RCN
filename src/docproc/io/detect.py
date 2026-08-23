"""File type detection via magic bytes/extension + scanned-PDF text probe."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PDF_TEXT = "pdf_text"
PDF_SCANNED = "pdf_scanned"
DOCX = "docx"
MARKDOWN = "md"
HTML = "html"
IMAGE = "image"
UNKNOWN = "unknown"

_TEXT_SUFFIXES = {
    ".md": MARKDOWN,
    ".markdown": MARKDOWN,
    ".html": HTML,
    ".htm": HTML,
}


@dataclass
class Detection:
    file_type: str
    detail: str = ""


def _pipeline_threshold() -> int:
    from docproc import paths

    return int(paths.pipeline_config()["detection"]["scanned_text_threshold_chars"])


def _magic_type(head: bytes) -> str | None:
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return IMAGE
    if head.startswith(b"\xff\xd8\xff"):
        return IMAGE
    if head.startswith((b"GIF87a", b"GIF89a")):
        return IMAGE
    if head.startswith(b"BM"):
        return IMAGE
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return IMAGE
    return None


def _sniff_text_format(text_head: str) -> str:
    low = text_head.lower()
    if "<html" in low or "<!doctype html" in low or ("<body" in low and "</body>" in low):
        return HTML
    if low.lstrip().startswith(("#", "- [ ]", "- [x]")) or "\n## " in low:
        return MARKDOWN
    return UNKNOWN


def pdf_text_char_count(path) -> int:
    """Total non-whitespace characters of the PDF text layer (all pages)."""
    import pymupdf

    try:
        with pymupdf.open(str(path)) as doc:
            total = 0
            for page in doc:
                total += sum(ch.isprintable() and not ch.isspace() for ch in page.get_text())
            return total
    except Exception as exc:  # malformed pdf
        raise ParseError("pdf", path, f"cannot open/probe: {exc}") from exc


def detect_file_type(path) -> Detection:
    """Detect the file type by magic bytes, then extension/content fallback."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"file not found: {p}")
    with p.open("rb") as fh:
        head = fh.read(512)

    magic = _magic_type(head)
    if magic == "pdf":
        chars = pdf_text_char_count(p)
        threshold = _pipeline_threshold()
        if chars < threshold:
            return Detection(PDF_SCANNED, f"text_chars={chars}<{threshold}")
        return Detection(PDF_TEXT, f"text_chars={chars}")
    if magic == IMAGE:
        return Detection(IMAGE, "magic-bytes image signature")

    suffix = p.suffix.lower()
    if magic == "zip":
        # zip container: DOCX is the only office format in scope
        if suffix == ".docx":
            return Detection(DOCX, "zip magic + .docx")
        raise ParseError("zip", p, "zip container that is not .docx is unsupported")

    text_head = ""
    try:
        text_head = head.decode("utf-8", errors="replace")
    except Exception:
        pass

    if suffix in _TEXT_SUFFIXES:
        fmt = _TEXT_SUFFIXES[suffix]
        return Detection(fmt, f"extension {suffix}")

    sniffed = _sniff_text_format(text_head)
    if sniffed != UNKNOWN:
        return Detection(sniffed, "content sniff")
    if suffix in (".txt",):
        return Detection(MARKDOWN, "plain text treated as markdown source")
    return Detection(UNKNOWN, f"unrecognized magic+extension {suffix or '<none>'}")


class DocumentIOError(Exception):
    """Base class for structured I/O errors (never crash the pipeline)."""

    def __init__(self, file_type: str, path, reason: str):
        self.file_type = file_type
        self.path = str(path)
        self.reason = reason
        super().__init__(f"[{file_type}] {path}: {reason}")


class ParseError(DocumentIOError):
    pass


class UnsupportedFormatError(DocumentIOError):
    pass