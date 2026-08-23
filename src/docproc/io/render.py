"""PDF page rendering (fixed-DPI PIL images) + embedded-image extraction."""
from __future__ import annotations

import io

from PIL import Image

from docproc.io.detect import ParseError

DEFAULT_DPI = 150


def render_pdf_pages(path, dpi: int = DEFAULT_DPI) -> list[Image.Image]:
    """Render every page to an RGB PIL image."""
    import pymupdf

    try:
        pages: list[Image.Image] = []
        with pymupdf.open(str(path)) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append(img)
        return pages
    except Exception as exc:
        raise ParseError("pdf", path, f"render failed: {exc}") from exc


def extract_embedded_images_pdf(path) -> list[Image.Image]:
    """Extract embedded raster images referenced by PDF pages."""
    import pymupdf

    try:
        images: list[Image.Image] = []
        with pymupdf.open(str(path)) as doc:
            for page in doc:
                for info in page.get_images(full=True):
                    xref = info[0]
                    raw = doc.extract_image(xref)
                    img = Image.open(io.BytesIO(raw["image"]))
                    img.load()
                    images.append(img.convert("RGB"))
        return images
    except Exception as exc:
        raise ParseError("pdf", path, f"embedded image extraction failed: {exc}") from exc