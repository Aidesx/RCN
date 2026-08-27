"""Stage 7 tests: PDF page rendering determinism + embedded image extraction."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from docproc.io.detect import ParseError
from docproc.io.render import extract_embedded_images_pdf, render_pdf_pages

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "io"


class TestRender:
    def test_renders_one_page_rgb(self):
        pages = render_pdf_pages(FIX / "text_pdf.pdf")
        assert len(pages) == 1
        assert isinstance(pages[0], Image.Image)
        assert pages[0].mode == "RGB"
        assert pages[0].size[0] > 0 and pages[0].size[1] > 0

    def test_rendering_deterministic(self):
        a = np.asarray(render_pdf_pages(FIX / "scanned_pdf.pdf")[0])
        b = np.asarray(render_pdf_pages(FIX / "scanned_pdf.pdf")[0])
        assert a.shape == b.shape and np.array_equal(a, b)

    def test_dpi_changes_resolution_deterministically(self):
        small = render_pdf_pages(FIX / "text_pdf.pdf", dpi=72)[0]
        large = render_pdf_pages(FIX / "text_pdf.pdf", dpi=150)[0]
        assert large.size[0] > small.size[0]

    def test_truncated_pdf_structured_error(self):
        with pytest.raises(ParseError):
            render_pdf_pages(FIX / "truncated.pdf")


class TestEmbeddedImages:
    def test_extracts_embedded_logo(self):
        imgs = extract_embedded_images_pdf(FIX / "embedded_pdf.pdf")
        assert len(imgs) >= 1
        assert all(isinstance(i, Image.Image) for i in imgs)

    def test_no_embedded_images_in_scanned_render(self):
        # scanned fixture embeds a full-page image -> at least one found
        imgs = extract_embedded_images_pdf(FIX / "scanned_pdf.pdf")
        assert len(imgs) >= 1

    def test_corrupt_pdf_structured_error(self):
        with pytest.raises(ParseError):
            extract_embedded_images_pdf(FIX / "truncated.pdf")