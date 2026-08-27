"""Understanding report tests: contract, router dispatch, Markdown determinism."""
from pathlib import Path

import pytest

from docproc.nlp.report import render_markdown, understand, understand_file

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "io"

SAMPLE = ("ACME Corporation - INVOICE #INV-10482\n\n"
          "Nguyen Van A purchased 3 x office chair @ 45.00 and 1 x desk lamp.\n"
          "Total due is $147.50 within 30 days. Payment terms apply.\n"
          "Please reference the invoice number when paying.")


class TestUnderstandRecord:
    def test_full_contract_sections(self):
        r = understand(SAMPLE, source="test")
        assert set(r) == {"source", "doc_type", "structure", "keywords",
                          "topics", "fields", "summary"}
        assert r["structure"]["stats"]["paragraphs"] == 2
        assert len(r["keywords"]) == 10
        assert "topics" in r and "k" in r["topics"]
        assert set(r["fields"]) == {"doc_type", "fields", "matched",
                                    "missing_required"}

    def test_deterministic_record(self):
        a = understand(SAMPLE, source="x")
        b = understand(SAMPLE, source="x")
        assert json_equal(a, b)

    def test_empty_text_still_returns_record(self):
        r = understand("   ", source="blank")
        assert r["keywords"] == []
        assert r["structure"]["stats"]["words"] == 0
        assert r["doc_type"]["label"] == "unavailable"


class TestSummaryIntegration:
    def test_summary_engine_and_counts(self):
        r = understand(SAMPLE, source="t")
        sm = r["summary"]
        assert sm["engine"] == "extractive"
        n_sents = r["structure"]["stats"]["sentences"]
        assert sm["compression"]["original_sentences"] == n_sents
        assert 0 < sm["compression"]["kept"] <= n_sents

    def test_summary_k_passthrough(self):
        r = understand(SAMPLE, source="t", summary_k=1)
        assert len(r["summary"]["sentences"]) == 1

    def test_abstractive_falls_back_to_extractive(self, monkeypatch):
        from docproc.nlp import report

        def boom(text, mode=None, k=None):
            raise NotImplementedError("no checkpoint")

        monkeypatch.setattr(report, "summarize", boom)
        r = understand(SAMPLE, source="t", summary_mode="abstractive")
        assert r["summary"]["engine"] == "extractive"
        assert r["summary"].get("engine_fallback") is True


class TestFileDispatch:
    def test_text_file_routed_through_understanding(self):
        r = understand_file(FIX / "notes.md")
        assert r["file_type"] == "md"
        assert "structure" in r
        # router advisory (D3): label or graceful unavailable
        assert r["doc_type"]["label"] in (*paths_class_names(), None,
                                          "unavailable")

    def test_image_classification_only(self):
        r = understand_file(FIX / "photo.png")
        assert "classification-only" in r.get("note", "")
        assert "structure" not in r
        assert r["doc_type"]["label"] in (*paths_class_names(), "unavailable")

    def test_unknown_binary_raises(self):
        from docproc.io.detect import UnsupportedFormatError

        with pytest.raises(UnsupportedFormatError):
            understand_file(FIX / "unknown.bin")


class TestRouterGate:
    def test_rule_cues_strong(self):
        from docproc.nlp.report import _rule_cues

        cues = _rule_cues("TOTAL DUE $10\nINV-999\nPayment terms: net 30")
        assert cues["invoice"] >= 3

    def test_low_confidence_unavailable(self, monkeypatch):
        from docproc.nlp import report

        monkeypatch.setattr(report, "_router_confidence", lambda m, x: 0.05)
        r = understand("xyzzy quux blah nothing here", source="t")
        assert r["doc_type"]["label"] is None
        assert r["doc_type"]["via"] == "low_confidence"

    def test_low_confidence_rule_fallback(self, monkeypatch):
        from docproc.nlp import report

        monkeypatch.setattr(report, "_router_confidence", lambda m, x: 0.05)
        r = understand("TOTAL DUE $10\nINV-999\nPayment terms: net 30",
                       source="t")
        assert r["doc_type"]["label"] == "invoice"
        assert r["doc_type"]["via"] == "rule_cues"

    def test_high_confidence_uses_model(self, monkeypatch):
        from docproc.nlp import report

        monkeypatch.setattr(report, "_router_confidence", lambda m, x: 0.95)
        r = understand("xyzzy quux blah nothing here", source="t")
        assert r["doc_type"]["via"] == "e0b_svm"
        assert r["doc_type"]["confidence"] == 0.95

    def test_confidence_key_present(self):
        r = understand(SAMPLE, source="t")
        assert "confidence" in r["doc_type"]


class TestMarkdown:
    def test_renders_all_sections_and_deterministic(self):
        r = understand(SAMPLE, source="demo.txt")
        md1 = render_markdown(r)
        md2 = render_markdown(r)
        assert md1 == md2
        for heading in ["# Understanding — demo.txt", "## Summary",
                        "## Structure", "## Keywords", "## Topics"]:
            assert heading in md1

    def test_classification_only_record_renders(self):
        r = {"source": "photo.png",
             "doc_type": {"label": "invoice", "via": "e1_cnn"},
             "note": "classification-only input (no text layer; OCR out of scope)"}
        md = render_markdown(r)
        assert "classification-only" in md and "invoice" in md


def json_equal(a, b) -> bool:
    import json

    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def paths_class_names():
    from docproc import paths

    return paths.class_names()