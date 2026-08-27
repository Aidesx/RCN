"""L5 summary tests: determinism, contract, ordering, quality (07-summary v1)."""
import pytest

from docproc.nlp.summary import summarize_extractive

SAMPLE = (
    "INVOICE #INV-10482 issued by Acme Corporation. The invoice covers office chairs.\n\n"
    "Payment terms: net 30 days. Acme Corporation thanks you for the order.\n"
    "The office chairs ship next week from the Acme warehouse."
)

VIET = ("Hóa đơn số 123 của công ty Acme. Công ty Acme xác nhận tổng tiền "
        "45.000 đồng cho đơn hàng bàn ghế văn phòng.\n\n"
        "Công ty Acme sẽ giao bàn ghế vào tuần sau tại kho công ty.")

# Same token multiset in both sentences -> equal relevance; the paragraph
# initial sentence must win via position_bonus.
TIE = "Acme ships office chairs fast. Fast office chairs ships Acme."


class TestContract:
    def test_deterministic_repeat(self):
        a = summarize_extractive(SAMPLE, k=2)
        b = summarize_extractive(SAMPLE, k=2)
        assert a == b

    def test_shape(self):
        out = summarize_extractive(SAMPLE, k=2)
        assert set(out) == {"engine", "sentences", "compression"}
        assert out["engine"] == "extractive"
        assert set(out["compression"]) == {"original_sentences", "kept"}
        for s in out["sentences"]:
            assert set(s) == {"text", "paragraph", "score"}

    def test_respects_k(self):
        assert len(summarize_extractive(SAMPLE, k=1)["sentences"]) == 1
        total = summarize_extractive(SAMPLE, k=99)["compression"]
        assert total["kept"] == total["original_sentences"]

    def test_document_order(self):
        picked = [s["paragraph"] for s in
                  summarize_extractive(SAMPLE, k=4)["sentences"]]
        assert picked == sorted(picked)

    def test_empty_text(self):
        out = summarize_extractive("   \n\n ")
        assert out["sentences"] == []
        assert out["compression"] == {"original_sentences": 0, "kept": 0}

    def test_invalid_inputs(self):
        with pytest.raises(TypeError):
            summarize_extractive(42)
        with pytest.raises(ValueError):
            summarize_extractive(SAMPLE, k=0)

    def test_compression_counts(self):
        out = summarize_extractive(SAMPLE, k=2)
        assert out["compression"]["kept"] == 2
        assert out["compression"]["original_sentences"] >= 2


class TestQuality:
    def test_keyword_heavy_sentence_wins(self):
        doc = ("Acme Corporation files quarterly reports.\n\n"
               "Warehouse inventory: the Acme warehouse stores office chairs. "
               "Chairs leave the Acme warehouse weekly.")
        top = summarize_extractive(doc, k=1)["sentences"][0]["text"]
        assert "warehouse" in top.lower()

    def test_position_breaks_exact_tie(self):
        top = summarize_extractive(TIE, k=1)["sentences"][0]
        assert top["text"].startswith("Acme ships")

    def test_vietnamese_text(self):
        out = summarize_extractive(VIET, k=1)
        assert out["sentences"]
        text = out["sentences"][0]["text"].lower()
        assert "acme" in text or "công ty" in text

    def test_no_duplicate_sentences(self):
        sents = [s["text"] for s in summarize_extractive(SAMPLE, k=4)["sentences"]]
        assert len(sents) == len(set(sents))

    def test_scores_sorted_within_selection_irrelevant_but_finite(self):
        for s in summarize_extractive(SAMPLE, k=3)["sentences"]:
            assert isinstance(s["score"], float)


class TestDispatcher:
    def test_default_mode_extractive(self):
        from docproc.nlp.summary import summarize

        assert summarize(SAMPLE, k=1)["engine"] == "extractive"

    def test_abstractive_engine_errors_propagate(self, monkeypatch):
        from docproc.nlp import summary as mod

        def boom(*args, **kwargs):
            raise NotImplementedError("no checkpoint")

        monkeypatch.setattr(mod, "summarize_abstractive", boom)
        with pytest.raises(NotImplementedError):
            mod.summarize(SAMPLE, mode="abstractive", k=1)

    def test_abstractive_smoke_when_checkpoint_present(self):
        from pathlib import Path

        from docproc import paths

        ckpt = paths.ROOT / "models" / "artifacts" / "summarizer_mt5"
        if not (ckpt / "config.json").exists():
            pytest.skip("summarizer checkpoint not downloaded")
        from docproc.nlp.summary import summarize_abstractive

        out = summarize_abstractive(VIET + " " + SAMPLE)
        assert out["engine"] == "abstractive"
        assert out["text"]

    def test_unknown_mode_rejected(self):
        from docproc.nlp.summary import summarize

        with pytest.raises(ValueError):
            summarize(SAMPLE, mode="telepathy")
