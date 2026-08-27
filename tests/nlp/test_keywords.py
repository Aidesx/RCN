"""L2 keywords tests: determinism, stopword hygiene, bigrams, contract."""
import pytest

from docproc.nlp.keywords import extract_keywords

SAMPLE = (
    "INVOICE #INV-10482 issued by Acme Corporation. The invoice covers office chairs.\n\n"
    "Payment terms: net 30 days. Acme Corporation thanks you for the order.\n"
    "The office chairs ship next week from the Acme warehouse."
)

VIET = ("Hóa đơn số 123 của công ty Acme. Công ty Acme xác nhận tổng tiền "
        "45.000 đồng cho đơn hàng bàn ghế văn phòng.")


class TestContract:
    def test_deterministic_repeat(self):
        a = extract_keywords(SAMPLE, k=8)
        b = extract_keywords(SAMPLE, k=8)
        assert a == b

    def test_shape_and_k(self):
        out = extract_keywords(SAMPLE, k=5)
        assert len(out["keywords"]) == 5
        kw = out["keywords"][0]
        assert set(kw) == {"term", "score", "count"}

    def test_sorted_by_score_desc(self):
        kws = extract_keywords(SAMPLE, k=10)["keywords"]
        scores = [x["score"] for x in kws]
        assert scores == sorted(scores, reverse=True)

    def test_empty_text(self):
        assert extract_keywords("")["keywords"] == []
        assert extract_keywords("   \n\n  ")["keywords"] == []

    def test_invalid_inputs(self):
        with pytest.raises(TypeError):
            extract_keywords(42)
        with pytest.raises(ValueError):
            extract_keywords("hello world", k=0)


class TestQuality:
    def test_no_stopword_leakage(self):
        from docproc.nlp.keywords import _stopwords

        kws = extract_keywords(SAMPLE, k=10)["keywords"]
        leaked = [x for x in kws
                  if all(tok in _stopwords() for tok in x["term"].split())]
        assert leaked == []

    def test_bigrams_present_when_repeated(self):
        kws = extract_keywords(SAMPLE, k=10)["keywords"]
        assert any(" " in x["term"] for x in kws)
        # "office chairs" repeats across paragraphs — with the D1 no-overlap
        # rule it outranks its member unigrams (tie-break prefers bigrams).
        assert any("office chairs" == x["term"] for x in kws)

    def test_no_overlap_in_top_k(self):
        """Spec 03 D1: deduplicated output — no term shares a token with another."""
        kws = extract_keywords(SAMPLE, k=10)["keywords"]
        used: set[str] = set()
        for x in kws:
            parts = set(x["term"].split())
            assert not (parts & used), f"overlapping term in top-k: {x['term']}"
            used |= parts

    def test_repeated_single_word_no_self_bigram(self):
        out = extract_keywords("invoice invoice invoice", k=3)
        assert out["keywords"][0]["term"] == "invoice"
        assert out["keywords"][0]["count"] == 3
        assert all(" " not in x["term"] for x in out["keywords"])

    def test_numbers_filtered(self):
        out = extract_keywords("Item 10482 costs 45. Item 10482 again.", k=10)
        assert not any(x["term"].isdigit() for x in out["keywords"])

    def test_vietnamese_text(self):
        kws = extract_keywords(VIET, k=6)["keywords"]
        assert kws and any("acme" in x["term"] for x in kws)

    def test_single_word_doc(self):
        out = extract_keywords("invoice invoice invoice", k=3)
        assert out["keywords"][0]["term"] == "invoice"
        assert out["keywords"][0]["count"] == 3