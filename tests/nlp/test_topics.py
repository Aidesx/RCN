"""L3 topics tests: coherence selection, determinism, fallback, mixtures."""
import pytest

from docproc.nlp.topics import extract_topics, umass_coherence, n_topics_cap

CORPUS_DOC = "\n\n".join([
    "The quarterly report shows steady revenue growth across regions.",
    "Revenue growth came from logistics and warehouse automation investments.",
    "Warehouse automation reduced processing latency significantly.",
    "Marketing launched three campaigns; campaign feedback was positive overall.",
    "Campaign performance improved brand awareness metrics substantially.",
    "Finance approved additional budget for automation and marketing programs.",
    "Budget review highlighted logistics savings and marketing returns.",
])


class TestSelection:
    def test_auto_k_returns_curve_and_best(self):
        out = extract_topics(CORPUS_DOC)
        assert out["selected_by"] == "argmax_umass"
        ks = [c["k"] for c in out["coherence_curve"]]
        assert ks[0] == 3 and len(ks) >= 3
        best = max(out["coherence_curve"], key=lambda c: c["umass"])
        assert out["k"] == best["k"]

    def test_fixed_k_respected(self):
        out = extract_topics(CORPUS_DOC, k=4)
        assert out["selected_by"] == "fixed" and out["k"] == 4
        assert len(out["topics"]) == 4

    def test_deterministic_with_seed(self):
        a = extract_topics(CORPUS_DOC, k=3)
        b = extract_topics(CORPUS_DOC, k=3)
        assert a["topics"] == b["topics"]
        assert a["doc_topic_mixture"] == b["doc_topic_mixture"]


class TestShape:
    def test_topics_have_words_and_mixture_sums_one(self):
        out = extract_topics(CORPUS_DOC, k=3)
        for t in out["topics"]:
            assert t["top_words"] and all(isinstance(w, str) for w in t["top_words"])
        assert sum(out["doc_topic_mixture"]) == pytest.approx(1.0, abs=1e-3)

    def test_short_text_fallback(self):
        out = extract_topics("Just one tiny line.")
        assert out["selected_by"] == "fallback_short_text"
        assert out["k"] == 1 and out["coherence"] is None
        assert out["doc_topic_mixture"] == [1.0]

    def test_vietnamese_doc_runs(self):
        vi = ("\n\n".join([
            "Công ty báo cáo doanh thu tăng trưởng ổn định trong quý.",
            "Doanh thu đến từ logistics và tự động hóa kho vận hành.",
            "Tự động hóa giúp giảm thời gian xử lý đơn hàng đáng kể.",
        ]))
        out = extract_topics(vi, k=2)
        assert out["k"] == 2 and out["topics"]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            extract_topics(None)


class TestLabels:
    def test_topics_have_labels(self):
        out = extract_topics(CORPUS_DOC, k=3)
        assert len(out["topics"]) == 3
        for t in out["topics"]:
            assert isinstance(t.get("label"), str) and t["label"]

    def test_labels_are_doc_words(self):
        out = extract_topics(CORPUS_DOC, k=3)
        doc_tokens = {w.strip(".,;:!?()\"'") for w in CORPUS_DOC.lower().split()
                      if len(w.strip(".,;:!?()\"'")) > 1}
        for t in out["topics"]:
            assert all(w in doc_tokens for w in t["label"].split()), t["label"]

    def test_labels_deterministic(self):
        a = extract_topics(CORPUS_DOC, k=3)
        b = extract_topics(CORPUS_DOC, k=3)
        assert [t["label"] for t in a["topics"]] == \
            [t["label"] for t in b["topics"]]

    def test_labels_distinct(self):
        out = extract_topics(CORPUS_DOC, k=4)
        labels = [t["label"] for t in out["topics"]]
        assert len(set(labels)) == len(labels)

    def test_fallback_has_label(self):
        out = extract_topics("Just one tiny line.")
        assert out["topics"][0]["label"] in {"one", "tiny", "line"}

    def test_vietnamese_doc_labels(self):
        vi = ("\n\n".join([
            "Công ty báo cáo doanh thu tăng trưởng ổn định trong quý.",
            "Doanh thu đến từ logistics và tự động hóa kho vận hành.",
            "Tự động hóa giúp giảm thời gian xử lý đơn hàng đáng kể.",
        ]))
        out = extract_topics(vi, k=2)
        assert all(t["label"] for t in out["topics"])


class TestUMass:
    def test_cap_never_exceeds_bounds(self):
        assert n_topics_cap(100, 10000) == 10
        assert n_topics_cap(4, 500) >= 3

    def test_umass_finite_on_tiny_dtm(self):
        from sklearn.feature_extraction.text import CountVectorizer

        docs = ["apple banana apple", "banana cherry banana", "apple date"]
        dtm = CountVectorizer().fit_transform(docs)
        score = umass_coherence(dtm, [[0, 1], [1, 2]])
        import math

        assert math.isfinite(score)
