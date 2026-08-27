"""Stage 8 fast tests: text baseline (E0b) — tiny tmp corpora, no big data."""
import csv
import json

import numpy as np
import pytest

from docproc import paths
from docproc.models.text_classifier import (
    _make_estimators,
    evaluate_baseline,
    load_split_texts,
    read_text_manifest,
    train_baseline,
)
from docproc.preprocess.text import TextVectorizer

CLASSES = ["letter", "receipt"]


MARKERS = {
    "letter": "Dear friend, thank you for your kind letter regarding",
    "receipt": "SALES RECEIPT TOTAL CASH THANK YOU",
    "invoice": "INVOICE NUMBER TOTAL DUE PAYMENT TERMS NET",
    "report": "QUARTERLY PERFORMANCE REPORT ABSTRACT FINDINGS",
    "form": "APPLICATION FORM REFERENCE SIGNATURE SECTION APPROVED",
    "article": "CITY COUNCIL PUBLISHED REPORTER RESIDENTS REACTED",
}


def _tiny_corpus(tmp_path):
    """42 docs across ALL 6 classes; per-class split 5/1/1 so StratifiedKFold(5)
    works and macro-F1 spans every project label."""
    texts_dir = tmp_path / "text"
    rows = []
    i = 0
    for cls, marker in MARKERS.items():
        d = texts_dir / cls
        d.mkdir(parents=True)
        for j in range(7):
            doc_id = f"{cls}_{j}"
            p = d / f"{doc_id}.txt"
            p.write_text(f"{marker} item {i} reference {1000+i}.", encoding="utf-8")
            rows.append((cls, doc_id, p.name))
            i += 1
    man = tmp_path / "text_manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "class", "file_name", "split"])
        for cls, doc_id, fname in rows:
            j = int(doc_id.rsplit("_", 1)[1])
            split = "train" if j < 5 else "validation" if j == 5 else "test"
            w.writerow([doc_id, cls, fname, split])
    return man, texts_dir


class TestManifest:
    def test_real_manifest_counts(self):
        assert len(read_text_manifest("train")) == 252
        assert len(read_text_manifest("test")) == 54

    def test_load_split_texts_aligned(self):
        texts, y = load_split_texts("test")
        assert len(texts) == len(y) == 54
        assert min(y) >= 0 and max(y) <= 5
        assert all(isinstance(t, str) and t for t in texts)


class TestEstimators:
    def test_both_model_families_configured(self):
        items = _make_estimators(paths.load_config("text"))
        assert [n for n, _, _ in items] == ["svm", "random_forest"]


class TestEvaluateBaseline:
    def _vecs(self, tmp_path):
        man, text_root = _tiny_corpus(tmp_path)
        Xtr, ytr = load_split_texts("train", man, text_root)
        vec = TextVectorizer().fit(Xtr)
        return man, text_root, np.asarray(ytr)

    def test_majority_predictor_fails_gate(self, tmp_path):
        man, text_root, ytr = self._vecs(tmp_path)
        names = paths.class_names()
        counts = {n: int((ytr == i).sum()) for i, n in enumerate(names)}
        from docproc.evaluation.metrics import majority_class_index_from_counts
        maj = majority_class_index_from_counts(counts, names)

        class AlwaysMaj:
            def predict(self, X):
                return np.full(X.shape[0], maj)

        out = evaluate_baseline(AlwaysMaj(), TextVectorizer().fit(
            load_split_texts("train", man, text_root)[0]), man, text_root)
        assert out["acceptance_gate_04_8"]["pass"] is False

    def test_oracle_passes_gate(self, tmp_path):
        man, text_root, _ = self._vecs(tmp_path)
        Xte, yte = load_split_texts("test", man, text_root)
        truth = np.asarray(yte)

        class Oracle:
            def predict(self, X):
                return truth[: X.shape[0]]

        vec = TextVectorizer().fit(load_split_texts("train", man, text_root)[0])
        out = evaluate_baseline(Oracle(), vec, man, text_root)
        g = out["acceptance_gate_04_8"]
        assert g["accuracy"] == pytest.approx(1.0) and g["pass"] is True


class TestTrainBaselineEndToEnd:
    def test_fit_artifacts_and_eval(self, tmp_path):
        man, text_root = _tiny_corpus(tmp_path)
        art = tmp_path / "artifacts"
        meta, est, vec = train_baseline(manifest_path=man, artifacts_dir=art,
                                        text_root=text_root)
        assert meta["model"] in ("svm", "random_forest")
        assert len(meta["candidates"]) == 2
        assert (art / "text_vectorizer.joblib").is_file()
        assert (art / f"text_model_{meta['model']}.joblib").is_file()

        out = evaluate_baseline(est, vec, man, text_root)
        assert set(out) == {"e0b", "majority_baseline", "acceptance_gate_04_8"}
        # tiny templated corpus is linearly separable -> near-perfect
        assert out["e0b"]["macro_f1"] >= 0.99

    def test_vectorizer_roundtrip_after_train(self, tmp_path):
        man, text_root = _tiny_corpus(tmp_path)
        _, _, vec = train_baseline(manifest_path=man, artifacts_dir=tmp_path / "a",
                                   text_root=text_root)
        loaded = TextVectorizer.load(tmp_path / "a" / "text_vectorizer.joblib")
        texts, _ = load_split_texts("test", man, text_root)
        a = vec.transform(texts)
        b = loaded.transform(texts)
        assert (a != b).nnz == 0