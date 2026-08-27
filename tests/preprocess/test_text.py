"""Stage 3 tests: TF-IDF text vectorization wrapper."""
import numpy as np
import pytest

from docproc.preprocess.text import TextVectorizer

CORPUS = [
    "invoice number 1234 total amount due",
    "receipt thank you for your purchase",
    "quarterly report on sales figures",
    "official letter from the department",
    "form to be filled by the applicant",
]

EMPTY = ["", "   ", "\n"]


class TestFitTransform:
    def test_fit_transform_shape(self):
        vec = TextVectorizer()
        m = vec.fit_transform(CORPUS)
        assert m.shape == (len(CORPUS), m.shape[1])
        assert m.shape[1] > 0
        assert np.isfinite(m.data).all()

    def test_transform_after_fit_deterministic(self):
        vec = TextVectorizer().fit(CORPUS)
        a = vec.transform(CORPUS)
        b = vec.transform(CORPUS)
        assert (a != b).nnz == 0
        assert a.shape == b.shape

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            TextVectorizer().transform(CORPUS)

    def test_fit_on_empty_corpus_raises(self):
        with pytest.raises(ValueError):
            TextVectorizer().fit([])


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        vec = TextVectorizer().fit(CORPUS)
        p = tmp_path / "vec.joblib"
        vec.save(p)
        loaded = TextVectorizer.load(p)
        a = vec.transform(CORPUS)
        b = loaded.transform(CORPUS)
        assert (a != b).nnz == 0


class TestConfig:
    def test_uses_config_ngram_range(self):
        vec = TextVectorizer()
        assert vec.vectorizer.ngram_range == (1, 2)

    def test_stop_words_from_config(self):
        assert TextVectorizer().vectorizer.get_params()["stop_words"] == "english"