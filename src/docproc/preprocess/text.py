"""TF-IDF vectorization wrapper; params from configs/text.yaml; joblib save/load."""
from __future__ import annotations

from pathlib import Path

from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

from docproc import paths


class TextVectorizer:
    def __init__(self, config_path: Path | None = None):
        cfg = (
            paths.load_config("text")
            if config_path is None
            else paths.load_yaml_file(config_path)
        )["vectorizer"]["tfidf"]
        self.vectorizer = TfidfVectorizer(
            max_features=int(cfg["max_features"]),
            ngram_range=tuple(cfg["ngram_range"]),
            sublinear_tf=bool(cfg["sublinear_tf"]),
            stop_words=cfg["stop_words"],
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "TextVectorizer":
        if not texts:
            raise ValueError("cannot fit a vectorizer on an empty corpus")
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: list[str]):
        if not self._fitted:
            raise RuntimeError("vectorizer must be fit before transform")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        self.fit(texts)
        return self.transform(texts)

    def save(self, path) -> None:
        dump(self, path)

    @staticmethod
    def load(path) -> "TextVectorizer":
        return load(path)