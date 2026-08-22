"""TF-IDF vectorization wrapper (text branch, E0b baseline).

Deterministic wrapper over sklearn's TfidfVectorizer; hyperparameters come
from configs/text.yaml (no hard-coding). Provides fit/transform/save/load
with joblib so the trained vectorizer can be reused at inference time.
"""
from __future__ import annotations

from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

from docproc import paths


class TextVectorizer:
    def __init__(self, config_path: Path | None = None):
        cfg = (
            paths.load_config("text")
            if config_path is None
            else _read_yaml(config_path)
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


def _read_yaml(p) -> dict:
    import yaml
    from pathlib import Path

    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))