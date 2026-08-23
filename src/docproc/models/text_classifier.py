"""Text baseline (E0b): TF-IDF + SVM/RF with GridSearchCV (04 §6 E0, text.yaml).

Owns: loading text documents per manifest split, fitting/tuning both model
families via 5-fold CV on f1_macro, and persisting artifacts. Evaluation
reporting reuses the same artifact conventions as the image arm.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from docproc import paths
from docproc.io.parsers import extract_text
from docproc.preprocess.text import TextVectorizer


def read_text_manifest(split: str | None = None,
                       manifest_path: Path | None = None) -> list[dict]:
    p = Path(manifest_path) if manifest_path else paths.SPLITS_DIR / "text_manifest.csv"
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split is not None:
        rows = [r for r in rows if r["split"] == split]
    if not rows:
        raise ValueError(f"no rows for split '{split}' in {p}")
    return rows


def _default_text_root() -> Path:
    return paths.DATASETS_DIR / "text"


def load_split_texts(split: str, manifest_path: Path | None = None,
                     text_root: Path | None = None):
    """Return (texts, labels) for one split; classes ordered per dataset.yaml."""
    names = paths.class_names()
    root = Path(text_root) if text_root else _default_text_root()
    texts, labels = [], []
    for r in read_text_manifest(split, manifest_path):
        page = root / r["class"] / r["file_name"]
        texts.append(extract_text(page))
        labels.append(names.index(r["class"]))
    return texts, labels


def _make_estimators(cfg: dict):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC

    factories = {
        "linear_svc": lambda params: LinearSVC(random_state=int(
            cfg["cross_validation"]["seed"]), **params),
        "random_forest": lambda params: RandomForestClassifier(
            random_state=int(cfg["cross_validation"]["seed"]), **params),
    }
    return [(name, factories[cfg["models"][name]["class"]](dict()),
             cfg["models"][name]["param_grid"])
            for name in cfg["models"]]


def train_baseline(manifest_path: Path | None = None,
                   artifacts_dir: Path | None = None,
                   text_root: Path | None = None):
    """GridSearchCV over configured models; returns (best_pipeline_meta, cv_rows)."""
    from sklearn.model_selection import GridSearchCV

    cfg = paths.load_config("text")
    cv = int(cfg["cross_validation"]["folds"])
    scoring = cfg["cross_validation"]["scoring"]

    X_train, y_train = load_split_texts("train", manifest_path, text_root)
    vec = TextVectorizer()
    Xv = vec.fit_transform(X_train)

    results = []
    for name, est, grid in _make_estimators(cfg):
        gs = GridSearchCV(est, grid, cv=cv, scoring=scoring, n_jobs=-1)
        gs.fit(Xv, y_train)
        results.append({
            "model": name,
            "best_params": gs.best_params_,
            "cv_f1_macro": float(gs.best_score_),
            "estimator": gs.best_estimator_,
        })

    best = max(results, key=lambda r: r["cv_f1_macro"])
    out_dir = artifacts_dir or (paths.ROOT / "models" / "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    vec.save(out_dir / "text_vectorizer.joblib")
    import joblib

    joblib.dump(best["estimator"], out_dir / f"text_model_{best['model']}.joblib")

    meta = {k: v for k, v in best.items() if k != "estimator"}
    meta["candidates"] = [{k: v for k, v in r.items() if k != "estimator"} for r in results]
    return meta, best["estimator"], vec


def evaluate_baseline(estimator, vec: TextVectorizer,
                      manifest_path: Path | None = None,
                      text_root: Path | None = None) -> dict:
    from docproc.evaluation.metrics import (
        acceptance_gate, compute_metrics, majority_class_index_from_counts)

    names = paths.class_names()
    X_test, y_test = load_split_texts("test", manifest_path, text_root)
    y_pred = estimator.predict(vec.transform(X_test))
    metrics = compute_metrics(y_test, y_pred, names)

    counts = {}
    for r in read_text_manifest("train", manifest_path):
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    base_idx = majority_class_index_from_counts(counts, names)
    base_pred = [base_idx] * len(y_test)
    baseline = compute_metrics(np.asarray(y_test), np.asarray(base_pred), names)
    baseline["baseline_class"] = names[base_idx]
    gate = acceptance_gate(metrics, baseline)
    return {"e0b": metrics, "majority_baseline": baseline,
            "acceptance_gate_04_8": gate}