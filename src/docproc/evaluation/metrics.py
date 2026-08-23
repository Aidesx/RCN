"""Metrics, confusion matrix, majority baseline helpers, acceptance gate."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def majority_class_index(y_train: np.ndarray) -> int:
    counts = Counter(y_train.tolist())
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def majority_class_index_from_counts(counts_by_name: dict[str, int],
                                     names: list[str]) -> int:
    """Majority class index from per-class name counts (deterministic tie-break
    by fixed class order). Lets callers skip decoding the whole train split."""
    best, best_n = 0, -1
    for i, n in enumerate(names):
        c = counts_by_name.get(n, 0)
        if c > best_n:
            best, best_n = i, c
    return best


def acceptance_gate(e1: dict, baseline: dict, min_margin: float = 0.10,
                    min_macro_f1: float = 0.50) -> dict:
    """04 §8 minimum model quality: E1 >= baseline + 10 pts AND macro-F1 >= 0.50."""
    margin = e1["accuracy"] - baseline["accuracy"]
    return {
        "accuracy": e1["accuracy"],
        "baseline_accuracy": baseline["accuracy"],
        "margin_over_baseline": margin,
        "macro_f1": e1["macro_f1"],
        "required_margin": min_margin,
        "required_macro_f1": min_macro_f1,
        "pass": bool(margin >= min_margin and e1["macro_f1"] >= min_macro_f1),
    }