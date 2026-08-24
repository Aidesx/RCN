"""Evaluation reporting: run a predictor on the frozen test split and write
the full artifact set (metrics_eval.json, confusion_matrix.csv,
learning_curves.png) plus the 04 §8 acceptance-gate verdict.

Pure orchestration — accepts any predict_fn (X -> y_pred), so tests exercise
this module without TensorFlow.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from docproc import paths
from docproc.evaluation.metrics import (
    acceptance_gate,
    compute_metrics,
    majority_class_index_from_counts,
)
from docproc.training.data import load_split_arrays, split_class_counts


def save_confusion_csv(cm: list[list[int]], path: Path, names: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + names)
        for name, row in zip(names, cm):
            w.writerow([name] + row)


def save_curves(history_csv: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(history_csv, encoding="utf-8")))
    if not rows:
        return  # nothing to plot — avoid IndexError on rows[0]
    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(r["loss"]) for r in rows], label="train")
    if "val_loss" in rows[0]:
        axes[0].plot(epochs, [float(r["val_loss"]) for r in rows], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [float(r["accuracy"]) for r in rows], label="train")
    if "val_accuracy" in rows[0]:
        axes[1].plot(epochs, [float(r["val_accuracy"]) for r in rows], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def report_run(run_dir: Path, predict_fn, split: str = "test", arm: str = "cnn") -> dict:
    """Evaluate predict_fn on the frozen split; write artifacts; return summary.

    The majority baseline uses per-class counts of the training manifest rows
    (no image decoding needed).
    """
    run_dir = Path(run_dir)
    names = paths.class_names()

    X_test, y_test = load_split_arrays(split, arm=arm)
    y_pred = np.asarray(predict_fn(X_test))
    metrics = compute_metrics(y_test, y_pred, names)

    train_counts = split_class_counts("train")
    base_idx = majority_class_index_from_counts(train_counts, names)
    y_base = np.full_like(y_test, base_idx)
    baseline = compute_metrics(y_test, y_base, names)
    baseline["baseline_class"] = names[base_idx]

    gate = acceptance_gate(metrics, baseline)

    out = {
        "run_name": run_dir.name,
        "test_pages": int(len(y_test)),
        "model_metrics": metrics,
        "majority_baseline": baseline,
        "acceptance_gate_04_8": gate,
    }
    (run_dir / "metrics_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_confusion_csv(metrics["confusion_matrix"], run_dir / "confusion_matrix.csv", names)
    history_csv = run_dir / "history.csv"
    if history_csv.is_file():
        save_curves(history_csv, run_dir / "learning_curves.png")
    return out