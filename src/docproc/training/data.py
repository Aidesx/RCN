"""Dataset access: manifest ownership + model-agnostic tensor loading.

The ARMS registry maps each model arm to its input config and tensor
function (from the tested preprocess path), so one loader serves both the
64x64 CNN and the 224x224 MobileNetV2 fine-tune arm. Manifest rows are
validated once here instead of in every consumer.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from docproc import paths
from docproc.preprocess.image import cnn_tensor, finetune_tensor

REQUIRED_COLUMNS = {"doc_id", "class", "page_file", "source", "split"}

ARMS: dict[str, dict] = {
    "cnn": {"config": "cnn", "tensor_fn": cnn_tensor},
    "finetune": {"config": "finetune", "tensor_fn": finetune_tensor},
}


def _arm_size(arm: str) -> tuple[int, int]:
    cfg = paths.load_config(ARMS[arm]["config"])
    return int(cfg["input"]["height"]), int(cfg["input"]["width"])


def read_manifest(split: str | None = None, manifest_path: Path | None = None) -> list[dict]:
    """Read manifest rows (optionally one split), validating the schema."""
    p = Path(manifest_path) if manifest_path else paths.SPLITS_DIR / "manifest.csv"
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest {p} missing columns: {sorted(missing)}")
        rows = list(reader)
    if split is not None:
        rows = [r for r in rows if r["split"] == split]
    return rows


def split_class_counts(split: str) -> dict[str, int]:
    return dict(Counter(r["class"] for r in read_manifest(split=split)))


def load_split_arrays(split: str, arm: str = "cnn",
                      manifest_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load one manifest split as (X, y) tensors for the given model arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm '{arm}'; available: {sorted(ARMS)}")
    rows = read_manifest(split=split, manifest_path=manifest_path)
    if not rows:
        raise ValueError(f"no rows for split '{split}'")
    classes = paths.class_names()
    h, w = _arm_size(arm)
    tensor_fn = ARMS[arm]["tensor_fn"]
    X = np.zeros((len(rows), h, w, 3), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        page = Path(r["page_file"])
        if not page.is_absolute():
            page = paths.ROOT / page
        X[i] = tensor_fn(page)
        y[i] = classes.index(r["class"])
    return X, y


def make_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    seed: int = 42,
):
    """Wrap arrays in tf.data pipelines: shuffled train, plain validation."""
    import tensorflow as tf

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(y_train), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    return train_ds, val_ds