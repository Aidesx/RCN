"""Training harness: seeds, callbacks, run records (config + history + metrics).

Every run writes a self-contained record under runs/<run_name>/:
- config.yaml   : resolved training configuration snapshot
- history.csv   : per-epoch train/val metrics
- metrics.json  : final summary (best epoch, best val_accuracy, test-free)
- best.keras    : checkpoint with the best validation accuracy weights
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def _callbacks(cfg: dict, run_dir: Path):
    import tensorflow as tf

    t = cfg["training"]
    es_cfg = t.get("early_stopping", {})
    ckpt_cfg = t.get("checkpoint", {})
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=es_cfg.get("monitor", "val_accuracy"),
            patience=int(es_cfg.get("patience", 6)),
            restore_best_weights=bool(es_cfg.get("restore_best_weights", True)),
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best.keras"),
            monitor=ckpt_cfg.get("monitor", "val_accuracy"),
            mode=ckpt_cfg.get("mode", "max"),
            save_best_only=bool(ckpt_cfg.get("save_best_only", True)),
            verbose=0,
        ),
    ]
    sched = t.get("scheduler", {})
    if sched.get("enabled"):
        r = sched["reduce_lr_on_plateau"]
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=float(r["factor"]), patience=int(r["patience"])
            )
        )
    return callbacks


def _write_history(run_dir: Path, history) -> None:
    hist = history.history
    keys = list(hist.keys())
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(len(hist[keys[0]])):
            w.writerow([i + 1] + [hist[k][i] for k in keys])


def run_training(model, train_ds, val_ds, cfg: dict, run_name: str,
                 extra_config: dict | None = None, run_dir: Path | None = None):
    """Fit the model per config; write the run record; return (history, metrics)."""
    import yaml

    from docproc import paths

    if run_dir is None:
        run_dir = paths.RUNS_DIR / run_name
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(int(cfg["training"]["seed"]))

    snapshot = {"training": cfg["training"], "extra": extra_config or {}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot), encoding="utf-8")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["training"]["max_epochs"]),
        callbacks=_callbacks(cfg, run_dir),
        verbose=2,
    )
    _write_history(run_dir, history)

    val_acc = history.history.get("val_accuracy", [])
    best_epoch = int(np.argmax(val_acc)) + 1 if val_acc else None
    metrics = {
        "run_name": run_name,
        "epochs_run": len(history.history.get("loss", [])),
        "best_val_accuracy": max(val_acc) if val_acc else None,
        "best_epoch": best_epoch,
        "final_train_loss": history.history.get("loss", [None])[-1],
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return history, metrics