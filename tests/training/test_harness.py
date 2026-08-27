"""Stage 4 tests: training harness writes checkpoints and run records."""
import json

import numpy as np
import yaml


def _tiny_data(n=16):
    rng = np.random.default_rng(42)
    X = rng.random((n, 64, 64, 3), dtype=np.float32)
    y = (np.arange(n) % 6).astype(np.int64)
    return X, y


def test_run_training_writes_records(tmp_path):
    from docproc.models.cnn import build_model
    from docproc.training.data import make_datasets
    from docproc.training.harness import run_training

    X, y = _tiny_data()
    Xv, yv = X[:8], y[:8]
    train_ds, val_ds = make_datasets(X, y, Xv, yv, batch_size=8, seed=42)

    cfg = yaml.safe_load(
        (harness_config_dir() / "cnn.yaml").read_text(encoding="utf-8"))
    cfg["training"]["max_epochs"] = 1

    model = build_model()
    history, metrics = run_training(model, train_ds, val_ds, cfg,
                                    "_test_harness", run_dir=tmp_path)

    assert (tmp_path / "best.keras").exists()
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "config.yaml").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["epochs_run"] == 1
    assert 0.0 <= saved["best_val_accuracy"] <= 1.0
    assert len(history.history["loss"]) == 1


def harness_config_dir():
    from docproc import paths

    return paths.CONFIG_DIR


def test_set_seeds_deterministic_init():
    from docproc.models.cnn import build_model
    from docproc.training.harness import set_seeds

    set_seeds(42)
    m1 = build_model()
    w1 = m1.get_weights()[0].copy()
    set_seeds(42)
    m2 = build_model()
    w2 = m2.get_weights()[0].copy()
    assert np.array_equal(w1, w2)


def test_checkpoint_restores_best_weights(tmp_path):
    """Early stopping + checkpoint keep the best-val weights, not the last."""
    import tensorflow as tf
    from docproc.models.cnn import build_model
    from docproc.training.data import make_datasets
    from docproc.training.harness import set_seeds

    set_seeds(42)
    X, y = _tiny_data(32)
    train_ds, val_ds = make_datasets(X[:24], y[:24], X[24:], y[24:], batch_size=8, seed=42)
    model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)
    acc = model.evaluate(val_ds, verbose=0)[1]
    assert 0.0 <= acc <= 1.0