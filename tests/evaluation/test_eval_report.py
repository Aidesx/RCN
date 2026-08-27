"""Tests for the evaluation report module — fake predictors only, no TF."""
from pathlib import Path

import numpy as np
import pytest

from docproc.evaluation.report import report_run, save_confusion_csv


@pytest.fixture(scope="module")
def real_test_arrays():
    from docproc.training.data import load_split_arrays

    return load_split_arrays("test", arm="cnn")


class TestReportRun:
    def test_majority_predictor_writes_artifacts_gate_fails(self, real_test_arrays, tmp_path):
        X, y = real_test_arrays
        names_idx = int(np.bincount(y).argmax())

        def predict_fn(Xb):
            return np.full(len(Xb), names_idx, dtype=np.int64)

        out = report_run(tmp_path, predict_fn)
        assert (tmp_path / "metrics_eval.json").is_file()
        assert (tmp_path / "confusion_matrix.csv").is_file()
        gate = out["acceptance_gate_04_8"]
        assert gate["pass"] is False
        assert gate["margin_over_baseline"] == pytest.approx(0.0, abs=1e-9)

    def test_oracle_predictor_passes_gate(self, real_test_arrays, tmp_path):
        X, y = real_test_arrays

        def predict_fn(Xb):
            return np.asarray(y)[: len(Xb)]

        out = report_run(tmp_path, predict_fn)
        gate = out["acceptance_gate_04_8"]
        assert gate["accuracy"] == pytest.approx(1.0)
        assert gate["macro_f1"] == pytest.approx(1.0)
        assert gate["pass"] is True

    def test_curves_png_written_when_history_exists(self, real_test_arrays, tmp_path):
        X, y = real_test_arrays
        (tmp_path / "history.csv").write_text(
            "epoch,loss,accuracy,val_loss,val_accuracy\n"
            "1,2.0,0.2,1.9,0.25\n2,1.5,0.4,1.6,0.35\n",
            encoding="utf-8",
        )

        def predict_fn(Xb):
            return np.asarray(y)[: len(Xb)]

        report_run(tmp_path, predict_fn)
        assert (tmp_path / "learning_curves.png").is_file()


class TestConfusionCsv:
    def test_header_and_rows(self, tmp_path):
        cm = [[1, 2], [3, 4]]
        save_confusion_csv(cm, tmp_path / "cm.csv", ["a", "b"])
        lines = (tmp_path / "cm.csv").read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "true\\pred,a,b"
        assert lines[1] == "a,1,2" and lines[2] == "b,3,4"


class TestRealRunReproducible:
    def test_e1_report_matches_recorded_metrics(self, real_test_arrays, tmp_path):
        """Deepened report must reproduce the frozen E1 numbers bit-for-bit."""
        import json

        from docproc import paths

        run_dir = paths.RUNS_DIR / "E1"
        recorded = json.loads((run_dir / "metrics_eval.json").read_text(encoding="utf-8"))

        import tensorflow as tf

        model = tf.keras.models.load_model(run_dir / "best.keras")

        def predict_fn(Xb):
            return np.argmax(model.predict(Xb, verbose=0), axis=1)

        out = report_run(tmp_path, predict_fn)
        old = recorded["model_metrics"] if "model_metrics" in recorded else recorded["e1"]
        assert out["model_metrics"]["accuracy"] == old["accuracy"]
        assert out["model_metrics"]["confusion_matrix"] == old["confusion_matrix"]
        assert out["acceptance_gate_04_8"]["pass"] is True