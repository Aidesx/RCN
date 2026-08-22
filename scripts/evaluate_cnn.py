"""Evaluate a trained CNN run on the frozen test split (thin CLI adapter).

Usage: python scripts/evaluate_cnn.py --run-name E1 [--arm cnn|finetune]
Artifacts are written by docproc.evaluation.report.report_run.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc import paths  # noqa: E402
from docproc.evaluation.report import report_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="E1")
    ap.add_argument("--arm", default="cnn", choices=["cnn", "finetune"])
    args = ap.parse_args()

    run_dir = paths.RUNS_DIR / args.run_name
    if not (run_dir / "best.keras").is_file():
        print(f"no checkpoint at {run_dir / 'best.keras'}")
        return 2

    import tensorflow as tf

    model = tf.keras.models.load_model(run_dir / "best.keras")

    def predict_fn(X):
        return np.argmax(model.predict(X, verbose=0), axis=1)

    out = report_run(run_dir, predict_fn, arm=args.arm)
    gate = out["acceptance_gate_04_8"]
    print(json.dumps(gate, indent=2))
    cm = out["e1"]["confusion_matrix"]
    names = paths.class_names()
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("            " + "  ".join(f"{n:>9}" for n in names))
    for n, row in zip(names, cm):
        print(f"{n:>11} " + "  ".join(f"{v:>9}" for v in row))
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())