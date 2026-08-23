"""Stage 8 / E0b: train + evaluate the TF-IDF+SVM/RF text baseline.

Usage: python scripts/run_text_baseline.py
Writes: models/artifacts/text_*.{joblib}, runs/E0b/metrics_eval.json,
        runs/E0b/confusion_matrix.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc import paths  # noqa: E402
from docproc.evaluation.report import save_confusion_csv  # noqa: E402
from docproc.models.text_classifier import evaluate_baseline, train_baseline  # noqa: E402


def main() -> int:
    meta, est, vec = train_baseline()
    out = evaluate_baseline(est, vec)

    run_dir = paths.RUNS_DIR / "E0b"
    run_dir.mkdir(parents=True, exist_ok=True)
    out["cv"] = meta
    (run_dir / "metrics_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_confusion_csv(out["e0b"]["confusion_matrix"], run_dir / "confusion_matrix.csv",
                       paths.class_names())

    print("CV results:")
    for c in meta["candidates"]:
        print(f"  {c['model']:<14} f1_macro={c['cv_f1_macro']:.4f}  {c['best_params']}")
    g = out["acceptance_gate_04_8"]
    print(json.dumps(g, indent=2))
    return 0 if g["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())