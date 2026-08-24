"""Train the self-built CNN (Architecture A) on the manifest splits.

Smoke run (Stage 4 gate):  python scripts/run_cnn.py --run-name E1_smoke --fraction 0.3 --epochs 3
Full E1 (Stage 5):         python scripts/run_cnn.py --run-name E1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc import paths  # noqa: E402
from docproc.models.cnn import build_model  # noqa: E402
from docproc.training.data import load_split_arrays, make_datasets  # noqa: E402
from docproc.training.harness import run_training  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="E1")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="deterministic slice: first N%% of train rows (manifest order)")
    ap.add_argument("--epochs", type=int, default=None, help="override max_epochs")
    args = ap.parse_args()

    cfg = paths.load_config("cnn")
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    batch_size = int(cfg["training"]["batch_size"])
    seed = int(cfg["training"]["seed"])

    X_train, y_train = load_split_arrays("train")
    if args.fraction < 1.0:
        n = max(1, int(len(y_train) * args.fraction))
        X_train, y_train = X_train[:n], y_train[:n]
    X_val, y_val = load_split_arrays("validation")

    train_ds, val_ds = make_datasets(X_train, y_train, X_val, y_val,
                                     batch_size=batch_size, seed=seed)

    model = build_model()
    extra = {
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "fraction": args.fraction,
    }
    print(f"[{args.run_name}] train={len(y_train)} val={len(y_val)} batch={batch_size}")
    history, metrics = run_training(model, train_ds, val_ds, cfg, args.run_name, extra_config=extra)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())