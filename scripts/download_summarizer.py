"""Download the small (<7B) multilingual abstractive summarizer checkpoint.

Usage:
  python scripts/download_summarizer.py

Fetches configs/summary.yaml -> abstractive.checkpoint from HuggingFace into
models/artifacts/summarizer_mt5/ so summarize_abstractive() runs offline.
Prefers safetensors; falls back to pytorch_model.bin when unavailable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc import paths

_BASE_PATTERNS = ["*.json", "*.model", "*.txt"]
_WEIGHT_PATTERNS = ["model.safetensors", "pytorch_model.bin"]


def _has_weights(d: Path) -> bool:
    return any((d / w).exists() for w in _WEIGHT_PATTERNS)


def main() -> int:
    target = paths.ROOT / "models" / "artifacts" / "summarizer_mt5"
    target.mkdir(parents=True, exist_ok=True)
    repo = paths.load_config("summary")["abstractive"]["checkpoint"]

    from huggingface_hub import snapshot_download

    for weights in _WEIGHT_PATTERNS:
        if _has_weights(target):
            break
        print(f"downloading {repo} ({weights}) ...")
        snapshot_download(repo_id=repo, local_dir=target,
                          allow_patterns=_BASE_PATTERNS + [weights])
    if not _has_weights(target):
        print("error: no weight file found in repo", file=sys.stderr)
        return 2
    print(f"ok -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
