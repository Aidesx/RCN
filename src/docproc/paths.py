"""Single owner of repo-layout knowledge: roots, dirs, config loading."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"
DATASETS_DIR = ROOT / "datasets"
SPLITS_DIR = DATASETS_DIR / "splits"
RAW_DIR = DATASETS_DIR / "raw"
TESTS_DIR = ROOT.parent / "rcn-tests"  # suite kept outside the repo (user decision, 06 v2.7)


def load_yaml_file(path) -> dict:
    """Load any YAML file by explicit path (configs are the only YAML we ship)."""
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_config(name: str) -> dict:
    return load_yaml_file(CONFIG_DIR / f"{name}.yaml")


def dataset_config() -> dict:
    return load_config("dataset")


def pipeline_config() -> dict:
    return load_config("pipeline")


def class_names() -> list[str]:
    """Fixed project class order (letter=0 .. receipt=5) per configs/dataset.yaml."""
    return list(dataset_config()["classes"])