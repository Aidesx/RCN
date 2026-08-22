"""evaluation package: metrics, baselines, acceptance gates."""
from docproc.evaluation.metrics import (
    acceptance_gate,
    compute_metrics,
    majority_baseline_metrics,
    majority_class_index,
)

__all__ = [
    "acceptance_gate",
    "compute_metrics",
    "majority_baseline_metrics",
    "majority_class_index",
]