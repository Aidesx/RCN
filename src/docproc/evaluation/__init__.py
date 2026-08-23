"""evaluation package: metrics, baselines, acceptance gates."""
from docproc.evaluation.metrics import (
    acceptance_gate,
    compute_metrics,
    majority_class_index,
    majority_class_index_from_counts,
)

__all__ = [
    "acceptance_gate",
    "compute_metrics",
    "majority_class_index",
    "majority_class_index_from_counts",
]