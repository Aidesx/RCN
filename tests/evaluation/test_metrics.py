"""Stage 6 tests: metrics + majority baseline + acceptance gate."""
import numpy as np

from docproc.evaluation.metrics import (
    acceptance_gate,
    compute_metrics,
    majority_class_index,
    majority_class_index_from_counts,
)

NAMES = ["a", "b", "c"]


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1])
        m = compute_metrics(y, y.copy(), NAMES)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert sum(sum(r) for r in m["confusion_matrix"]) == len(y)

    def test_known_accuracy_and_confusion(self):
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 2])
        m = compute_metrics(y_true, y_pred, NAMES)
        assert m["accuracy"] == pytest_approx(0.8)
        cm = m["confusion_matrix"]
        assert cm[0][0] == 1 and cm[0][1] == 1 and cm[1][1] == 2 and cm[2][2] == 1

    def test_macro_f1_robust_to_imbalance(self):
        y_true = np.array([0] * 9 + [1])
        y_pred = np.array([0] * 9 + [0])
        m = compute_metrics(y_true, y_pred, ["a", "b"])
        assert m["accuracy"] == pytest_approx(0.9)
        assert m["macro_f1"] < 0.6


class TestMajorityBaseline:
    def test_majority_class_index(self):
        y_train = np.array([0, 0, 0, 1, 1, 2])
        assert majority_class_index(y_train) == 0

    def test_tie_breaks_deterministically(self):
        y_train = np.array([0, 0, 1, 1])
        idx = majority_class_index(y_train)
        assert idx in (0, 1)

    def test_baseline_predicts_majority_everywhere(self):
        counts = {"a": 1, "b": 1, "c": 3}
        idx = majority_class_index_from_counts(counts, NAMES)
        y_test = np.array([0, 1, 2])
        m = compute_metrics(y_test, np.full_like(y_test, idx), NAMES)
        assert NAMES[idx] == "c"
        assert m["accuracy"] == pytest_approx(1 / 3)

    def test_from_counts_tie_break_by_fixed_order(self):
        assert majority_class_index_from_counts({"a": 2, "b": 2}, NAMES) == 0
        assert majority_class_index_from_counts({}, NAMES) == 0


class TestAcceptanceGate:
    def test_gate_pass(self):
        e1 = {"accuracy": 0.45, "macro_f1": 0.55}
        base = {"accuracy": 0.28}
        assert acceptance_gate(e1, base)["pass"] is True

    def test_gate_fail_on_margin(self):
        e1 = {"accuracy": 0.35, "macro_f1": 0.55}
        base = {"accuracy": 0.28}
        g = acceptance_gate(e1, base)
        assert g["pass"] is False and g["margin_over_baseline"] < 0.10

    def test_gate_fail_on_macro_f1(self):
        e1 = {"accuracy": 0.50, "macro_f1": 0.40}
        base = {"accuracy": 0.28}
        assert acceptance_gate(e1, base)["pass"] is False


def pytest_approx(x):
    import pytest

    return pytest.approx(x)