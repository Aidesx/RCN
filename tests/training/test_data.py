"""Stage 4 tests: manifest -> tensor arrays and tf.data datasets."""
import numpy as np
import pytest

from docproc.training.data import load_split_arrays, make_datasets, split_class_counts

EXPECTED_TRAIN = {"letter": 70, "form": 70, "report": 70,
                  "article": 70, "invoice": 70, "receipt": 140}


@pytest.fixture(scope="module")
def train_arrays():
    return load_split_arrays("train")


class TestSplitArrays:
    def test_shapes_and_dtypes(self, train_arrays):
        X, y = train_arrays
        assert X.shape == (sum(EXPECTED_TRAIN.values()), 64, 64, 3)
        assert X.dtype == np.float32
        assert y.dtype == np.int64

    def test_value_range_normalized(self, train_arrays):
        X, _ = train_arrays
        assert X.min() >= 0.0 and X.max() <= 1.0

    def test_labels_in_range(self, train_arrays):
        _, y = train_arrays
        assert y.min() >= 0 and y.max() <= 5

    def test_per_class_counts_match_split_stats(self, train_arrays):
        X, y = train_arrays
        counts = {i: int((y == i).sum()) for i in range(6)}
        expected = [EXPECTED_TRAIN[c] for c in
                    ["letter", "form", "report", "article", "invoice", "receipt"]]
        assert counts == dict(zip(range(6), expected))

    def test_split_class_counts_helper(self):
        assert split_class_counts("train") == EXPECTED_TRAIN

    def test_unknown_split_raises(self):
        with pytest.raises(ValueError):
            load_split_arrays("dev")


class TestMakeDatasets:
    def test_batch_shapes(self):
        rng = np.random.default_rng(42)
        X = rng.random((10, 64, 64, 3), dtype=np.float32)
        y = rng.integers(0, 6, size=10).astype(np.int64)
        Xv, yv = X[:4], y[:4]
        train_ds, val_ds = make_datasets(X, y, Xv, yv, batch_size=4, seed=42)
        xb, yb = next(iter(train_ds))
        assert tuple(xb.shape) == (4, 64, 64, 3)
        assert int(np.max(yb.numpy())) < 6
        vx, vy = next(iter(val_ds))
        assert tuple(vx.shape) == (4, 64, 64, 3)
        assert len(list(val_ds)) == 1