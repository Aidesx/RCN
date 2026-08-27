"""Fast tests for the deepened dataset module: manifest validation + arm
registry — exercised via tmp manifests and tiny PNGs (no TF, no real data)."""
import numpy as np
import pytest
from PIL import Image

from docproc import paths
from docproc.training.data import (
    ARMS,
    load_split_arrays,
    make_datasets,
    read_manifest,
    split_class_counts,
)


@pytest.fixture()
def tmp_manifest(tmp_path):
    """3-page manifest + tiny PNGs in a tmp tree; page paths absolute."""
    pages = {}
    for i, cls in enumerate(["letter", "form", "receipt"]):
        p = tmp_path / f"p{i}.png"
        Image.new("RGB", (20, 12), (i * 40, 10, 10)).save(p)
        pages[cls] = p
    man = tmp_path / "manifest.csv"
    man.write_text(
        "doc_id,class,page_file,source,split\n"
        f"a,letter,{pages['letter']},rvlcdip,train\n"
        f"b,form,{pages['form']},sroie,train\n"
        f"c,receipt,{pages['receipt']},sroie,test\n",
        encoding="utf-8",
    )
    return man


class TestReadManifest:
    def test_read_all_and_filter(self, tmp_manifest):
        assert len(read_manifest(manifest_path=tmp_manifest)) == 3
        rows = read_manifest(split="train", manifest_path=tmp_manifest)
        assert {r["doc_id"] for r in rows} == {"a", "b"}

    def test_missing_column_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("doc_id,class\nx,letter\n", encoding="utf-8")
        with pytest.raises(ValueError):
            read_manifest(manifest_path=bad)

    def test_real_manifest_schema_ok(self):
        rows = read_manifest()
        assert len(rows) == 700 and all(r["source"] for r in rows)


class TestArmRegistry:
    def test_registry_has_both_arms(self):
        assert set(ARMS) == {"cnn", "finetune"}

    def test_cnn_arm_shape(self, tmp_manifest):
        X, y = load_split_arrays("train", arm="cnn", manifest_path=tmp_manifest)
        assert X.shape == (2, 64, 64, 3) and X.dtype == np.float32

    def test_finetune_arm_shape(self, tmp_manifest):
        X, y = load_split_arrays("train", arm="finetune", manifest_path=tmp_manifest)
        assert X.shape == (2, 224, 224, 3)

    def test_unknown_arm_raises(self, tmp_manifest):
        with pytest.raises(ValueError):
            load_split_arrays("train", arm="vgg", manifest_path=tmp_manifest)

    def test_labels_follow_class_order(self, tmp_manifest):
        _, y = load_split_arrays("test", arm="cnn", manifest_path=tmp_manifest)
        names = paths.class_names()
        assert int(y[0]) == names.index("receipt")


class TestCounts:
    def test_split_class_counts_real(self):
        counts = split_class_counts("train")
        assert sum(counts.values()) == 490 and counts["receipt"] == 140


class TestMakeDatasets:
    def test_batch_shapes(self):
        rng = np.random.default_rng(42)
        X = rng.random((10, 64, 64, 3), dtype=np.float32)
        y = rng.integers(0, 6, size=10).astype(np.int64)
        train_ds, val_ds = make_datasets(X, y, X[:4], y[:4], batch_size=4, seed=42)
        xb, yb = next(iter(train_ds))
        assert tuple(xb.shape) == (4, 64, 64, 3)
        assert int(np.max(yb.numpy())) < 6