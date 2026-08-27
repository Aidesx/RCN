"""Stage 3 golden tests: image preprocessing is deterministic and exact.

Golden arrays were frozen by scripts/make_preprocess_goldens.py; equality is
bit-for-bit so any change to decode/resize/dtype/normalize is caught.
"""
import numpy as np
import pytest
from PIL import Image

from docproc.preprocess.image import cnn_tensor, finetune_tensor, image_to_tensor, tensor_from_file

GOLDEN = __import__("pathlib").Path(__file__).resolve().parents[1] / "golden"
FIXTURE = GOLDEN / "fixture_doc_16x13.png"


def _golden(name):
    return np.load(GOLDEN / f"{name}.npy")


class TestGolden:
    def test_cnn_tensor_matches_golden(self):
        assert np.array_equal(cnn_tensor(FIXTURE), _golden("cnn_64"))

    def test_finetune_tensor_matches_golden(self):
        assert np.array_equal(finetune_tensor(FIXTURE), _golden("finetune_224"))

    def test_shape_and_dtype(self):
        assert cnn_tensor(FIXTURE).shape == (64, 64, 3)
        assert finetune_tensor(FIXTURE).shape == (224, 224, 3)
        assert cnn_tensor(FIXTURE).dtype == np.float32

    def test_normalized_range(self):
        arr = cnn_tensor(FIXTURE)
        assert arr.min() >= 0.0 and arr.max() <= 1.0

    def test_value_range_uses_full_scale(self):
        arr = cnn_tensor(FIXTURE)
        assert arr.max() == pytest.approx(1.0)
        assert arr.min() == pytest.approx(0.0)


class TestDeterminism:
    def test_same_input_same_output(self):
        assert np.array_equal(cnn_tensor(FIXTURE), cnn_tensor(FIXTURE))

    def test_repeated_call_stable(self):
        a = tensor_from_file(FIXTURE, (32, 48))
        b = tensor_from_file(FIXTURE, (32, 48))
        assert np.array_equal(a, b)


class TestChannelHandling:
    def test_grayscale_replicated_to_three_channels(self):
        gray = Image.new("L", (10, 8), 120)
        arr = image_to_tensor(gray, (8, 10))
        assert arr.shape == (8, 10, 3)
        for c in range(3):
            assert np.array_equal(arr[:, :, c], arr[:, :, 0])

    def test_colored_input_not_gray_replicated(self):
        rgb = Image.new("RGB", (10, 8), (10, 120, 240))
        arr = image_to_tensor(rgb, (8, 10))
        assert not np.array_equal(arr[:, :, 0], arr[:, :, 1])


class TestErrors:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            tensor_from_file("tests/golden/does_not_exist.png", (64, 64))

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            image_to_tensor(Image.new("L", (4, 4)), (0, 64))