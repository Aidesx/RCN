"""Stage 4 tests: Architecture A matches the frozen spec (03 §2 / 04 §4 / cnn.yaml)."""
import numpy as np
import pytest

from docproc.models.cnn import build_model, class_names, num_classes

EXPECTED_PARAMS = 4_249_798  # conv 2,432 + conv 51,264 + dense 4,194,560 + out 1,542


@pytest.fixture(scope="module")
def model():
    return build_model()


class TestArchitecture:
    def test_output_shape(self, model):
        out = model(np.zeros((2, 64, 64, 3), dtype=np.float32), training=False)
        assert tuple(out.shape) == (2, num_classes())

    def test_softmax_rows_sum_to_one(self, model):
        out = model(np.zeros((2, 64, 64, 3), dtype=np.float32), training=False)
        np.testing.assert_allclose(out.numpy().sum(axis=1), 1.0, rtol=1e-5)

    def test_param_count_matches_spec(self, model):
        assert model.count_params() == EXPECTED_PARAMS

    def test_layer_sequence(self, model):
        kinds = [type(l).__name__ for l in model.layers if type(l).__name__ != "InputLayer"]
        assert kinds == [
            "Conv2D", "MaxPooling2D", "Conv2D",
            "MaxPooling2D", "Flatten", "Dense", "Dropout", "Dense",
        ]

    def test_conv_filters_and_kernels_from_config(self, model):
        convs = [l for l in model.layers if type(l).__name__ == "Conv2D"]
        assert [c.filters for c in convs] == [32, 64]
        assert [c.kernel_size for c in convs] == [(5, 5), (5, 5)]
        assert all(c.padding == "same" for c in convs)

    def test_dropout_rate_from_config(self, model):
        drop = next(l for l in model.layers if type(l).__name__ == "Dropout")
        assert drop.rate == 0.5


class TestCompile:
    def test_loss_and_optimizer(self, model):
        assert model.loss == "sparse_categorical_crossentropy"
        assert isinstance(model.optimizer, __import__("tensorflow").keras.optimizers.Adam)
        assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(1e-3)


class TestClassMapping:
    def test_class_names_order_fixed(self):
        assert class_names() == ["letter", "form", "report", "article", "invoice", "receipt"]

    def test_num_classes_is_six(self):
        assert num_classes() == 6