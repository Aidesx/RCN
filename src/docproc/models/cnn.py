"""Architecture A — self-built CNN (course Ch9 template scaled to 64x64).

Layer spec is fixed by 03 §2 / 04 §4 and lives in configs/cnn.yaml:
Conv2D(32,5,same,ReLU) -> MaxPool(2) -> Conv2D(64,5,same,ReLU) -> MaxPool(2)
-> Flatten -> Dense(256,ReLU) -> Dropout(0.5) -> Dense(C,softmax).
"""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from docproc import paths


def _load_config(config_path: Path | None) -> dict:
    return paths.load_config("cnn") if config_path is None else paths.load_yaml_file(config_path)


def num_classes() -> int:
    return len(paths.class_names())


def class_names() -> list[str]:
    return paths.class_names()


def build_model(config_path: Path | None = None, n_classes: int | None = None) -> tf.keras.Model:
    """Build and compile the self-built CNN exactly per configs/cnn.yaml."""
    cfg = _load_config(config_path)
    h, w, c = int(cfg["input"]["height"]), int(cfg["input"]["width"]), int(cfg["input"]["channels"])
    a = cfg["architecture"]
    classes = n_classes if n_classes is not None else num_classes()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(h, w, c)),
            tf.keras.layers.Conv2D(int(a["conv1_filters"]), int(a["conv1_kernel"]),
                                   strides=1, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=int(a["pool_size"]), strides=2),
            tf.keras.layers.Conv2D(int(a["conv2_filters"]), int(a["conv2_kernel"]),
                                   strides=1, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=int(a["pool_size"]), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(int(a["dense_units"]), activation="relu"),
            tf.keras.layers.Dropout(float(a["dropout"])),
            tf.keras.layers.Dense(classes, activation="softmax"),
        ],
        name="cnn_architecture_a",
    )

    t = cfg["training"]
    lr = float(t["learning_rate"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=t["loss"],
        metrics=["accuracy"],
    )
    return model