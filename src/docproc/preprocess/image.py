"""Deterministic image decode -> resize -> float32 [0,1]; grayscale replicated to RGB."""
from __future__ import annotations

import numpy as np
from PIL import Image

from docproc.paths import load_config

RESAMPLING = Image.Resampling.BICUBIC


def _read_image(path) -> Image.Image:
    from pathlib import Path

    if not Path(path).is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    with Image.open(path) as im:
        im.load()
    return im.convert("RGB")


def image_to_tensor(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Convert a PIL image to a float32 tensor of shape (H, W, 3), values in [0,1]."""
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"invalid target size: {size}")
    resized = image.convert("RGB").resize((size[1], size[0]), RESAMPLING)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return arr


def tensor_from_file(path, size: tuple[int, int]) -> np.ndarray:
    """Read a page image file and return the model tensor (H, W, 3), [0,1] float32."""
    return image_to_tensor(_read_image(path), size)


def _model_size(config_name: str) -> tuple[int, int]:
    cfg = load_config(config_name)
    return int(cfg["input"]["height"]), int(cfg["input"]["width"])


def file_to_pil(path) -> Image.Image:
    """Public accessor: decoded RGB PIL image from a file path."""
    return _read_image(path)


def cnn_tensor(path) -> np.ndarray:
    """Tensor for the self-built CNN (64x64x3) — sizes from configs/cnn.yaml."""
    return tensor_from_file(path, _model_size("cnn"))


def finetune_tensor(path) -> np.ndarray:
    """Tensor for the MobileNetV2 fine-tune arm (224x224x3) — sizes from configs/finetune.yaml."""
    return tensor_from_file(path, _model_size("finetune"))