"""Generate Stage 3 golden files: deterministic fixture image + expected tensors.

Run once (or after changing the preprocessing contract):
    python scripts/make_preprocess_goldens.py
Golden arrays freeze the exact decode/resize/normalize behavior; tests assert
the module output equals them bit-for-bit.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docproc import paths  # noqa: E402
from docproc.preprocess.image import image_to_tensor  # noqa: E402

GOLDEN = paths.TESTS_DIR / "golden"
GOLDEN.mkdir(parents=True, exist_ok=True)

FIXTURE = GOLDEN / "fixture_doc_16x13.png"

im = Image.new("L", (16, 13), 200)
d = ImageDraw.Draw(im)
for x in range(16):
    d.line([(x, 0), (x, 12)], fill=100 + x * 8)
d.rectangle([1, 1, 6, 5], fill=30)
d.line([(0, 0), (15, 12)], fill=255, width=1)
d.point((7, 9), fill=0)
im.save(FIXTURE)

for name, size in [("cnn_64", (64, 64)), ("finetune_224", (224, 224))]:
    arr = image_to_tensor(im.convert("RGB"), size)
    np.save(GOLDEN / f"{name}.npy", arr)
    print(name, arr.shape, arr.dtype, float(arr.min()), float(arr.max()))

print("fixture:", FIXTURE.name)