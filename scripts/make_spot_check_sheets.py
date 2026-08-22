"""Generate spot-check contact sheets: 20 sampled pages per class in a grid.

Output: datasets/splits/spot_check/<class>.png — user reviews visually and
confirms label agreement (target >=99% per configs/dataset.yaml).
"""
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc import paths  # noqa: E402

SEED = 42
GRID = (5, 4)  # cols x rows = 20
THUMB = 200
OUT = paths.SPLITS_DIR / "spot_check"
OUT.mkdir(parents=True, exist_ok=True)

rng = random.Random(SEED)
rows = list(csv.DictReader(open(paths.SPLITS_DIR / "manifest.csv", encoding="utf-8")))
by_class = defaultdict(list)
for r in rows:
    page = Path(r["page_file"])
    if not page.is_absolute():
        page = paths.ROOT / page
    by_class[r["class"]].append(str(page))

for cls, files in sorted(by_class.items()):
    sample = rng.sample(files, min(20, len(files)))
    cell = THUMB + 40
    sheet = Image.new("RGB", (GRID[0] * cell, GRID[1] * cell), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(sample):
        im = Image.open(path).convert("L").resize((THUMB, THUMB))
        x = (i % GRID[0]) * cell
        y = (i // GRID[0]) * cell
        sheet.paste(im, (x + 20, y + 20))
        draw.text((x + 20, y + 2), os.path.basename(path), fill="black")
    out = OUT / f"{cls}.png"
    sheet.save(out)
    print("wrote", out, f"({len(sample)} pages)")