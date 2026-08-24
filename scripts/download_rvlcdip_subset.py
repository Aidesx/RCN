"""Build the v0 dataset from RVL-CDIP mirror (jordyvl/rvl_cdip_100_examples_per_class).

Keeps only the 5 classes present in RVL-CDIP that match project classes:
    letter, form, report (scientific report), article (news article), invoice.
receipt is NOT in RVL-CDIP -> self-collected later (spec 04 §1 sources a/b).

Merges the mirror's train/validation/test splits as raw source material; the
project's own document-level 70/15/15 split is created later (05 Stage 2).
Output: datasets/raw/<class>/*.png (grayscale) + datasets/raw/PROVENANCE.csv.
"""
import csv
import io
import os
from collections import Counter

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "jordyvl/rvl_cdip_100_examples_per_class"
FILES = {
    "train": "data/train-00000-of-00001-81f1d229db782541.parquet",
    "validation": "data/validation-00000-of-00001-00031909a6e73300.parquet",
    "test": "data/test-00000-of-00001-d5e0db6590d27073.parquet",
}
LABELS = {0: "letter", 1: "form", 5: "report", 9: "article", 11: "invoice"}
PER_CLASS = 100
RAW = os.path.join("datasets", "raw")
SRC = os.path.join(RAW, "_source")

os.makedirs(SRC, exist_ok=True)
for cls in LABELS.values():
    os.makedirs(os.path.join(RAW, cls), exist_ok=True)

counts = Counter()
seen = set()

with open(os.path.join(RAW, "PROVENANCE.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["project_class", "local_file", "source_split", "rvlcdip_filename", "label_id", "source"])
    for split, parquet in FILES.items():
        local = hf_hub_download(REPO, parquet, repo_type="dataset")
        df = pd.read_parquet(local)
        for _, row in df.iterrows():
            label = int(row["label"])
            if label not in LABELS:
                continue
            cls = LABELS[label]
            if counts[cls] >= PER_CLASS:
                continue
            src = row["image"]["path"]
            if src in seen:
                continue
            seen.add(src)
            img = Image.open(io.BytesIO(row["image"]["bytes"]))
            if img.mode != "L":
                img = img.convert("L")
            out_name = f"{cls}_{counts[cls]:04d}.png"
            img.save(os.path.join(RAW, cls, out_name))
            writer.writerow([cls, out_name, split, src, label, "rvlcdip"])
            counts[cls] += 1
        print("after", split, ":", dict(counts), flush=True)
        if all(counts[c] >= PER_CLASS for c in LABELS.values()):
            break

print("DONE", dict(counts), "total", sum(counts.values()))