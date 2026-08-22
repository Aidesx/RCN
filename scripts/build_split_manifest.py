"""Stage 2: build the document-level 70/15/15 split manifest.

Input : datasets/raw/PROVENANCE.csv (existing pages, folder-per-class)
Output: datasets/splits/manifest.csv + datasets/splits/split_stats.json
Rules : 04 §1 + docs/manifest-schema.md — split by doc_id, stratified by class,
        seed from configs/dataset.yaml, frozen test split.

Uses docproc.paths for all locations, so it runs from any working directory.
"""
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc import paths  # noqa: E402

CFG = paths.dataset_config()
RAW = paths.RAW_DIR
RAW_REL = CFG["paths"]["raw"]  # repo-relative prefix stored in the manifest
SPLITS = paths.SPLITS_DIR
PROV = paths.DATASETS_DIR / "raw" / "PROVENANCE.csv"
RATIOS = CFG["split"]
SEED = RATIOS["seed"]
CLASSES = CFG["classes"]

os.makedirs(SPLITS, exist_ok=True)
rng = random.Random(SEED)

rows = list(csv.DictReader(open(PROV, encoding="utf-8")))
assert len(rows) >= 300, f"need >=300 pages, have {len(rows)}"
assert rows and "source" in rows[0], (
    "PROVENANCE.csv is stale: missing 'source' column — regenerate it "
    "(500 rvlcdip + 200 sroie receipt rows) before building splits")
for r in rows:
    p = RAW / r["project_class"] / r["local_file"]
    assert p.is_file(), f"provenance references missing raw file: {p}"

docs = defaultdict(list)  # doc_id -> [(class, page_file, source)]
for r in rows:
    cls = r["project_class"]
    assert cls in CLASSES, f"unknown class {cls}"
    doc_id = os.path.splitext(r["rvlcdip_filename"])[0]
    page_file = f"{RAW_REL}/{cls}/{r['local_file']}"
    source = r.get("source", "rvlcdip")
    docs[doc_id].append((cls, page_file, source))

doc_ids = list(docs.keys())
rng.shuffle(doc_ids)
per_class = Counter(docs[d][0][0] for d in doc_ids)
assign = {}
for d in doc_ids:
    c = docs[d][0][0]
    k = per_class[c]
    if k <= 0.70 * sum(1 for x in doc_ids if docs[x][0][0] == c):
        assign[d] = "train"
    elif k <= 0.85 * sum(1 for x in doc_ids if docs[x][0][0] == c):
        assign[d] = "validation"
    else:
        assign[d] = "test"
    per_class[c] -= 1

man_path = SPLITS / "manifest.csv"
with open(man_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["doc_id", "class", "page_file", "source", "split"])
    for d in doc_ids:
        for cls, page_file, source in docs[d]:
            w.writerow([d, cls, page_file, source, assign[d]])

stats = {"train": {}, "validation": {}, "test": {}}
for d in doc_ids:
    split = assign[d]
    for cls, _, _ in docs[d]:
        stats[split][cls] = stats[split].get(cls, 0) + 1

n_total = sum(sum(v.values()) for v in stats.values())
stats["total_pages"] = n_total
stats["seed"] = SEED
stats["spot_check"] = {
    "status": "SAMPLE done (4-5 pages/class visually reviewed by AI); FULL 20/class human review PENDING",
    "target": CFG["requirements"]["label_spot_check_pages"],
    "agreement_target": CFG["requirements"]["label_agreement"],
}
json.dump(stats, open(SPLITS / "split_stats.json", "w"), indent=2)

print("pages total:", n_total)
for split in ["train", "validation", "test"]:
    print(split, dict(stats[split]))
missing = [c for c in CLASSES if c not in {x["project_class"] for x in rows}]
print("classes missing entirely:", missing if missing else "none")