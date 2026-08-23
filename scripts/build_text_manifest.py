"""Stage 8: document-level 70/15/15 stratified split for the synthetic text corpus.

Input : datasets/text/PROVENANCE_TEXT.csv
Output: datasets/splits/text_manifest.csv + text_split_stats.json
Rules : same discipline as the image split (04 §1) — seed 42, frozen test.
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc import paths  # noqa: E402

SEED = 42
rows = list(csv.DictReader(
    open(paths.DATASETS_DIR / "text" / "PROVENANCE_TEXT.csv", encoding="utf-8")))
assert len(rows) >= 300, len(rows)

rng = random.Random(SEED)
by_class = defaultdict(list)
for r in rows:
    by_class[r["project_class"]].append(r)

assign = {}
for cls, docs in sorted(by_class.items()):
    ids = [r["doc_id"] for r in docs]
    rng.shuffle(ids)
    n = len(ids)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    for i, doc_id in enumerate(ids):
        assign[doc_id] = ("train" if i < n_train
                          else "validation" if i < n_train + n_val else "test")

man = paths.SPLITS_DIR / "text_manifest.csv"
with open(man, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["doc_id", "class", "file_name", "split"])
    for r in rows:
        w.writerow([r["doc_id"], r["project_class"], r["file_name"], assign[r["doc_id"]]])

stats = {s: dict(Counter(r["project_class"] for r in rows if assign[r["doc_id"]] == s))
         for s in ["train", "validation", "test"]}
stats["total_docs"] = len(rows)
stats["seed"] = SEED
json.dump(stats, open(paths.SPLITS_DIR / "text_split_stats.json", "w"), indent=2)

print("docs:", len(rows))
for s in ["train", "validation", "test"]:
    print(s, sum(stats[s].values()), stats[s])