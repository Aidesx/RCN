"""Extract the 12-doc VI eval set (deterministic, seed 42) from the demo pool into a small
self-contained jsonl that can live INSIDE the repo. Docs 3,4,6,8,10 are flagged clean=false
because vit5_base_vietnews scores R2=1.0 on them (training-set overlap / contamination).
"""
import json, pathlib, random

SRC = pathlib.Path("models/artifacts/demo_vi_internal_memo_summaries.jsonl")
DST = pathlib.Path("benchmarks/data/eval_vi_20260908.jsonl")
N = 12
SEED = 42
CONTAMINATED = {3, 4, 6, 8, 10}

rng = random.Random(SEED)
rows = []
with open(SRC, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        try:
            r = json.loads(line)
        except Exception:
            continue
        t, s = (r.get("text") or "").strip(), (r.get("summary") or "").strip()
        tw, sw = len(t.split()), len(s.split())
        if 250 <= tw <= 900 and 10 <= sw <= 60:
            rows.append((t, s))
            if len(rows) >= N * 80:
                break
rng.shuffle(rows)
rows = rows[:N]
DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, "w", encoding="utf-8", newline="") as fh:
    for i, (t, s) in enumerate(rows):
        rec = {"id": i, "text": t, "summary": s,
               "clean": i not in CONTAMINATED,
               "note": "train-overlap risk (vit5_base_vietnews R2=1.0)" if i in CONTAMINATED else ""}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} docs -> {DST} (clean={N - len(CONTAMINATED)})")
