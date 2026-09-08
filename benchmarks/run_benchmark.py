"""In-repo benchmark: VI summarizer models in models/artifacts/ through the app seam.

Data: benchmarks/data/eval_vi_20260908.jsonl (12 docs; docs flagged clean=false are excluded
from the official table — they overlap vit5_base_vietnews's training set, see RESULTS.md).
Generation config mirrors the demo (configs/summary.yaml abstractive block).
Metrics: ROUGE-1/2/L (word F1), copy5% (verbatim 5-gram copying from source), num-hall%
(numbers in summary missing from source), trigram repetition, s/doc. Wilcoxon vs vit5_v1.

Usage (from RCN/):
    .venv/Scripts/python.exe benchmarks/run_benchmark.py [--data benchmarks/data/eval_vi_20260908.jsonl]
Writes: benchmarks/results/<YYYY-MM-DD>/results.md, results.json, per_doc.csv
"""
import argparse, csv, gc, json, math, pathlib, re, sys, time, warnings
warnings.filterwarnings("ignore")

RCN = pathlib.Path(__file__).resolve().parents[1]
ART = RCN / "models" / "artifacts"
DEFAULT_DATA = RCN / "benchmarks" / "data" / "eval_vi_20260908.jsonl"

sys.path.insert(0, str(RCN / "src"))
sys.path.insert(0, str(RCN))

import yaml  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402
from transformers import T5TokenizerFast  # noqa: E402

TOKEN_RE = re.compile(r"[\w\u00C0-\u1EF9]+", re.UNICODE)
NUM_RE = re.compile(r"\d[\d.,]*")

def toks(s):
    return TOKEN_RE.findall(s.lower())

def nums(s):
    out = set()
    for m in NUM_RE.finditer(s):
        v = re.sub(r"[.,]", "", m.group(0))
        if v and v != "0":
            out.add(v.lstrip("0") or "0")
    return out

def lcs_len(a, b):
    if len(a) < len(b):
        a, b = b, a
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            if x == b[j - 1]:
                dp[j] = max(dp[j], dp[j - 1]) if x != b[j - 1] else prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]

def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0

def rouge1(h, r):
    h, r = set(h), set(r)
    if not r:
        return 0.0
    m = len(h & r)
    return f1(m / len(h) if h else 0.0, m / len(r))

def rouge2(h, r):
    def ng(t):
        return set(zip(*[t[i:] for i in range(2)]))
    h, r = ng(h), ng(r)
    if not r:
        return 0.0
    m = len(h & r)
    return f1(m / len(h) if h else 0.0, m / len(r))

def rouge_l(h, r):
    l = lcs_len(h, r)
    return f1(l / len(h), l / len(r)) if l else 0.0

def copy5(hyp, src):
    if len(hyp) < 5:
        return 1.0 if hyp and all(w in src for w in hyp) else 0.0
    hs = {tuple(hyp[i:i + 5]) for i in range(len(hyp) - 4)}
    ss = {tuple(src[i:i + 5]) for i in range(len(src) - 4)} if len(src) >= 5 else set()
    return len(hs & ss) / len(hs) if hs else 0.0

def num_halluc(hyp, src):
    hn, sn = nums(hyp), nums(src)
    return 0.0 if not hn else len(hn - sn) / len(hn)

def tri_repeat(hyp):
    t = [tuple(hyp[i:i + 3]) for i in range(len(hyp) - 2)]
    return (len(t) - len(set(t))) / len(t) if t else 0.0

def mean(x):
    return sum(x) / len(x) if x else 0.0

def stdev(x):
    if len(x) < 2:
        return 0.0
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))

def load_tok(bd):
    try:
        return AutoTokenizer.from_pretrained(str(bd))
    except Exception:
        pass
    try:
        return T5TokenizerFast(tokenizer_file=str(bd / "tokenizer.json"))
    except Exception:
        pass
    try:
        return AutoTokenizer.from_pretrained(str(bd), use_fast=False)
    except Exception as e:
        raise RuntimeError(f"no tokenizer: {e}") from e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, default=DEFAULT_DATA)
    ap.add_argument("--skip-dup", action="store_true", default=True,
                    help="skip vit5_base_vietnews_summarization_copy + vit5_v1_fp16 (same weights)")
    args = ap.parse_args()

    pool = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines()]
    clean_pool = [d for d in pool if d.get("clean", True)]
    print(f"pool: {len(pool)} docs ({len(clean_pool)} clean)")

    cfg = yaml.safe_load((RCN / "configs" / "summary.yaml").read_text(encoding="utf-8"))
    ab = cfg["abstractive"]
    gen_kw = dict(
        max_new_tokens=ab.get("max_new_tokens", 128),
        min_new_tokens=ab.get("min_new_tokens", 8),
        num_beams=ab.get("num_beams", 4),
        length_penalty=ab.get("length_penalty", 1.2),
        early_stopping=ab.get("early_stopping", True),
        no_repeat_ngram_size=ab.get("no_repeat_ngram_size", 3),
    )
    print("gen config:", gen_kw)

    dirs = sorted(p for p in ART.iterdir() if p.is_dir())
    if args.skip_dup:
        dirs = [d for d in dirs if d.name not in ("vit5_base_vietnews_summarization_copy", "vit5_v1_fp16")]
    print("models:", len(dirs))

    outdir = RCN / "benchmarks" / "results" / time.strftime("%Y-%m-%d")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_fh = open(outdir / "per_doc.csv", "w", newline="", encoding="utf-8")
    csv_w = csv.DictWriter(csv_fh, fieldnames=["model", "doc", "clean", "r2", "copy5", "num_hall", "tri", "gen_s", "hyp"])
    csv_w.writeheader()

    agg_all, agg_clean = [], []
    r2_clean = {}
    for d in dirs:
        bd = d / "final" if (d / "final").is_dir() else d
        t0 = time.time()
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(bd), low_cpu_mem_usage=True)
            model.eval()
            tok = load_tok(bd)
            load_s = round(time.time() - t0, 1)
        except Exception as e:
            print(f"== {d.name}: LOAD FAIL {str(e)[:120]}")
            continue
        m_all = {k: [] for k in ("r1", "r2", "rl", "c5", "nh", "tr", "gt")}
        m_clean = {k: [] for k in ("r1", "r2", "rl", "c5", "nh", "tr", "gt")}
        for doc in pool:
            clean = bool(doc.get("clean", True))
            tin = time.time()
            inp = tok(doc["text"], return_tensors="pt", truncation=True, max_length=1024)
            out = model.generate(**inp, **gen_kw)
            hyp = tok.decode(out[0], skip_special_tokens=True).strip()
            ht = toks(hyp)
            rt = toks(doc["summary"])
            st = toks(doc["text"])
            vals = dict(r1=rouge1(ht, rt), r2=rouge2(ht, rt), rl=rouge_l(ht, rt),
                        c5=copy5(ht, st), nh=num_halluc(hyp, doc["text"]), tr=tri_repeat(ht),
                        gt=round(time.time() - tin, 1))
            for k in m_all:
                m_all[k].append(vals[k])
                if clean:
                    m_clean[k].append(vals[k])
            csv_w.writerow(dict(model=d.name, doc=doc["id"], clean=clean, r2=vals["r2"],
                                copy5=vals["c5"], num_hall=vals["nh"], tri=vals["tr"],
                                gen_s=vals["gt"], hyp=hyp[:300]))
            print(f"  [{d.name}] doc{doc['id']} {vals['gt']}s")
        csv_fh.flush()
        def agg_of(m):
            return dict(r1=round(mean(m["r1"]), 4), r1_sd=round(stdev(m["r1"]), 4),
                        r2=round(mean(m["r2"]), 4), r2_sd=round(stdev(m["r2"]), 4),
                        rl=round(mean(m["rl"]), 4), copy5=round(mean(m["c5"]), 4),
                        num_hall=round(mean(m["nh"]), 4), trirep=round(mean(m["tr"]), 4),
                        s_per_doc=round(mean(m["gt"]), 1), n=len(m["r2"]))
        a_all = {"model": d.name, "load_s": load_s, **agg_of(m_all)}
        a_clean = {"model": d.name, **agg_of(m_clean)}
        agg_all.append(a_all)
        if len(m_clean["r2"]) == len(clean_pool):
            agg_clean.append(a_clean)
            r2_clean[d.name] = m_clean["r2"]
        print(f"== {d.name}: clean-R2 {a_clean['r2']}+-{a_clean['r2_sd']} (n={a_clean['n']}) "
              f"copy5 {a_clean['copy5']} numhall {a_clean['num_hall']} {a_clean['s_per_doc']}s/doc")
        try:
            del model
        except Exception:
            pass
        gc.collect()
    csv_fh.close()

    from scipy import stats
    champ = "vit5_v1"
    base = r2_clean.get(champ, [])
    for a in agg_clean:
        other = r2_clean.get(a["model"])
        if a["model"] == champ or not other or len(other) != len(base):
            a["p_vs_v1"] = None
            a["wins_vs_v1"] = None
            continue
        try:
            a["p_vs_v1"] = round(float(stats.wilcoxon(base, other, zero_method="wilcox").pvalue), 4)
        except Exception:
            a["p_vs_v1"] = None
        a["wins_vs_v1"] = sum(1 for x, y in zip(base, other) if x > y)

    (outdir / "results.json").write_text(
        json.dumps({"meta": {"date": time.strftime("%Y-%m-%d"), "pool": str(args.data),
                             "clean_docs": [d["id"] for d in clean_pool]},
                    "all_docs": agg_all, "clean": agg_clean},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    rows = sorted(agg_clean, key=lambda a: -a["r2"])
    lines = ["| model | ROUGE-2 | R2±sd | ROUGE-L | copy5% | num-hall% | tri% | s/doc | p vs v1 | wins |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for a in rows:
        p = f"{a['p_vs_v1']:.4f}" if a.get("p_vs_v1") is not None else "-"
        w = "-" if a.get("wins_vs_v1") is None else a["wins_vs_v1"]
        lines.append(f"| {a['model']} | {a['r2']} | {a['r2_sd']} | {a['rl']} | {a['copy5']*100:.0f} | "
                     f"{a['num_hall']*100:.0f} | {a['trirep']*100:.1f} | {a['s_per_doc']} | {p} | {w} |")
    (outdir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved:", outdir)

if __name__ == "__main__":
    main()
