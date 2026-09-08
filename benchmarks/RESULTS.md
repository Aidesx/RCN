# Benchmark — RCN v1 summarizer models

The repo's official evaluation of the abstractive summarizers in `models/artifacts/`. The eval
set ships inside the repo (`benchmarks/data/`), the runner is re-runnable
(`benchmarks/run_benchmark.py`), and results are versioned under `benchmarks/results/`.

## Eval data

- `benchmarks/data/eval_vi_20260908.jsonl` — **12 Vietnamese news articles** (250–900 words)
  with human reference summaries, drawn deterministically (seed 42).
- **5 of the 12 docs (ids 3, 4, 6, 8, 10) are flagged `clean: false`** because they overlap the
  training set of the community checkpoint `vit5-base-vietnews`: that model scores ROUGE-2 =
  1.0000 on all five — evidence of memorization, not summarization. The official table is
  computed on the **7 clean docs** only; the contaminated ones stay in the data file, flagged, so
  the exclusion is auditable rather than silent.

## Method

- Models live in `models/artifacts/` and run through the exact seam the app uses
  (AutoTokenizer → `T5TokenizerFast(tokenizer_file=…)` fallback), with the exact demo config
  from `configs/summary.yaml` (beam 4, length_penalty 1.2, max 128 tokens, input ≤ 1024 tokens),
  on CPU.
- Metrics: **ROUGE-2** (word-level F1) · **copy5%** — share of the summary that is a verbatim
  5-gram of the source (measures "copy-paste") · **#-hall%** — share of numbers in the summary
  that never appear in the source (measures invented figures) · trigram repetition · seconds per
  document. **Wilcoxon signed-rank** (paired per-doc) against `vit5_v1`.

## Official results — 7 clean docs (full per-doc: `results/2026-09-08/`)

| # | Model | Kind | ROUGE-2 | copy5% | #-hall% | s/doc |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `vit5_soup_0.7_v1` | weight soup (exploratory) | **0.289** | 52 | 0 | 9.7 |
| 2 | `vit5_soup_0.7_v2` | weight soup (exploratory) | 0.266 | 46 | 0 | 11.0 |
| 3 | **`vit5_v1`** ⭐ | fine-tuned, ~127k VI pairs | 0.255 | **44** | **0** | **8.7** |
| 4 | `vit5_soup_0.5_v3` | weight soup | 0.238 | 55 | 0 | 11.8 |
| 5 | `vit5_soup_0.5_v1` | soup v1 ⊕ VietNews @0.5 | 0.231 | 48 | 0 | 11.8 |
| 6 | `summarizer_mt5` | mT5 XLSum, multilingual (zero-shot) | 0.211 | **39** | 0 | 24.6 |
| 7 | `vit5_base_vietnews` | community checkpoint (HF) | 0.135 | 48 | 0 | 8.1 |
| 8 | `vit5_v2` | fine-tuned (overnight, degraded) | 0.100 | 89 | 4 | 16.2 |
| 9 | `vit5_base_original` | VietAI/vit5-base (untrained) | 0.089 | 12 | **25** | 22.2 |

*Remaining entries (chamdenti, trong269, antechai, NishiKyen, giaPhu, extra 0.3/0.5 soups):
ROUGE-2 0.08–0.16 — see `results/2026-09-08/per_doc.csv`.*

## How to read this

- **Fine-tuning is the difference that matters.** `vit5_v1` scores 0.255 vs 0.089 for the
  untrained base (**+186%**) and cuts invented numbers from 25% to **0%**.
- **The top group is statistically tied** (paired Wilcoxon p = 0.27–0.73 vs `vit5_v1`):
  soup 0.7 v1/v2 ≈ vit5_v1 on 7 docs.
- **`vit5_v1` paraphrases more than it copies** — copy5 44%, the lowest of the top group; the
  soups lean toward the source (46–55%) and the untrained base merely echoes fragments (12%).
- **Zero-shot mT5** paraphrases the most (copy5 39%) but costs ~3× wall time.
- **The Sep-2026 community checkpoints** (trong269, antechai, giaPhu, …) do not beat the
  self-trained model on Vietnamese news.

## Recommendation (demo)

- **Default: `vit5_v1`** — statistically tied with the top soups, but with a fully recorded
  training recipe, the lowest copy rate in the top group (real paraphrase), 0% invented numbers,
  ~9 s/doc on CPU, and a half-precision copy (`vit5_v1_fp16`) for GPU demos.
- Tell the model-soup story with `vit5_soup_0.5_v1` (recipe: v1 ⊕ VietNews @0.5, recorded); use
  `vit5_base_original` as the "untrained baseline" contrast, and `summarizer_mt5` to show the
  multilingual/zero-shot direction.
- The α = 0.7 soups score highest numerically but were exploratory blends with no recorded
  recipe — fine as artifacts, not as a defensible demo default.

## Reproduce

```bash
# from the repo root RCN/
.venv/Scripts/python.exe benchmarks/run_benchmark.py
# new results are written to benchmarks/results/<date>/; compare with 2026-09-08
```
