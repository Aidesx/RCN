# RCN — Understand documents on your machine

**RCN** is an offline document-understanding toolkit for Vietnamese and English text. Point it at a
text, a file, or a folder and it returns a layered understanding record — structure, keywords,
topics, fields, and summary. Everything runs locally and is deterministic (seeded) for the same
input.

Scanned pages and photos are **classified only** (which kind of document they are); the current
build does not read their content. Summarization is extraction-first, with an optional small
local seq2seq model for paraphrased summaries and automatic fallback when the model is absent.

## Why RCN?

- **Layered understanding, not just labels.** Every document becomes a structured record with
  five levels: `structure` (L1) → `keywords` (L2) → `topics` (L3) → `fields` (L4) → `summary` (L5).
- **Offline and private.** All processing happens on your machine. The seq2seq summarizer is a
  sub-7B model run locally via `transformers`; scanned pages never leave your disk.
- **Text-layer first.** Text-layer PDFs and office documents are parsed directly; scanned pages are
  rendered and *classified* by document type (receipt, invoice, letter, …) rather than
  transcribed.
- **Honest about missing pieces.** If a model artifact is absent, the router says `unavailable`
  and the rest of the pipeline still completes instead of failing or guessing.
- **Deterministic.** Every random step is seeded (42); same input → same output.
- **Two surfaces, one seam.** The Streamlit app ("RCN Studio") and the CLI share a single
  `understand()` entry point, so the UI and the terminal never disagree.

## Quickstart

Requires Python 3.11. Create a virtual environment and install pinned dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Optional — download the multilingual seq2seq summarizer (needed only for abstractive mode):

```bash
python scripts/download_summarizer.py
```

Run the pipeline on one document:

```bash
python scripts/understand_text.py datasets/text/invoice/invoice_0000.txt
```

…on a whole folder (batch, recursive; writes `*.understanding.json` + `*.understanding.md`
next to each input, or into `--out`):

```bash
python scripts/understand_text.py datasets/text --out runs/demo
```

…or start the web app:

```bash
.venv/Scripts/python -m streamlit run scripts/app.py   # http://localhost:8501
```

`--demo` prints structure statistics for a built-in sample string — useful as a smoke test with
no I/O:

```bash
python scripts/understand_text.py --demo
```

Run the test suite (177 functional tests need no model artifacts; the 12 model-dependent tests
run on demand):

```bash
python -m pytest -q            # functional: L1–L5, IO, dataset, UI logic (~1 min)
python -m pytest -q -m model   # requires real SVM / keras / summarizer artifacts
```

## What you get

The single seam returns one JSON record per document:

```text
{
  "source": "datasets/text/invoice/invoice_0001.txt",
  "doc_type": "invoice",                    # router label ("unavailable" if no model)
  "structure": {…},                          # L1: paragraphs, sentences, words, stats
  "keywords": […],                           # L2: top-k TF-IDF keyphrases (uni + bigram)
  "topics": […],                             # L3: LDA topics, k chosen by UMass coherence
  "fields": {…},                             # L4: invoice number, dates (ISO), amounts …
  "summary": {…},                            # L5: extractive (MMR) or abstractive (seq2seq)
  "timing": {…}
}
```

A human-readable Markdown report is rendered from the same record via
`docproc.nlp.report.render_markdown`.

## How it works

```
input (text / pdf / docx / md / html / png / jpg)
   │
   ├─ io.detect          file type via magic bytes + scanned-PDF probe
   ├─ text branch  → parsers extract text directly from the file
   ├─ image branch → render page / image → classify document type only
   │
   └─ nlp layers → L1 structure → L2 keywords → L3 topics → L4 fields → L5 summary
```

Six document classes are supported: `article`, `form`, `invoice`, `letter`, `receipt`, `report`.

- **Text router** — TF-IDF (uni+bigram, bilingual stopwords) → LinearSVC / RandomForest
  (GridSearchCV, 5-fold, `f1_macro`).
- **Image router** — small CNN ("Architecture A": Conv2D → MaxPool → Conv2D → MaxPool →
  Dense → Softmax), 64×64 input. Used for scanned PDF pages, PNG and JPG.
- **Summarizer** — extractive MMR scoring of sentences with keyphrase priors (λ=0.7), plus an
  optional abstractive engine (fine-tuned ViT5 or multilingual mT5, <7B) with automatic
  fallback to extractive.

Configuration lives in `configs/*.yaml` — no hardcoded hyperparameters.
`configs/summary.yaml` sets the default mode, the sentence count and the abstractive checkpoint.

Deeper write-ups: [Architecture](ARCHITECTURE.md) (system overview, training, honest status) and
[src/ARCHITECTURE.md](src/ARCHITECTURE.md) (per-module code map).

## Quality & benchmarks

**Software.** The functional suite is deterministic and golden-tested: 177 tests cover IO
parsing against `.expected.txt` fixtures, pixel-exact image preprocessing against frozen `.npy`
arrays, L1–L5 NLP behavior, dataset split/leak checks and UI logic (Streamlit `AppTest`); 12
model-dependent tests are opt-in via `-m model`. See [tests/system.md](tests/system.md).

**Router accuracy** (held-out splits, acceptance gate: margin ≥ 0.1 over majority baseline and
macro-F1 ≥ 0.5):

| Router | Test set | Accuracy | Macro-F1 | Majority baseline | Run |
| --- | --- | --- | --- | --- | --- |
| Text (TF-IDF + SVM) | 54 documents | **1.00** | **1.00** | 0.17 / 0.05 | `runs/E0b` |
| Image (CNN, 64×64) | 105 page images | **0.57** | **0.51** | 0.29 / 0.07 | `runs/E1` |

The text classes are well-separated in TF-IDF space, which explains the near-perfect score; the
image CNN is a modest but real improvement over the majority baseline — a deliberate, honest
boundary for a course project on scanned-document *routing*.

**Summarizer benchmark** — 12 Vietnamese news articles with human reference summaries (eval set
bundled in `benchmarks/data/`), generated with the exact demo config (beam 4, max 128 tokens,
CPU). Entries that overlap a training corpus used by one of the compared checkpoints are flagged
`clean: false` in the eval file (a memorization guard) and excluded from the official table;
ROUGE-2 is reported on the clean subset:

| Model | Kind | ROUGE-2 | Copy 5-gram | #-halluc. |
| --- | --- | --- | --- | --- |
| `vit5_soup_0.7_v1` | weight soup (experimental) | 0.289 | 52% | 0% |
| `vit5_soup_0.7_v2` | weight soup (experimental) | 0.266 | 46% | 0% |
| **`vit5_v1`** ⭐ | fine-tuned from VietAI/vit5-base | 0.255 | **44%** | **0%** |
| `vit5_soup_0.5_v1` | weight soup @0.5 (v1 blend) | 0.231 | 48% | 0% |
| `summarizer_mt5` | mT5 XLSum multilingual (zero-shot) | 0.211 | 39% | 0% |
| `vit5_base_vietnews` | pretrained baseline | 0.135 | 48% | 0% |
| `vit5_base_original` | VietAI/vit5-base (untrained) | 0.089 | 12% | **25%** |

Reading the table: fine-tuning matters — the untrained base scores 0.089 and invents numbers in
25% of its summaries, while every trained model stays at 0% number hallucination. The top group
(soup 0.7 v1/v2, `vit5_v1`) is statistically indistinguishable (paired Wilcoxon p > 0.26).
**Recommended demo checkpoint: `vit5_v1`** — comparable ROUGE-2, the lowest verbatim-copy rate
in the top group (real paraphrase, not extraction), zero number hallucination and ~9 s/doc on
CPU. Set it in `configs/summary.yaml`:

```yaml
abstractive:
  finetuned_checkpoint: vit5_v1   # was vit5_soup_0.5_v1
```

Methodology, the full per-doc data and the one-command reproducer:
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Training the summarizer

`scripts/train_summarizer.py` fine-tunes a ViT5/mT5 checkpoint on any `{text, summary}` JSONL
and runs on CPU or GPU (same script, Colab-friendly):

```bash
python scripts/train_summarizer.py \
  --train path/to/train.jsonl \
  --val   path/to/val.jsonl \
  --epochs 2 --batch-size 8 --base-model VietAI/vit5-base
```

The Vietnamese corpora and the recipes that produced the shipped checkpoints (`vit5_v1`,
`vit5_v2`, the weight-soup family) are kept outside this repository (not versioned).
Weight soup = element-wise interpolation between two fine-tuned checkpoints,
`θ = (1−α)·θ₁ + α·θ₂`, e.g. `vit5_soup_0.5_v1` = a 0.5 blend of `vit5_v1`.

## Repository layout

```text
README.md                   This file
ARCHITECTURE.md             System overview, training, honest status
src/ARCHITECTURE.md         Per-module code map (deep reference)
benchmarks/                 Summarizer benchmark: eval set, runner, versioned results
configs/                    YAML configs (dataset, text, cnn, pipeline, summary, fields)
src/docproc/                Core package — see src/ARCHITECTURE.md
  io/                       detect · parsers (pdf/docx/md/html) · render (scans)
  preprocess/               image tensors (64×64) · text TF-IDF vectorizer
  models/                   cnn.py (image router) · text_classifier.py (SVM/RF)
  training/                 dataset registry · seeded training harness
  evaluation/               metrics · acceptance gate · run reports
  nlp/                      structure · keywords · topics · fields · summary · report seam
scripts/                    understand_text.py (CLI) · app.py (RCN Studio) · train_summarizer.py
datasets/                   raw images (~700) + text corpus (360 EN docs) + splits (gitignored;
                            provenance: datasets/text/PROVENANCE_TEXT.csv)
models/artifacts/           Trained artifacts (gitignored): joblib vectorizer/SVM, keras CNN,
                            summarizer checkpoints (vit5_v1, soups, mT5, pretrained baselines)
runs/                       Per-experiment metrics (E0b, E1, E-U0/U2 …)
tests/                      pytest suite — map in tests/system.md
requirements.txt            Pinned dependencies (verified in .venv)
```

## Reference

| Doc | What it answers |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the pipeline fits together; training history; what is and isn't shipped |
| [src/ARCHITECTURE.md](src/ARCHITECTURE.md) | What each module/file in `src/docproc/` does |
| [tests/system.md](tests/system.md) | Test suite map, roles, conventions |
| [benchmarks/RESULTS.md](benchmarks/RESULTS.md) | Summarizer evaluation methodology + full results |
