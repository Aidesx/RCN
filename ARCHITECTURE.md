# RCN — Architecture

RCN is an offline document-understanding pipeline for Vietnamese and English. It turns a text,
file, or folder into a layered record — **L1 structure → L2 keywords → L3 topics → L4 fields →
L5 summary** — plus an advisory document-type label. This document covers how the pieces fit
together and what is trained and shipped.

## At a glance

| Component | Technology | Status |
| --- | --- | --- |
| Document ingestion | PyMuPDF, pypdf, pdfplumber, python-docx, BeautifulSoup4, html2text | shipped |
| Text routing | TF-IDF (uni+bigram) → LinearSVC (GridSearchCV, 5-fold, `f1_macro`) | shipped |
| Image routing | TensorFlow CNN "Architecture A" (64×64) | shipped |
| L1–L5 | see *Document layers* below | shipped |
| Abstractive summarizer | local ViT5/mT5 seq2seq (<7B) | shipped (opt-in) |
| Scanned pages | rendered and classified by document type | partial support |
| 224×224 MobileNetV2 arm | `configs/finetune.yaml`, `finetune_tensor()` | explored, not shipped |

## Design principles

- **Understanding-first.** L1–L3 are the product; the router label is advisory metadata and never
  fails the pipeline.
- **Local and text-layer-first.** Text-layer documents are parsed directly; scans and photos go
  through the image branch. The only model bigger than a vectorizer is a sub-7B seq2seq
  summarizer that runs locally through `transformers`.
- **One seam.** `docproc.nlp.report.understand()` / `understand_file()` serve the CLI, the UI, and
  the tests identically.
- **Deterministic.** All randomness is seeded (42): same input → same JSON record.
- **Graceful degradation.** Missing artifacts yield `doc_type: "unavailable"` / extractive
  fallback instead of crashes; the abstractive tests self-skip when the checkpoint is absent.
- **Config, not constants.** Hyperparameters live in `configs/*.yaml`; `docproc.paths` is the only
  module that knows the layout.

## Request flow

```text
input (text / file / folder)
   │
   ├─ io.detect          file type: magic bytes + scanned-PDF probe
   ├─ text branch  → parsers  extract text → L1 → L2 → L3 → L4 → L5
   ├─ image branch → render   page / image → document-type label
   │
   └─ output  →  JSON record + Markdown report
```

`understand_file(path)` prepends `io.detect_file_type`; text documents run the full L1–L5 chain.
The CLI (`scripts/understand_text.py`) and the Streamlit UI (`scripts/app.py`) are thin surfaces
over the same seam.

## Document layers (L1–L5)

Each layer is a pure function of its input — no hidden state.

| Layer | Module | What it does |
| --- | --- | --- |
| L1 structure | `nlp/structure.py` | paragraph → sentence → word split; counts and averages |
| L2 keywords | `nlp/keywords.py` | top-k keyphrases by in-document TF-IDF (uni+bigram), with bilingual stopword and number filtering |
| L3 topics | `nlp/topics.py` | LDA with k chosen by UMass coherence (3…10); readable labels via PCA + K-Means |
| L4 fields | `nlp/fields.py` | per-class regex schemas (invoice #, ISO dates, totals…), overridable in `configs/fields.yaml`; values normalized |
| L5 summary | `nlp/summary.py` | extractive MMR (λ=0.7) or abstractive seq2seq; falls back to extractive |

## Routing (advisory label)

Six classes: `article`, `form`, `invoice`, `letter`, `receipt`, `report`.

- **Text** — TF-IDF + LinearSVC (C=0.1 from GridSearchCV), trained by
  `scripts/run_text_baseline.py`.
- **Image** — CNN "Architecture A": Conv2D(32,5) → MaxPool → Conv2D(64,5) → MaxPool → Dense(256)
  → Dropout(0.5) → Softmax(6), 64×64×3 input, Adam lr=1e-3; trained by `scripts/run_cnn.py`.

Measured on held-out splits (acceptance gate: ≥ 0.1 margin over the majority baseline and
macro-F1 ≥ 0.5):

| Router | Test n | Accuracy | Macro-F1 | Majority baseline |
| --- | --- | --- | --- | --- |
| Text (SVM) | 54 | **1.00** | **1.00** | 0.17 / 0.05 |
| Image (CNN) | 105 | **0.57** | **0.51** | 0.29 / 0.07 |

Routing is advisory — the understanding layers never depend on it.

## Summarization

Two engines behind one `summarize()`:

1. **Extractive (default)** — sentence scoring with L2 keyphrase priors + MMR anti-redundancy
   (λ=0.7). Deterministic.
2. **Abstractive (opt-in)** — a local seq2seq checkpoint generates a fresh summary; the
   checkpoint is set in `configs/summary.yaml`. Missing model or deps → automatic extractive
   fallback.

Self-trained checkpoints (fine-tuned from `VietAI/vit5-base`; recipes archived with the
artifacts):

| Artifact | Benchmark ROUGE-2 (clean subset) |
| --- | --- |
| `vit5_v1` ⭐ (recommended demo) | 0.255 |
| `vit5_soup_0.5_v1` | 0.231 |
| `summarizer_mt5` (zero-shot) | 0.211 |
| `vit5_base_original` (untrained) | 0.089 |

Full methodology and per-doc numbers: [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Data & splits

- Corpora ship in-repo: images built by `scripts/download_rvlcdip_subset.py`, text by
  `scripts/make_text_corpus.py`; provenance in `datasets/raw/PROVENANCE.csv` and
  `datasets/text/PROVENANCE_TEXT.csv`.
- Split 70/15/15 (train/val/test) with a leak-checked manifest at `datasets/splits/manifest.csv`.
- Summarization corpora (XLSum-VI, VietNews, …) live outside the repository (not versioned).

## Tests & benchmark

- 177 functional tests + 12 model-dependent tests — roles and conventions in
  [tests/system.md](tests/system.md).
- Summarizer benchmark on the bundled eval set — full results in
  [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Status

- **Shipped:** full L1–L5 pipeline over text documents; scanned pages and photos classified by
  type; CLI + Streamlit UI; deterministic golden-tested IO; the router models above; a fine-tuned
  Vietnamese abstractive summarizer with recorded recipes.
- **Explored, deliberately not shipped:** the 224×224 MobileNetV2 arm (`configs/finetune.yaml` +
  `finetune_tensor()` only).
