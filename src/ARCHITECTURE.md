# `src/docproc/` — code map

Deep reference for everything under `src/docproc/`. If you want the system-level view, read
[../ARCHITECTURE.md](../ARCHITECTURE.md) first; if you want to know which file does what, stay
here. Last synced 2026-09-09 against the working tree.

## Data flow

```
input file / text
   │
   ├─ io.detect_file_type()     magic bytes + scan probe → kind of file
   ├─ io.extract_text()         deterministic text extraction per format (no OCR)
   │
   ├─ report._router_text()     text → SVM label        (advisory)
   ├─ report._router_image()    image/scan → CNN label  (advisory)
   │
   ├─ structure.analyze_structure()   L1  word → sentence → paragraph
   ├─ keywords.extract_keywords()     L2  top-k TF-IDF keyphrases
   ├─ topics.extract_topics()         L3  LDA + UMass coherence (+ labeled reps)
   ├─ fields.extract_fields()         L4  regex fields per doc_type
   ├─ summary.summarize()             L5  extractive MMR | abstractive seq2seq + fallback
   │
   └─ report.understand()             THE seam → record dict
      report.render_markdown()        → human-readable Markdown
```

## `paths.py` — layout & config authority

The only module that knows where things live (`ROOT / CONFIG_DIR / RUNS_DIR / DATASETS_DIR /
ARTIFACTS_DIR`). Everything else imports from here — no hardcoded paths anywhere else.

| Function | Purpose |
| --- | --- |
| `load_config(name)` | load `configs/<name>.yaml` |
| `class_names()` | the six classes: article, form, invoice, letter, receipt, report |
| `artifacts_dir()` etc. | resolved artifact/runs/datasets directories |

## `io/` — ingestion

| File | Contents | Role |
| --- | --- | --- |
| `detect.py` | `detect_file_type()` — magic bytes + text-format sniff + scanned-PDF character probe; structured errors `DocumentIOError` / `ParseError` / `UnsupportedFormatError` | decides the branch: text / image / scan |
| `parsers.py` | `extract_text()` dispatches `_extract_pdf / _extract_docx / _extract_markdown / _extract_html` | deterministic text extraction, no OCR |
| `render.py` | `render_pdf_pages()` (PDF → images, default dpi), `extract_embedded_images_pdf()` | feeds the image branch — pages are *classified*, never read |

## `preprocess/` — model inputs

| File | Contents |
| --- | --- |
| `image.py` | decode RGB → **bicubic** resize → float32 [0,1]. `cnn_tensor()` = 64×64×3 (used by Architecture A). `finetune_tensor()` = 224×224×3 exists for the *unshipped* finetune arm (`configs/finetune.yaml`). Fully deterministic |
| `text.py` | `TextVectorizer` — TF-IDF wrapper (uni+bigram, bilingual stopwords), `save()/load()` via joblib |

## `models/` — router architecture

Two kinds of thing live here — be careful not to confuse them:

| File | Contents |
| --- | --- |
| `cnn.py` | `build_model()` — **Architecture A** exactly as declared in `configs/cnn.yaml`: Conv2D(32,5)-Pool-Conv2D(64,5)-Pool-Flatten-Dense(256)-Dropout(0.5)-Softmax(6); Adam lr=1e-3, sparse categorical CE |
| `text_classifier.py` | `train_baseline()` — GridSearchCV over LinearSVC vs RandomForest (5-fold, `f1_macro`), writes artifacts; `evaluate_baseline()` — test eval + majority baseline + acceptance gate |

The **learned weights** are *not* in this directory:

- `models/artifacts/text_vectorizer.joblib`, `models/artifacts/text_model_svm.joblib` — the
  trained text router (rebuilt 2026-09-08 by `scripts/run_text_baseline.py`, see `runs/E0b`).
- `runs/E1/best.keras` — the trained image CNN.
- `models/artifacts/summarizer_mt5/`, `models/artifacts/vit5_*/` — abstractive summarizer
  checkpoints.

All artifact directories are gitignored — they are reproduced, never committed.

## `training/` — shared training infrastructure

| File | Purpose |
| --- | --- |
| `data.py` | dataset registry (image arms 64×64 / 224×224): `read_manifest()`, `load_split_arrays()`, `make_datasets()` from `datasets/splits/manifest.csv` |
| `harness.py` | `run_training()` — seed 42, snapshot config into `runs/<name>/config.yaml`, EarlyStopping + ModelCheckpoint, write `history.csv` + `metrics.json` |

## `evaluation/` — shared measurement

| File | Purpose |
| --- | --- |
| `metrics.py` | `compute_metrics()` (accuracy / macro-F1 / confusion matrix); majority-class baselines; `acceptance_gate()` → PASS/FAIL against baseline |
| `report.py` | `report_run()` — frozen test on the held-out split, writes `metrics_eval.json`, confusion CSV, learning-curve PNGs |

## `nlp/` — the understanding core (L1–L5 + report)

| File | Layer | What it does |
| --- | --- | --- |
| `structure.py` | L1 | `analyze_structure()` — split paragraphs → sentences (`_split_sentences`) → words; returns stats + paragraph list |
| `keywords.py` | L2 | `extract_keywords()` — in-document TF-IDF over uni+bigram; filters bilingual stopwords (`stopwords.txt`) and pure numbers |
| `topics.py` | L3 | `extract_topics()` — LDA (seed 42), k chosen by **UMass coherence** over k=3…10; `_cluster_representatives()` — PCA + K-Means to pick readable topic labels from keyphrases |
| `fields.py` | L4 | `extract_fields(text, doc_type)` — regex schemas per class (invoice #, ISO dates, totals, parties…); per-class overrides in `configs/fields.yaml`; `_normalize()` value normalization |
| `summary.py` | L5 | `summarize_extractive()` — sentence scoring with L2 keyphrase priors + MMR (λ configurable, deterministic); `summarize_abstractive()` — beam-search generation from a local <7B checkpoint in `models/artifacts/` (`finetuned_checkpoint` preferred, else `checkpoint`); model load `lru_cache`d; missing model/deps → falls back to extractive. Config: `configs/summary.yaml` |
| `report.py` | seam | **the only entry point**: `understand(text)` / `understand_file(path)` → record `{source, doc_type, structure, keywords, topics, fields, summary, timing}`; `render_markdown(record)`; `_router_text()` (loads SVM artifacts) and `_router_image()` (loads the keras CNN) |

Routing contract: the router is **advisory**. If an artifact is missing the label is
`"unavailable"` and L1–L5 still run — the pipeline never hard-fails on a missing model.

## Conventions (whole tree)

1. **Deterministic / seeded** — same input ⇒ same output (seed 42 everywhere).
2. **One seam** — CLI, UI and tests all go through `nlp.report.understand()`.
3. **Understanding-first** — L1–L3 are the product; the router is a label.
4. **No OCR, no cloud LLM** — a local <7B seq2seq model is the only generative component.
5. **Config, not constants** — hyperparameters live in `configs/`, not in code.
