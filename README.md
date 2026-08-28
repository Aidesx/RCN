# RCN — Narrow AI Document Understanding

Offline document understanding without OCR. A CLI application that ingests text, files, and folders and produces a layered understanding record — structure, keywords, topics, fields, and summary — entirely on-device.

Documents are routed by type: TF-IDF + SVM for text and a small CNN for images/scans. Scanned pages are classified only and never OCR'd. Summarization is extraction-first (MMR over L1+L2) with an optional small local seq2seq model (<7B) and automatic fallback to extractive. All outputs are deterministic and seeded.

---

## Quick links

- [Architecture](./ARCHITECTURE.md) — system overview and pipeline
- [In-repo Code Map](./src/ARCHITECTURE.md) — per-module implementation details
- [Contributing](./CONTRIBUTING.md) — how to work in this repo

---

## Tech stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11 |
| Text classification | scikit-learn TF-IDF + LinearSVC / RandomForest + GridSearchCV (5-fold, f1_macro) |
| Image classification | TensorFlow 2.21 — Architecture A: Conv2D(32,5)->MaxPool->Conv2D(64,5)->MaxPool->Dense(256)->Dropout(0.5)->Softmax |
| NLP | In-document TF-IDF (uni+bigram, bilingual stopwords), LDA (sklearn, seed 42, UMass coherence), regex schemas per class |
| Summarization | Extraction (MMR, λ=0.7) + Abstractive mT5/vit5-base (<7B) via `transformers` + `Seq2SeqTrainer` |
| Document I/O | PyMuPDF, pypdf, pdfplumber, python-docx, BeautifulSoup4, html2text |
| App | Streamlit 1.62 — `scripts/app.py` (RCN Studio) |
| Tests / Lint | pytest 9.1.1, ruff 0.16.4 |

---

## Getting started

Create the environment and install pinned dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Download the summarization checkpoint once (mT5 XLSum multilingual, <7B):

```bash
python scripts/download_summarizer.py
```

Run the understanding pipeline from the CLI:

```bash
# single text
python scripts/understand_text.py --text "Your document text here"

# single file (pdf-text, docx, md, html, png/jpg)
python scripts/understand_text.py --file datasets/text/invoice/invoice_0001.txt

# folder (batch)
python scripts/understand_text.py --folder datasets/text --out runs/demo

# demo fixture (no input required)
python scripts/understand_text.py --demo
```

Run the web UI (Streamlit):

```bash
.venv/Scripts/python -m streamlit run scripts/app.py
```

| Surface | URL | Notes |
| --- | --- | --- |
| RCN Studio | `http://localhost:8501` | Streamlit app in `scripts/app.py`. Batch mode auto-scans `datasets/text`. |
| CLI JSON | `stdout` + `*.understanding.json` | Same `understand()` seam as the UI. |
| CLI Markdown | `*.understanding.md` | Rendered via `docproc.nlp.report.render_markdown`. |

Run tests and lint:

```bash
python -m pytest -q
python -m ruff check src scripts
```

Fine-tune the summarizer (single script works on CPU or GPU/Colab):

```bash
python scripts/train_summarizer.py \
  --train rcn-aux/datasets/summary/xlsum_vietnamese_train.jsonl \
  --val   rcn-aux/datasets/summary/xlsum_vietnamese_val.jsonl \
  --epochs 2 --batch-size 8 --base-model VietAI/vit5-base
```

| Input | Handling |
| --- | --- |
| PDF (text layer) | Deterministic text extraction (text branch) |
| PDF (scanned) | Render page → image branch (classify only, **no OCR**) |
| DOCX / Markdown / HTML | Parse → extract text |
| PNG / JPG | Image branch directly |

| Output | Format |
| --- | --- |
| Structured record | JSON (`doc_type`, `structure`, `keywords`, `topics`, `fields`, `summary`) |
| Report | Markdown |
| Summary | Terminal / UI |

---

## Repository structure

```
AGENTS.md                   Agent context — read first
README.md                   This file
ARCHITECTURE.md             System overview and pipeline (this repo)
src/ARCHITECTURE.md         Per-module implementation map
configs/                    All hyperparameters + fields.yaml (schema overrides)
  dataset.yaml              Classes + split 70/15/15
  text.yaml                 TF-IDF + SVM/RF GridSearch
  cnn.yaml                  Architecture A (64×64) + finetune (224×224)
  finetune.yaml             MobileNetV2 phases (FUTURE)
  pipeline.yaml             classification.confidence_threshold
  summary.yaml              Summary mode + abstractive checkpoint
  fields.yaml               L4 regex schema overrides per class
src/docproc/                Core — see src/ARCHITECTURE.md
  paths.py                  Single source of truth for layout + config loading
  io/                       detect (magic bytes + scan probe) · parsers · render
  preprocess/               image (64×64 / 224×224 bicubic) · text (TF-IDF wrapper)
  models/                   cnn.py (Architecture A) · text_classifier.py (SVM/RF)
  training/                 data.py (2-arm registry) · harness.py (seeded fit)
  evaluation/               metrics.py · report.py (shared across experiments)
  nlp/                      structure (L1) · keywords (L2) · topics (L3 LDA+UMass)
                            fields (L4) · summary (L5 extractive+abstractive) · report (seam)
docs/design/                01-knowledge-audit → 06-project-specification + 07-summary
datasets/                   raw/ text/ splits/ (gitignored, see PROVENANCE.csv)
models/artifacts/           Trained artifacts (gitignored): text_vectorizer.joblib,
                            text_model_svm.joblib, summarizer_mt5/, vit5_finetuned/
runs/                       Per-experiment logs + metrics (gitignored): E1/, E0b/, E-U0/ ...
scripts/                    CLI + tooling: understand_text.py, app.py, train_summarizer.py
requirements.txt            Pinned versions verified in .venv
```

