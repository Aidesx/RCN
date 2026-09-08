# Test suite map

How the `tests/` tree is organized, what each directory covers, and how the suite is split
between functional and model-dependent tests. Tests mirror `src/docproc/` module by module.

## Structure

```text
tests/
├── system.md                   ← this file
├── fixtures/                   shared sample documents (PDF, DOCX, images, …)
├── golden/                     frozen numpy arrays (preprocessing)
│
├── io/                         ingestion: detect file type + parse text + render pages
│   ├── test_detect.py         magic bytes + scanned-PDF probe → classification
│   ├── test_parsers.py        text from PDF/DOCX/MD/HTML (golden-tested)
│   └── test_render.py         PDF → images + embedded images
│
├── preprocess/                 model-input preparation
│   ├── test_image.py          bicubic resize → tensor (golden: pixel-exact vs .npy)
│   └── test_text.py           TF-IDF vectorizer wrapper (joblib save/load)
│
├── models/                     model architecture definitions
│   └── test_cnn.py            Architecture A: Conv2D→MaxPool→Dense→Softmax
│
├── training/                   training infrastructure
│   ├── test_data.py           two-arm dataset registry (64×64 / 224×224)
│   └── test_harness.py        seeded fit + config snapshot + EarlyStopping
│
├── evaluation/                 measurement
│   ├── test_metrics.py        accuracy, macro-F1, confusion matrix, acceptance gate
│   └── test_eval_report.py    frozen-test report + learning curves
│
├── nlp/                        the document-understanding core (L1–L5)
│   ├── test_structure.py      L1: word → sentence → paragraph + stats
│   ├── test_keywords.py       L2: top-k keyphrases (in-doc TF-IDF, uni+bigram)
│   ├── test_topics.py         L3: LDA + UMass coherence k-selection
│   ├── test_fields.py         L4: per-class regex schema (invoice/receipt/…)
│   ├── test_summary.py        L5: extractive MMR + abstractive seq2seq + fallback
│   └── test_report.py         understand() seam + render_markdown()
│
├── text_classifier/            text router
│   └── test_baseline.py       GridSearchCV SVM/RF + artifact dump
│
├── dataset/                    data module
│   └── test_module.py         manifest + 70/15/15 split + leak check
│
└── ui/                         Streamlit UI (AppTest — no real browser)
    ├── test_smoke.py          renders without crashing; demo mode; real-text paste
    ├── test_features.py       highlighting, coherence chart, history, batch + CSV
    └── test_e2e_core.py       E2E core flows (extractive/abstractive/upload/batch)
```

## Running

```bash
# From the repo root RCN/
python -m pytest -q              # default: 177 FUNCTIONAL tests (no model artifacts needed)
python -m pytest -q -m model     # 12 MODEL-DEPENDENT tests (SVM / keras CNN / seq2seq checkpoint)
python -m pytest tests/io/ -q    # just ingestion
```

## Role split (2026-09-08)

- **Default suite (`pytest -q`) = functional.** L1–L5, IO, dataset, text-classifier logic, UI
  chrome — deterministic and green even on a machine with no trained artifacts
  (`pyproject.toml` sets `addopts = -m "not model"`).
- **`pytest -m model` = model-dependent.** Needs the real artifacts:
  `tests/ui/*` (UI runs the *real* core: SVM + abstractive checkpoint),
  `tests/training/test_harness.py` (trains the keras CNN), the `TestRouterGate` group (4 tests,
  real SVM), `test_abstractive_smoke_when_checkpoint_present`, and
  `test_e1_report_matches_recorded_metrics`.
- **Model quality is not measured by unit tests.** Summarizer quality has its own benchmark —
  ROUGE-2 / verbatim-copy / number-hallucination on a bundled 12-document Vietnamese eval set;
  results and reproducer live in `benchmarks/` (see `benchmarks/RESULTS.md`).

## Conventions

- **Deterministic** — every test seeds 42; same input → same output.
- **Golden-tested** — file IO is compared against `.expected.txt` fixtures; image tensors are
  compared pixel-exact against frozen `.npy` arrays.
- **UI tests** use `streamlit.testing.v1.AppTest` — no browser required.
- **Fallback-safe** — abstractive tests self-skip when the checkpoint is missing.
