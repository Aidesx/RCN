# Knowledge Audit v2 — Document Understanding Project (RCN)

**Role:** Learning-path Auditor
**Date:** 2026-08-23 (v2 — objective corrected; supersedes archived `archive-objective-v1/01`)
**Inputs:** user objective decision 2026-08-22 ("hiểu tài liệu": word → sentence → paragraph → keywords → topics), course syllabus chapters, existing RCN implementation.
**Rule:** every planned technique maps to taught material or is explicitly marked `EXTENSION` / `FUTURE`. Nothing enters because it "looks impressive".

---

## 1. Corrected Objective (why this audit was redone)

The original plan centered on **page-type classification with a self-built CNN**. The real
objective is **document understanding**: given text or a text-based document, the system must
report *what the document contains* — its language structure (word/sentence/paragraph), its
key terms, and its topics. Classification survives as a **router** that picks how to present
results, not as the headline. This file re-audits knowledge against the corrected goal.

## 2. Technique inventory vs. the corrected goal

| # | Technique | Course anchor | Status for this project | Used where |
|---|-----------|---------------|--------------------------|------------|
| 1 | Tokenization & segmentation rules (paragraph/sentence/word) | Programming + NLP basics | **DIRECT** | L1 `nlp/structure.py` (built) |
| 2 | TF-IDF weighting | Ch5/Ch7 vectorization | **DIRECT** | L2 keywords |
| 3 | N-gram keyphrases (1–2 grams) | Ch5 feature engineering concept | DIRECT (small step from #2) | L2 keyphrases |
| 4 | Stopword filtering | Ch5 preprocessing | DIRECT | L2 |
| 5 | Topic modeling (LDA) | Ch7 topic modeling | **DIRECT** (audit v1 marked it deferred; now core) | L3 topics |
| 6 | UMass coherence (topic quality score) | Not taught hands-on | **EXTENSION** (closed-form metric, ~20 lines, explainable) | L3 k-selection |
| 7 | CountVectorizer / sklearn pipelines | Taught tooling | DIRECT | L3 |
| 8 | Regex field extraction | Programming | DIRECT — **BUILT** (`nlp/fields.py`, L4 promoted 06 v4.2) | L4 fields |
| 9 | CNN image classification (Ch9 template) | Taught end-to-end | **BUILT & PROVEN** (E1 gate passed) | Router R1 |
| 10 | TF-IDF + SVM/RF text classification (Ch5/Ch7) | Taught | **BUILT & PROVEN** (E0b) | Router R2 |
| 11 | GridSearchCV model selection | Taught | BUILT | router artifacts |
| 12 | Deterministic parsing (PDF/DOCX/MD/HTML) | Library usage, no ML | BUILT (`io/`) | ingestion |

## 3. Deliberately NOT used

| Technique | Why excluded |
|-----------|--------------|
| LLM APIs / weak models | User decision: course knowledge first; offline constraint |
| OCR | `OUT_OF_SCOPE` (unchanged); input understanding is text-first |
| Word embeddings / transformers | Not needed for L2/L3; would be `FUTURE` |
| TextRank / graph keyphrases | `EXTENSION` not needed — TF-IDF variant suffices and stays explainable |
| gensim coherence c_v | Extra dependency rejected; UMass implemented in-repo (~20 lines) |

## 4. Gaps & risks against the corrected goal

1. **Vietnamese tokenization** — whitespace tokens over-segment Vietnamese compound words;
   acceptable for L1/L2 stats, noted as limitation in reports. No external segmenter (dependency discipline).
2. **Topic interpretability** — LDA on small corpora yields broad topics; mitigated by fixed seed,
   coherence-guided k, and top-word inspection in the report.
3. **Keyphrase ground truth absent** — no labeled gold keyphrases; evaluation relies on
   determinism, coverage metrics and qualitative review instead of accuracy claims.
4. **Router reuse** — E1 CNN was trained on page images of scanned docs; routing *text-based*
   documents uses the E0b SVM instead. Both already meet their gates; no retraining required.

## 5. Output artifact

This file anchors the rewritten design set (`02→06`). Change control follows §7 of
`06-project-specification.md` (spec first, then code).
