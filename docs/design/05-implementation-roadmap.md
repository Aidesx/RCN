# Implementation Roadmap v2 — Document Understanding Project (RCN)

**Role:** Senior ML Engineer & Software Architect
**Date:** 2026-08-23 (v2 — corrected objective; supersedes `archive-objective-v1/05`)
**Inputs:** 03 v2, 04 v2. Principle unchanged: verify each layer behind tests before wiring the next; MVP first.

---

## 1. Stage status (recap of executed work under old plan — all reused)

| Done | Asset |
|---|---|
| S0–S1 | environment, scaffold, configs, `docproc.paths` |
| S2 | image dataset + frozen split (700 pages) |
| S3 | preprocessing (image tensors 64/224, TF-IDF wrapper) |
| S4–S6 | CNN Architecture A trained & gated (E1: 57.1% test, PASS) |
| S7 | document I/O (detect/parse/render, golden-tested) |
| S8 | text baseline E0b (SVM artifacts) + L1 structure module |

## 2. Stages (new numbering — S9'–S13' ALL COMPLETE, verified 2026-08-24, suite 151/151)

### S9' — Keywords layer (L2) ✅ done
- **Build:** `nlp/keywords.py` per 03 §D1 (paragraph mini-corpus TF-IDF, stopwords bundled, uni+bigrams, top-k no-overlap).
- **Gate:** E-U0 comparison documented; determinism + zero-stopword-leak tests pass.
- Deps: none new. 

### S10' — Topics layer (L3) ✅ done
- **Build:** `nlp/topics.py`: CountVectorizer → LDA(seed 42), k∈3..10 by in-repo UMass coherence; return top words + doc mixture. Single-doc mode for short texts (fallback k=1 topic summary via highest-tf content words if too few paragraphs).
- **Gate:** E-U2 coherence curve produced on corpus; non-degenerate topics; seeded reproducibility test.
- Deps: stopwords list shared with S9' (bundled `stopwords.txt`).

### S11' — Understanding report assembly ✅ done
- **Build:** `nlp/report.py`: `understand(text, source)` → record `{source, doc_type, structure, keywords, topics, timing}`; `render_markdown(record)`; router wiring (image→CNN, text→SVM, missing→"unavailable" per D3).
- **Gate:** golden-schema test across every input type in 02 §3; byte-deterministic records.
- Deps: S9', S10', existing router artifacts.

### S12' — CLI understand ✅ done
- **Build:** extend `scripts/understand_text.py`: file / folder (recurse) / --demo; writes `<name>.understanding.json` + `.md`; exit codes 0/≠0.
- **Gate:** end-to-end on one file of each supported type; folder run produces N reports; demo works offline.

### S13' — Final test suite ✅ done
- **Build:** ~18 new tests (keywords 5, topics 5, report/router 5, CLI 3).
- **Gate:** full suite >60 passing (actual **167/167**); final commit state green.

### FUTURE (explicitly not this plan)
MobileNetV2 fine-tune (E3/E4) · fusion E5 · QA/summarization · OCR (out of scope).

## 3. Gates

| Gate | Definition |
|---|---|
| Keywords verified | determinism + stopword-leak=0 tests pass; E-U0 documented |
| Topics verified | coherence curve exists; argmax-k used; seeded repeat identical |
| Report contract | schema-complete deterministic records for all input types |
| CLI works | file+folder+demo offline with correct exit codes |
| Final acceptance | >60 tests pass at last commit state |

## 4. Priority

| Item | Priority |
|---|---|
| S9' keywords, S10' topics | MUST_HAVE (core objective) |
| S11' report assembly | MUST_HAVE |
| S12' CLI understand | MUST_HAVE (user-facing deliverable) |
| S13' suite >60 | MUST_HAVE |
| Router polish (provenance line in report) | SHOULD_HAVE |
| MobileNetV2, fusion, QA | FUTURE |

## 5. Repository deltas

```
src/docproc/nlp/{structure.py ✅, keywords.py ✅, topics.py ✅, report.py ✅, fields.py ✅, stopwords.txt ✅}
scripts/understand_text.py   (built)
rcn-tests/test_nlp_keywords.py · test_nlp_topics.py · test_nlp_report.py ✅ (151 tests total)
docs: README.md, ARCHITECTURE.md refresh (done)
```

## 6. Development order

Docs (this set) → S9' → S10' → S11' → S12' → S13'. Each stage lands with its tests before the next begins; spec changes ride ahead of code per §7.
