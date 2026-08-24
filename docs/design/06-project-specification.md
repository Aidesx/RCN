# Project Specification v4 — Document Understanding Project (RCN)

**Role:** Project Lead, ML Architect, Technical Reviewer
**Date:** 2026-08-23
**Status:** SOURCE OF TRUTH (§7 change control). Supersedes `archive-objective-v1/06` (v2.x series) — old set kept at `design/archive-objective-v1/`.
**Version:** 4.3 (2026-08-24) — Deepen pass 1 (course-anchored, still offline/deterministic): **L3 v2** topic labels — each LDA topic carries a keyphrase `label` chosen via seeded K-means (PCA-reduced TF-IDF vectors over the paragraph mini-corpus, Chap6) over L2 keywords; **L4 v2** config-driven schemas — `configs/fields.yaml` overrides per-class patterns + `normalize` (date/money/upper) + `required` validation (`missing_required` in record + Markdown warning); **Router v2** — SVM label gated by `classification.confidence_threshold` (0.5, softmax over margins), deterministic rule-cue fallback, `via` ∈ {e0b_svm, rule_cues, low_confidence}; low-confidence docs no longer guess (D3). Test suite: **167/167 pass**.
**Version:** 4.2 (2026-08-23) — **L4 field extraction implemented** (user-approved promotion from FUTURE): `nlp/fields.py` — regex schemas per class (invoice/receipt/report/letter/form/article) + generic date/amount fallback; wired into report (`fields` section, JSON+Markdown). Acceptance artifacts landed: `runs/E-U0/results.json` (252 docs: raw-frequency stopword leak **20.87%** vs TF-IDF **0%**, Jaccard overlap 0.271 → TF-IDF wins E-U0) and `runs/E-U2/coherence_curve.json` (k=3..10 curve on corpus; **argmax k=7**, UMass −0.270). Test suite: **151/151 pass**.
**Version:** 4.1 (2026-08-23) — Understanding pipeline IMPLEMENTED end-to-end: `nlp/keywords.py` (L2, paragraph-mini-corpus TF-IDF per D1, bilingual stopwords, uni+bigrams with overlap suppression); `nlp/topics.py` (L3 sklearn LDA seed 42, k∈3..10 by in-repo UMass coherence, short-text fallback); `nlp/report.py` (`understand()` seam, advisory router text-SVM/image-CNN with graceful "unavailable", deterministic `render_markdown()`); CLI `scripts/understand_text.py` (file/folder-recurse/--demo → JSON + Markdown, exit codes). Smoke: invoice folder → correct labels + keyphrases ("invoice inv-10000", "invoice number", dates…). Test suite: **139/139 pass**.
**Version:** 4.0 (2026-08-23) — spec authored for corrected objective (understanding-first; classifier = router; MobileNetV2/E5 FUTURE).
**Inputs:** `01-knowledge-audit.md` … `05-implementation-roadmap.md` (v2 set).

---

## 0. Why this specification exists (honest changelog)

The original objective — *page-type classification with a self-built CNN as the headline* —
did not match the owner's actual goal. On 2026-08-22 the owner corrected it: the project must
**understand document content** (structure → keywords → topics), with classification demoted to
a router. Consequences adopted:

1. Understanding layers L1/L2/L3 are the product; L1 already built.
2. Trained CNN (E1) and text SVM (E0b) are reused **as routers**; no further model training required.
3. MobileNetV2 (E3/E4), fusion (E5) move to **FUTURE**.
4. All previously passed gates remain valid for what they measured; nothing is re-run without cause.

## 1. Objective & scope

Build an offline CLI that turns text or text-based documents into a **document understanding
report**: L1 structure (word→sentence→paragraph + counts), L2 top-k keyphrases (in-document
TF-IDF over paragraph mini-corpus), L3 topics (LDA, k by UMass coherence), plus an advisory
6-class router label. Outputs: JSON + Markdown + terminal summary.

**Boundaries:** no OCR · no LLMs/network · no QA/summarization/translation · no layout extraction ·
scanned pages/images get classification only. Details: 02 §5.

## 2. Architecture summary

`io/` ingestion (built) → `nlp/structure.py` (built) → `nlp/keywords.py` → `nlp/topics.py` →
router artifacts (built) → `nlp/report.py` assembly → CLI. Design decisions D1–D4 in 03 §3
(paragraph-mini-corpus TF-IDF; sklearn LDA + in-repo UMass; advisory router; one report seam).

## 3. Data

| Asset | Role |
|---|---|
| Frozen image split (700 pages, 6 classes) | router R1 evidence (E1 numbers stand) |
| Synthetic text corpus (360 docs, seed 42) | E-U experiments material + demos |
| Owner-provided real texts | qualitative review |

No new data collection. Caveats carried: synthetic corpus ≠ hard benchmark; Vietnamese
whitespace-tokenization limitation documented.

## 4. Experiments summary

E-U0 keyword-scorer baseline · E-U1 keyphrase determinism/coverage · E-U2 topic coherence curve ·
R1/R2 router gates (recorded) · E-R report contract. Criteria: 04 §5.

Recorded results so far (unchanged): E1 image CNN — accuracy 0.5714, macro-F1 0.5143,
gate PASS vs majority baseline 0.2857 · E0b text SVM — synthetic-corpus test 1.000 (plumbing check).

## 5. Roadmap

Remaining stages S9' keywords → S10' topics → S11' report → S12' CLI → S13' final suite — **ALL COMPLETE** (verified 2026-08-24, 167/167 tests). Deepen pass 1 (v4.3): topic labels, config-driven field schemas, router confidence gate.
FUTURE: MobileNetV2, fusion, QA. Details: 05 §2–§4.

## 6. Acceptance criteria (v4 deliverable)

1. One command understands any supported input offline, deterministically.
2. Report contains complete L1+L2+L3 sections + L4 fields + router label (or graceful "unavailable").
3. Coherence-based k-selection documented with curve artifact.
4. Test suite >60 passing at final state (actual **167/167**).
5. README/ARCHITECTURE match the built system.

## 7. Change control (§7)

Spec changes precede code changes; every material change appends a version note here:
new version line on top, date + summary, never silent edits.
