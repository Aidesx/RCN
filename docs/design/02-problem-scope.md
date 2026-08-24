# Problem Scope v2 — Document Understanding Project (RCN)

**Role:** Requirements & Scope Owner
**Date:** 2026-08-23 (v2 — corrected objective; supersedes `archive-objective-v1/02`)
**Inputs:** user decisions 2026-08-22/23 — understanding levels L2+L3 in scope; classifier demoted to router; input = text + text-based documents; course techniques + supporting libraries only; no LLMs.
**Status:** AUTHORITATIVE for what is in/out of scope. Architecture follows in 03.

---

## 1. Problem statement

> **Given a piece of text or a text-based document, produce a structured "document
> understanding report": the language structure of the content, its key terms, its
> topics, and a document-type label used to organize the output.**

One CLI command, offline, deterministic. Report = JSON (machine) + Markdown (human).

## 2. Understanding layers (the product)

| Layer | Question answered | Output | Status |
|---|---|---|---|
| **L1 Structure** | How is the text organized? | paragraphs → sentences → words + counts | ✅ built (`nlp/structure.py`) |
| **L2 Keywords** | What is it about, term-wise? | top-k keyphrases (1–2 grams), in-document TF-IDF | ✅ built (`nlp/keywords.py`) |
| **L3 Topics** | What themes run through it? | k topics (top words) + per-doc topic mix | ✅ built (`nlp/topics.py`) |
| Router label | What kind of document? | one of 6 classes | ♻️ reuse trained CNN/SVM |
| **L4 Field extraction** | Exact values (numbers, dates, parties)? | fields per class schema | ✅ built (`nlp/fields.py`, promoted 06 v4.2) |
| QA / Summarization | Answer questions / shorten | — | **FUTURE** |

## 3. Inputs

| Input | Handling |
|---|---|
| Plain text / Markdown file | direct to understanding |
| HTML file | parse → text → understanding |
| DOCX file | parse → text → understanding |
| PDF **with text layer** | parse → text → understanding |
| Scanned PDF / image | **classification only** (CNN router label); no OCR — never read |
| Folder | recurse, one report per file |

## 4. Outputs

- Per document JSON record: `{source, doc_type(router), structure(L1), keywords(L2), topics(L3)}`
- Human-readable Markdown rendering of the same record
- Terminal summary line

## 5. Non-goals (unchanged, reinforced)

- ❌ OCR / reading scanned text (`OUT_OF_SCOPE`)
- ❌ LLM APIs, weak models, network calls at runtime
- ❌ QA, summarization, translation (FUTURE, not promised)
- ❌ Layout/table extraction
- ❌ Re-training vision models for comparison's sake (MobileNetV2 E3/E4 → FUTURE)

## 6. Deliverables & acceptance (summary)

1. Understanding pipeline L1+L2+L3 behind one function.
2. Router reuses existing trained artifacts (CNN `best.keras`, SVM joblib).
3. CLI: understand a file / folder / demo text.
4. Deterministic outputs: same input ⇒ byte-identical report.
5. Test suite >60 passing (current **151/151**, well above target).

## 7. Users & usage

Single-user local tool (project owner + grader). No serving/API. Runs on CPU-only VM/laptop.

## 8. Output artifact

This file is scope-authoritative; conflicts resolve in favor of this document via §7 of `06-project-specification.md`.
