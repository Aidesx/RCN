# System Architecture v2 — Document Understanding Project (RCN)

**Role:** AI/ML System Architect
**Date:** 2026-08-23 (v2 — corrected objective; supersedes `archive-objective-v1/03`)
**Inputs:** `01-knowledge-audit.md` v2, `02-problem-scope.md` v2
**Principle:** understanding-first; ML only where taught and already proven; every component deterministic or seeded.

---

## 1. Pipeline

```
Input (text / .txt / .md / .html / .docx / PDF-text)     [image/scanned-PDF → classify-only path]
        │
[1] File detection (magic bytes + scanned probe)          io/detect.py      (built)
[2] Text extraction per format                            io/parsers.py     (built)
        │
[3] UNDERSTANDING                                         nlp/*
    ├── L1 Structure: word → sentence → paragraph         nlp/structure.py  (built)
    ├── L2 Keywords: top-k keyphrases, in-doc TF-IDF      nlp/keywords.py   (built)
    ├── L3 Topics: LDA + UMass coherence k-selection      nlp/topics.py     (built)
    └── L4 Fields: per-class regex schemas + fallback     nlp/fields.py     (built)
        │
[4] Router label                                          models/* artifacts (built, reused)
    ├── text-based doc  → TF-IDF+SVM (E0b artifact)
    └── image/scanned   → CNN 64×64 (E1 best.keras)
        │
[5] Report assembly                                       nlp/report.py     (built)
        │
[6] Output: JSON + Markdown + terminal summary            scripts/understand_text.py
```

## 2. Component responsibility

| Module | Responsibility | Key interface | Built? |
|---|---|---|---|
| `io/` | detect type, parse text, render pages | `detect_file_type`, `extract_text` | ✅ |
| `nlp/structure.py` | L1 hierarchy + counts | `analyze_structure(text)` | ✅ |
| `nlp/keywords.py` | L2 top-k keyphrases | `extract_keywords(text, k=10)` | ✅ |
| `nlp/topics.py` | L3 topics + coherence k-selection | `extract_topics(text, k=None)` | ✅ |
| `nlp/fields.py` | L4 fields per class schema + generic fallback | `extract_fields(doc_type, text)` | ✅ |
| `nlp/report.py` | assemble understanding record + Markdown | `understand(text, source)` / `render_markdown(record)` | ✅ |
| `models/cnn.py`, artifacts | router (images) | predict fn on 64×64 tensors | ✅ |
| `models/text_classifier.py`, artifacts | router (text docs) | SVM + vectorizer joblib | ✅ |
| `scripts/understand_text.py` | CLI entry | file/folder/demo | ✅ |

## 3. Design decisions

### D1 — Keywords = in-document TF-IDF over a paragraph mini-corpus
Treat the document's own paragraphs as the document frequency corpus:
`score(term) = tf(term, doc) × idf_paragraphs(term)`. Terms frequent in the whole document
but concentrated in few paragraphs rank highest — generic glue words score low without any
external corpus or hardcoded IDF table. Stopwords filtered via a small bundled list.
N-grams: unigrams + bigrams, bigram score = mean of member unigram scores (deduplicated,
no overlap in output top-k). **Explainable, deterministic, zero dependencies.**

### D2 — Topics = sklearn LatentDirichletAllocation, k chosen by UMass coherence
`CountVectorizer(stopwords) → LDA(seed 42)`; candidate k ∈ 3..10; pick argmax UMass coherence
computed in-repo (~20 lines, closed-form over top-N word co-document counts); return top words
per topic + document-topic mixture. Fixed seed ⇒ reproducible.
**L3 v2 (v4.3):** each topic carries a keyphrase `label` — theme representatives chosen by
seeded K-means over unit-normalized TF-IDF keyword vectors (PCA-reduced when the paragraph
mini-corpus is wide), course-anchored in Chap6; greedy per-topic best match, distinct labels.

### D3 — Router is advisory, never blocking
Understanding works even if router artifacts are missing: report degrades gracefully
(`doc_type: "unavailable"`) instead of failing. Router adds organizational value only.
**Router v2 (v4.3):** SVM label is gated by confidence (`classification.confidence_threshold`,
softmax over one-vs-rest margins); below threshold a deterministic high-precision rule-cue
fallback may still label the doc (`via: "rule_cues"`), otherwise `via: "low_confidence"` with
no label — low-confidence docs never guess.

### D4 — One assembly function owns the contract
`nlp/report.understand()` is the single seam tests and CLI cross. Everything below it is swappable.

## 4. Risks

| Risk | Mitigation |
|---|---|
| Vietnamese whitespace tokenization over-splits compounds | documented limitation; stats unaffected structurally |
| LDA topics too broad on small texts | coherence-guided k + top-word inspection; fixed seed |
| Keyphrase quality has no gold labels | determinism + coverage metrics + qualitative review (04 §5–6) |
| Router mismatch (text classifier trained on synthetic corpus) | label is advisory; report states provenance |

## 5. Output artifact

Anchors for 04 (experiments) and 05 (roadmap): modules above, decisions D1–D4, risks table.
