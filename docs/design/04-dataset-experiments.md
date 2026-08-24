# Dataset & Experiments v2 — Document Understanding Project (RCN)

**Role:** Data & Experiment Owner
**Date:** 2026-08-23 (v2 — corrected objective; supersedes `archive-objective-v1/04`)
**Inputs:** 02 v2, 03 v2. Existing assets are REUSED; no new data collection required.

---

## 1. Data assets

| Asset | Location | Size | Used for | Status |
|---|---|---|---|---|
| Page-image dataset (RVL-CDIP subset + SROIE receipts) | `datasets/raw`, frozen split `datasets/splits/manifest.csv` | 700 pages, 6 classes, 490/105/105 doc-level | Router R1 numbers (already measured as E1) | frozen ✅ |
| Synthetic text corpus (templated, seed 42) | `datasets/text`, split `text_manifest.csv` | 360 docs, 60/class, txt/md/html | L3 topic experiments + R2 router | ✅ reusable |
| Real user texts / project docs | ad-hoc inputs | — | qualitative review of L1/L2/L3 output | collected on demand |

No new dataset construction is in scope. The synthetic corpus caveat carries over: it
validates plumbing and gives stable demo material; quality claims come from the metrics below,
not from "accuracy" on templated text.

## 2. Metrics for understanding layers

| Layer | Metric | Why |
|---|---|---|
| L2 Keywords | **Determinism** (byte-identical repeat), **coverage** (% of top-k tokens appearing ≥2× in doc), **stopword leakage** = 0, qualitative top-10 review | no gold labels exist; these prove the ranking is stable and sane |
| L3 Topics | **UMass coherence** (in-repo) across k∈3..10 with argmax selection; per-topic top-word inspection; document-topic mixture sums to 1 | coherence is the standard unsupervised topic-quality proxy |
| Router | reuse E1 image gate + E0b text gate (already PASSed) | no retraining |
| Whole report | schema test: every record has all sections; Markdown renders | contract |

## 3. Experiment matrix

| # | Name | Question | Setup | Success criteria |
|---|---|---|---|---|
| **E-U0** | Keyword scorer baseline | Does TF-IDF beat raw frequency? | frequency-ranking vs D1 TF-IDF ranking on corpus docs; compare overlap of top-k with stopwords/generic terms | TF-IDF top-k contains fewer function words than frequency baseline (qualitative + stopword-leak=0) |
| **E-U1** | Keyphrases | Are keyphrases useful & deterministic? | `extract_keywords` on all 360 corpus docs + 5 real docs; k=10 | byte-identical on re-run; coverage reported; bigrams present when meaningful |
| **E-U2** | Topics | What k fits? Are topics interpretable? | LDA over corpus (and per-single-doc mode for long texts); coherence curve k=3..10; fixed seed | coherence curve produced; chosen k documented; topics have non-degenerate word lists |
| **R1/R2** | Router | Which class label to attach? | reuse E1 (image) and E0b (text) artifacts on their frozen tests | previously recorded gates stand (E1 57.1%/PASS; E0b 1.0/PASS w/ caveat) |
| **E-R** | Report contract | Is the assembled record complete & stable? | golden-schema test on sample docs incl. Vietnamese | all fields present; Markdown renders; deterministic |

## 4. Evaluation protocol

Frozen splits unchanged. All understanding functions are pure/seeded: same input ⇒ same output
(asserted by tests). Latency is not a headline metric (offline tool), but a single timing line
is logged per report run.

## 5. Acceptance criteria (this objective)

| Area | Minimum acceptance | Result (2026-08-23) |
|---|---|---|
| L2 | determinism ✓, zero stopword leakage, coverage reported, E-U0 comparison documented | ✅ tests + `runs/E-U0/results.json`: freq leak **20.87%** vs TF-IDF **0%**, overlap 0.271 |
| L3 | coherence curve computed, argmax-k used, topics non-empty, mixtures sum≈1 | ✅ `runs/E-U2/coherence_curve.json`: k=3..10, **argmax k=7** (−0.270) |
| L4 (promoted) | per-class schemas match on corpus samples; generic fallback; deterministic | ✅ invoice 5/5, receipt 4/4, article 3/3 on smoke; fallback tested |
| Router | labels attached from existing artifacts; graceful `"unavailable"` if artifacts missing | ✅ unchanged gates stand |
| Report | JSON schema complete for every input type in 02 §3; deterministic | ✅ schema test incl. `fields`; byte-identical repeats |
| Tests | >60 pass | ✅ **167/167** |

## 6. Recorded experiment results

- **E-U0** (`runs/E-U0/results.json`): raw-frequency top-10 leaks stopwords into 20.87% of slots
  across 252 train docs; TF-IDF keyphrases leak 0%; Jaccard overlap 0.271 → the two rankings
  genuinely differ and TF-IDF satisfies the hygiene criterion.
- **E-U2** (`runs/E-U2/coherence_curve.json`): UMass curve k=3..10 →
  [(3,−6.35),(4,−2.37),(5,−5.63),(6,−0.32),(7,−0.270),(8,−2.42),(9,−1.16),(10,−3.27)];
  argmax **k=7** selected; per-topic top words stored.
- **R1/R2**: stand as recorded (E1 57.1%/PASS · E0b synthetic 1.000 with caveat).

## 7. Output artifact

Anchors for 05 roadmap: experiments E-U0/U1/U2/R/E-R with criteria above.
