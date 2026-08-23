"""L3 topics: sklearn LDA (seeded) with k selected by in-repo UMass coherence."""
from __future__ import annotations

import math

from docproc.nlp.keywords import _content_tokens, _tokens
from docproc.nlp.structure import analyze_structure

SEED = 42
MIN_K = 3
MAX_K = 10
TOP_N = 8


def _para_token_lists(text: str) -> list[list[str]]:
    structure = analyze_structure(text)
    return [
        _content_tokens(_tokens(" ".join(p["sentences"])))
        for p in structure["paragraphs"]
    ]


def extract_topics(text: str, k: int | None = None,
                   max_k: int = MAX_K) -> dict:
    """Return topics + mixture; k=None selects argmax UMass coherence."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    para_tokens = [p for p in _para_token_lists(text) if p]
    vocab = sorted({t for p in para_tokens for t in p})

    if k is not None and k <= 0:
        raise ValueError("k must be positive when provided")

    if len(para_tokens) < MIN_K or len(vocab) < MIN_K:
        return _fallback_topic(para_tokens)

    docs = [" ".join(p) for p in para_tokens]

    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    upper = min(max_k, max(MIN_K, n_topics_cap(len(para_tokens), len(vocab))))
    ks = [k] if k is not None else list(range(MIN_K, upper + 1))

    vectorizer = CountVectorizer(vocabulary=vocab)
    dtm = vectorizer.fit_transform(docs)

    results = []
    for kk in ks:
        lda = LatentDirichletAllocation(
            n_components=kk, random_state=SEED, learning_method="batch",
            max_iter=20,
        )
        topic_doc = lda.fit_transform(dtm)
        comp = lda.components_
        top_ids = [list(comp[t].argsort()[::-1][:TOP_N]) for t in range(kk)]
        coh = umass_coherence(dtm, top_ids)
        results.append({"k": kk, "coherence": coh, "lda": lda,
                        "topic_doc": topic_doc, "top_ids": top_ids})

    best = max(results, key=lambda r: r["coherence"])
    names = vectorizer.get_feature_names_out()
    topics = [
        {"id": t,
         "top_words": [names[i] for i in best["top_ids"][t]]}
        for t in range(best["k"])
    ]
    mixture = best["topic_doc"].mean(axis=0)
    mixture = (mixture / mixture.sum()).round(6).tolist()

    return {
        "k": int(best["k"]),
        "selected_by": "argmax_umass" if k is None else "fixed",
        "coherence_curve": [{"k": r["k"], "umass": round(float(r["coherence"]), 6)}
                            for r in results],
        "coherence": round(float(best["coherence"]), 6),
        "topics": topics,
        "doc_topic_mixture": mixture,
    }


def n_topics_cap(n_paras: int, vocab_size: int) -> int:
    return max(MIN_K, min(MAX_K, n_paras - 1, vocab_size // TOP_N))


def umass_coherence(dtm, top_ids_per_topic: list[list[int]], eps: float = 1e-12) -> float:
    """Mean UMass coherence over topics: for ordered top words (w_t, w_i), t<i:
    log((D(w_t & w_i) + eps) / D(w_t))."""
    binary = (dtm > 0).astype(int)
    co_docs = (binary.T @ binary)  # vocabulary x vocabulary co-document counts
    flat = co_docs.A.ravel() if hasattr(co_docs, "A") else co_docs.toarray().ravel()
    n_vocab = co_docs.shape[0]

    def d(i):
        return flat[i * n_vocab + i]

    def dj(i, j):
        return flat[i * n_vocab + j]

    scores = []
    for ids in top_ids_per_topic:
        topic_score = 0.0
        pairs = 0
        for m in range(1, len(ids)):
            for j in range(m):
                w_t, w_i = ids[j], ids[m]
                topic_score += math.log((dj(w_t, w_i) + eps) / (d(w_t) + eps))
                pairs += 1
        if pairs:
            scores.append(topic_score / pairs)
    return sum(scores) / len(scores) if scores else 0.0


def _fallback_topic(para_tokens: list[list[str]]) -> dict:
    from collections import Counter

    freq = Counter(t for p in para_tokens for t in p)
    top = [w for w, _ in freq.most_common(TOP_N)]
    return {
        "k": 1,
        "selected_by": "fallback_short_text",
        "coherence_curve": [],
        "coherence": None,
        "topics": [{"id": 0, "top_words": top}],
        "doc_topic_mixture": [1.0],
    }