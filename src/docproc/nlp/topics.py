"""L3 topics: sklearn LDA (seeded) with k selected by in-repo UMass coherence."""
from __future__ import annotations

import math

from docproc.nlp.keywords import _content_tokens, _tokens, extract_keywords
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
    """LDA topics (+ per-topic keyphrase label & doc mixture); k=None → argmax UMass."""
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

    # L3 v2: attach a keyphrase label per topic (K-means/PCA over L2 keywords).
    kw = extract_keywords(text, k=max(10, TOP_N * 2))
    kw_items = [(item["term"], item["score"]) for item in kw["keywords"]]
    labels = _topic_labels([t["top_words"] for t in topics], kw_items,
                           para_tokens)
    for topic, label in zip(topics, labels):
        topic["label"] = label

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


def _keyword_vectors(keyword_terms: list[str],
                     para_tokens: list[list[str]]) -> dict[str, list[float]]:
    """TF-IDF vector of each keyword over the paragraph mini-corpus,
    unit-normalized (L2-style in-document weighting, D1)."""
    from collections import Counter

    if not para_tokens:
        return {}
    paras = [Counter(p) for p in para_tokens]
    n_paras = len(paras)
    df: Counter[str] = Counter()
    for p in paras:
        df.update(p.keys())

    def idf(tok: str) -> float:
        return math.log((1 + n_paras) / (1 + df.get(tok, 0))) + 1.0

    out: dict[str, list[float]] = {}
    for term in keyword_terms:
        toks = term.split()
        vec = [sum(p.get(t, 0) * idf(t) for t in toks) for p in paras]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            out[term] = [x / norm for x in vec]
    return out


def _overlap(topic_words: list[str], term: str) -> int:
    tokens = set(term.split())
    return sum(1 for w in topic_words if w in tokens)


def _cluster_representatives(kw_items: list[tuple[str, float]],
                             vecs: dict[str, list[float]],
                             n_topics: int,
                             para_tokens: list[list[str]]) -> list[str]:
    """Theme representatives: seeded K-means on normalized keyword vectors
    (PCA-reduced when the mini-corpus is wide); one keyword per cluster
    (nearest to its centroid). Falls back to the raw keyword list."""
    pool = [(t, s) for t, s in kw_items if t in vecs]
    if len(pool) < 2 or len(para_tokens) < 2:
        return [t for t, _ in pool]

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import Normalizer

    X = np.asarray([vecs[t] for t, _ in pool], dtype=float)
    if X.shape[1] > 10:
        X = PCA(n_components=10, random_state=SEED).fit_transform(X)
    X = Normalizer().fit_transform(X)

    k = min(n_topics, len(pool))
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
    reps: list[str] = []
    for c in range(k):
        members = [i for i, lab in enumerate(km.labels_) if lab == c]
        cent = km.cluster_centers_[c]
        i = max(members, key=lambda i: float(X[i] @ cent))
        reps.append(pool[i][0])
    return reps


def _topic_labels(topics_words: list[list[str]],
                  kw_items: list[tuple[str, float]],
                  para_tokens: list[list[str]]) -> list[str]:
    """One distinct keyphrase label per topic, deterministic (greedy best
    match against theme representatives, tie-break by L2 score)."""
    kw_scores = dict(kw_items)
    terms = [t for t, _ in kw_items]
    vecs = _keyword_vectors(terms, para_tokens)
    reps = _cluster_representatives(kw_items, vecs, len(topics_words),
                                    para_tokens)

    labels: list[str] = []
    used: set[str] = set()
    for tw in topics_words:
        cands = [t for t in reps if t not in used]
        if not cands:
            cands = [t for t in terms if t not in used]
        if not cands:
            labels.append(tw[0] if tw else "")
            continue
        best = max(cands, key=lambda t: _overlap(tw, t) * 10000
                   + kw_scores.get(t, 0.0))
        labels.append(best)
        used.add(best)
    return labels


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
        "topics": [{"id": 0, "top_words": top, "label": top[0] if top else ""}],
        "doc_topic_mixture": [1.0],
    }