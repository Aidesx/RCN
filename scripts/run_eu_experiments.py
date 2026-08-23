"""E-U0 keyword-scorer + E-U2 topic-coherence experiments -> runs/E-U*."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc import paths  # noqa: E402
from docproc.models.text_classifier import (  # noqa: E402
    load_split_texts,
    read_text_manifest,
)
from docproc.nlp.keywords import _tokens, extract_keywords  # noqa: E402


def _frequency_top(text: str, k: int = 10) -> list[str]:
    """Raw-frequency baseline: NO stopword filtering — that's the point of E-U0."""
    toks = _tokens(text)
    return [t for t, _ in Counter(toks).most_common(k)]


def run_eu0(docs: dict[str, str], k: int = 10) -> dict:
    from docproc.nlp.keywords import _stopwords

    rows = []
    for name, text in docs.items():
        freq = _frequency_top(text, k)
        tfidf = [x["term"].split() for x in extract_keywords(text, k)["keywords"]]
        tfidf_flat = [tok for grp in tfidf for tok in grp]

        def leak(terms_flat):
            hits = sum(1 for t in terms_flat if t in _stopwords())
            return hits / max(1, len(terms_flat))

        overlap = len(set(freq) & set(tfidf_flat)) / max(1, len(set(freq) | set(tfidf_flat)))
        rows.append({"doc": name, "freq_stopword_leak": round(leak(freq), 4),
                     "tfidf_stopword_leak": round(leak(tfidf_flat), 4),
                     "jaccard_overlap": round(overlap, 4)})
    summary = {
        "n_docs": len(rows),
        "avg_freq_stopword_leak": round(sum(r["freq_stopword_leak"] for r in rows) / len(rows), 4),
        "avg_tfidf_stopword_leak": round(sum(r["tfidf_stopword_leak"] for r in rows) / len(rows), 4),
        "avg_jaccard_overlap": round(sum(r["jaccard_overlap"] for r in rows) / len(rows), 4),
    }
    return {"summary": summary, "per_doc": rows}


def run_eu2(docs: dict[str, str], k_min: int = 3, k_max: int = 10) -> dict:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    from docproc.nlp.keywords import _stopwords
    from docproc.nlp.topics import umass_coherence

    names = sorted(docs)
    texts = [docs[n] for n in names]
    vec = CountVectorizer(stop_words=sorted(_stopwords()), min_df=2,
                          token_pattern=r"\b\w{2,}\b")
    dtm = vec.fit_transform(texts)

    curve = []
    best = None
    for kk in range(k_min, k_max + 1):
        lda = LatentDirichletAllocation(n_components=kk, random_state=42,
                                        learning_method="batch", max_iter=25)
        lda.fit_transform(dtm)
        H = lda.components_
        top_ids = [list(H[t].argsort()[::-1][:10]) for t in range(kk)]
        coh = umass_coherence(dtm, top_ids)
        curve.append({"k": kk, "umass": round(float(coh), 6)})
        if best is None or coh > best["umass"]:
            best = {"k": kk, "umass": round(float(coh), 6), "top_ids": top_ids}

    vocab = vec.get_feature_names_out()
    topics = [{"id": t, "top_words": [vocab[i] for i in best["top_ids"][t]]}
              for t in range(best["k"])]
    return {"selected_k": best["k"], "coherence_curve": curve, "topics": topics,
            "n_docs": len(names), "vocab_size": int(dtm.shape[1])}


def main() -> int:
    texts, _ = load_split_texts("train")  # corpus-level experiments on train docs
    names = [r["doc_id"] for r in read_text_manifest("train")]
    docs = dict(zip(names, texts))

    eu0 = run_eu0(docs)
    out0 = paths.RUNS_DIR / "E-U0"
    out0.mkdir(parents=True, exist_ok=True)
    (out0 / "results.json").write_text(json.dumps(eu0, indent=2), encoding="utf-8")

    eu2 = run_eu2(docs)
    out2 = paths.RUNS_DIR / "E-U2"
    out2.mkdir(parents=True, exist_ok=True)
    (out2 / "coherence_curve.json").write_text(json.dumps(eu2, indent=2), encoding="utf-8")

    print("E-U0:", json.dumps(eu0["summary"]))
    print("E-U2: selected k =", eu2["selected_k"])
    print("curve:", [(c["k"], c["umass"]) for c in eu2["coherence_curve"]])
    print("artifacts ->", out0 / "results.json", "&", out2 / "coherence_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())