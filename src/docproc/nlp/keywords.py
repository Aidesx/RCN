"""L2 keywords: top-k uni/bi-gram keyphrases scored by in-document TF-IDF."""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

from docproc.nlp.structure import analyze_structure

_WORD = re.compile(r"\w+(?:[-']\w+)*", re.UNICODE)
_STOPWORDS_PATH = Path(__file__).resolve().parent / "stopwords.txt"


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    words = _STOPWORDS_PATH.read_text(encoding="utf-8").split()
    return frozenset(w.lower() for w in words)


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _content_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _stopwords() and not t.isdigit() and len(t) > 1]


def extract_keywords(text: str, k: int = 10) -> dict:
    """Return {'keywords': [{'term','score','count'}...], 'candidates_scored': int}."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if k <= 0:
        raise ValueError("k must be positive")

    structure = analyze_structure(text)
    para_tokens = [
        _content_tokens(_tokens(" ".join(p["sentences"])))
        for p in structure["paragraphs"]
    ]
    n_paras = len(para_tokens)
    doc_tokens = [t for para in para_tokens for t in para]
    if not doc_tokens or n_paras == 0:
        return {"keywords": [], "candidates_scored": 0}

    def _df(term: str) -> int:
        return sum(1 for para in para_tokens if term in para)

    df = {t: _df(t) for t in set(doc_tokens)}

    def idf(term: str) -> float:
        return math.log((n_paras + 1) / (df.get(term, 0) + 1)) + 1.0

    tf_uni: dict[str, int] = {}
    for t in doc_tokens:
        tf_uni[t] = tf_uni.get(t, 0) + 1
    score_uni = {t: c * idf(t) for t, c in tf_uni.items()}

    candidates: dict[str, tuple[float, int]] = {
        t: (s, tf_uni[t]) for t, s in score_uni.items()
    }

    from collections import Counter

    pair_counts = Counter(zip(doc_tokens, doc_tokens[1:]))
    for (a, b), cnt in pair_counts.items():
        bg = f"{a} {b}"
        if bg not in candidates:
            candidates[bg] = ((score_uni[a] + score_uni[b]) / 2.0, cnt)

    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1][0], kv[0]))

    chosen: list[dict] = []
    used_unigrams: set[str] = set()
    for term, (score, cnt) in ranked:
        if len(chosen) >= k:
            break
        parts = term.split()
        if len(parts) == 2 and parts[0] in used_unigrams and parts[1] in used_unigrams:
            continue
        chosen.append({"term": term, "score": round(score, 6), "count": cnt})
        if len(parts) == 1:
            used_unigrams.add(term)

    return {"keywords": chosen, "candidates_scored": len(candidates)}