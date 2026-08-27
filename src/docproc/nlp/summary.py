"""L5 summary: extractive summarization built on L1 structure + L2 keyphrases.

Spec 07-summary v1: keep the word -> sentence -> paragraph analysis, then
compress by scoring sentences against in-document TF-IDF keyphrases and
selecting a non-redundant subset with MMR (Maximal Marginal Relevance).
Abstractive generation plugs into summarize() behind configs/summary.yaml
(engine swap; graceful fallback to extractive when no checkpoint is present).
Deterministic: same input => same summary (pure math, seeded components only).
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from docproc import paths
from docproc.nlp.keywords import _content_tokens, _tokens, extract_keywords
from docproc.nlp.structure import analyze_structure

_ABSTRACTIVE_DIRNAME = "summarizer_mt5"


def _checkpoint_ref(cfg: dict) -> Path | None:
    """Local checkpoint dir (finetuned override first) or None if absent."""
    artifacts = paths.ROOT / "models" / "artifacts"
    ft = cfg.get("finetuned_checkpoint")
    if ft:
        p = artifacts / str(ft)
        if not ((p / "config.json").exists()):
            raise NotImplementedError(
                f"finetuned_checkpoint '{ft}' configured but not found under "
                f"{artifacts}")
        return p
    local = artifacts / _ABSTRACTIVE_DIRNAME
    if (local / "config.json").exists():
        return local
    return None


def _keyword_weights(text: str, pool: int) -> dict[str, float]:
    return {x["term"]: float(x["score"])
            for x in extract_keywords(text, k=pool)["keywords"]}


def _sentence_records(structure: dict) -> list[dict]:
    records: list[dict] = []
    for para in structure["paragraphs"]:
        for pos, sent in enumerate(para["sentences"]):
            tokens = _content_tokens(_tokens(sent))
            records.append({
                "text": sent,
                "paragraph": para["index"],
                "position": pos,
                "tokens": tokens,
                "token_set": set(tokens),
            })
    return records


def _relevance(record: dict, weights: dict[str, float],
               bonus: float) -> float:
    """Keyword-weight overlap normalized by length; paragraph-initial bonus."""
    tokens = record["tokens"]
    base = sum(weights.get(t, 0.0) for t in tokens)
    for a, b in pairwise(tokens):
        bg = f"{a} {b}"
        if bg in weights:
            base += weights[bg]
    rel = base / math.sqrt(len(record["token_set"]) + 1)
    if record["position"] == 0:
        rel += bonus
    return rel


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b) if inter else 0.0


def _mmr_select(records: list[dict], relevances: list[float],
                lam: float, k: int) -> list[int]:
    """Greedy MMR: lam*relevance - (1-lam)*max-similarity-to-chosen."""
    remaining = sorted(range(len(records)),
                       key=lambda i: (-relevances[i], i))
    chosen = [remaining.pop(0)]
    while len(chosen) < k and remaining:
        best_i, best_v = None, None
        for i in remaining:
            red = max((_jaccard(records[i]["token_set"],
                                records[c]["token_set"]) for c in chosen),
                      default=0.0)
            v = lam * relevances[i] - (1.0 - lam) * red
            if best_v is None or v > best_v + 1e-12:
                best_i, best_v = i, v
        chosen.append(best_i)
        remaining.remove(best_i)
    return sorted(chosen)


def summarize_extractive(text: str, k: int | None = None) -> dict:
    """Select top-k sentences scored by L2 keyphrase overlap over L1 sentences.

    Returns {'engine', 'sentences': [{text, paragraph, score}], 'compression'}.
    Sentences come back in original document order.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cfg = paths.load_config("summary")
    xcfg = cfg.get("extractive", {})
    if k is None:
        k = int(cfg.get("k_sentences", 3))
    if k <= 0:
        raise ValueError("k must be positive")

    structure = analyze_structure(text)
    records = _sentence_records(structure)
    n = len(records)
    empty = {"engine": "extractive", "sentences": [],
             "compression": {"original_sentences": 0, "kept": 0}}
    if n == 0:
        return empty

    pool = int(xcfg.get("keyword_pool", 30))
    lam = float(xcfg.get("mmr_lambda", 0.7))
    bonus = float(xcfg.get("position_bonus", 0.1))
    weights = _keyword_weights(text, pool)
    relevances = [_relevance(r, weights, bonus) for r in records]

    picked = _mmr_select(records, relevances, lam, min(k, n))
    sentences = [{"text": records[i]["text"],
                  "paragraph": records[i]["paragraph"],
                  "score": round(relevances[i], 6)} for i in picked]
    return {"engine": "extractive", "sentences": sentences,
            "compression": {"original_sentences": n, "kept": len(sentences)}}


@lru_cache(maxsize=4)
def _load_seq2seq(ref_str: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5TokenizerFast

    try:
        tokenizer = AutoTokenizer.from_pretrained(ref_str)
    except Exception:
        tj = Path(ref_str) / "tokenizer.json"
        if not tj.exists():
            raise
        tokenizer = T5TokenizerFast(tokenizer_file=str(tj),
                                    model_max_length=512)
    model = AutoModelForSeq2SeqLM.from_pretrained(ref_str)
    model.eval()
    return tokenizer, model


def summarize_abstractive(text: str, k: int | None = None,
                          checkpoint: str | None = None) -> dict:
    """Generate a rewritten passage with a small local seq2seq checkpoint.

    Raises NotImplementedError whenever the engine cannot run locally
    (deps missing or checkpoint absent) so callers can fall back to
    extractive per spec D2. ``k`` is accepted for API symmetry and ignored.
    ``checkpoint`` optionally names a models/artifacts/ dir directly,
    overriding configs/summary.yaml selection (override hook).
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cfg = paths.load_config("summary").get("abstractive", {})
    if checkpoint:
        ref = paths.ROOT / "models" / "artifacts" / checkpoint
        if not ((ref / "config.json").exists()):
            raise NotImplementedError(
                f"checkpoint '{checkpoint}' not found under models/artifacts/")
    else:
        ref = _checkpoint_ref(cfg)
        if ref is None:
            raise NotImplementedError(
                f"no local summarizer checkpoint under models/artifacts/"
                f"{_ABSTRACTIVE_DIRNAME}; run scripts/download_summarizer.py")
    try:
        import torch

        tokenizer, model = _load_seq2seq(str(ref))
    except ImportError as exc:  # engine deps are optional by design
        raise NotImplementedError(f"summarizer deps missing: {exc}") from exc
    except Exception as exc:
        raise NotImplementedError(f"summarizer checkpoint unusable: {exc}") from exc

    n_in = analyze_structure(text)["stats"]["sentences"]
    clean = re.sub(r"\s+", " ", text.strip())
    if not clean:
        return {"engine": "abstractive", "text": "", "model": ref.name,
                "compression": {"original_sentences": 0, "kept": 0}}

    inputs = tokenizer([clean], return_tensors="pt", truncation=True,
                       max_length=int(cfg.get("max_input_tokens", 512)))
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=int(cfg.get("max_new_tokens", 128)),
            num_beams=int(cfg.get("num_beams", 4)),
            no_repeat_ngram_size=int(cfg.get("no_repeat_ngram_size", 3)),
            min_new_tokens=int(cfg.get("min_new_tokens", 8)),
        )
    generated = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    return {"engine": "abstractive", "text": generated, "model": ref.name,
            "compression": {"original_sentences": n_in, "kept": None}}


def summarize(text: str, mode: str | None = None, k: int | None = None) -> dict:
    """Dispatch by engine mode; defaults from configs/summary.yaml."""
    chosen = mode or paths.load_config("summary").get(
        "default_mode", "extractive")
    if chosen == "extractive":
        return summarize_extractive(text, k=k)
    if chosen == "abstractive":
        return summarize_abstractive(text, k=k)
    raise ValueError(f"unknown summary mode: {chosen!r}")
