"""Text structure understanding — level 1 (user objective, 06 v3.0).

Segments raw text into the word -> sentence -> paragraph hierarchy and
reports counts at every level. Deterministic rule-based (course-explainable):
no ML, no network, unicode-aware (works for Vietnamese and English).
"""
from __future__ import annotations

import re

_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r'(?<=[.!?…])["\')\]]?\s+(?=["\'(\[]?[A-ZÀ-Ỹ0-9])')
_WORD = re.compile(r"\w+(?:[-']\w+)*", re.UNICODE)

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "e.g", "i.e", "etc", "vs", "no", "inc", "ltd", "co", "dept", "est",
}


def _split_sentences(paragraph: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace + capital,
    protecting common abbreviations from false splits."""
    parts = _SENT_SPLIT.split(paragraph.strip())
    merged: list[str] = []
    for part in parts:
        if merged:
            prev_tail = merged[-1].rstrip()[:-1].rstrip().split()[-1].lower() \
                if merged[-1].rstrip()[:-1].rstrip() else ""
            if prev_tail in _ABBREVIATIONS:
                merged[-1] = f"{merged[-1]} {part}"
                continue
        merged.append(part)
    # Drop parts with no word characters (e.g. a lone leading ".")
    return [p.strip() for p in merged if p.strip() and _WORD.search(p)]


def analyze_structure(text: str) -> dict:
    """Return the word -> sentence -> paragraph hierarchy + counts."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]

    out_paragraphs = []
    total_words = total_sentences = 0
    all_words: list[str] = []
    for idx, para in enumerate(paragraphs):
        sentences = _split_sentences(para)
        para_words: list[str] = []
        for sent in sentences:
            words = _WORD.findall(sent)
            para_words.extend(words)
            all_words.extend(words)
        total_words += len(para_words)
        total_sentences += len(sentences)
        out_paragraphs.append({
            "index": idx,
            "sentence_count": len(sentences),
            "word_count": len(para_words),
            "sentences": sentences,
        })

    return {
        "stats": {
            "characters": len(text),
            "words": total_words,
            "unique_words": len({w.lower() for w in all_words}),
            "sentences": total_sentences,
            "paragraphs": len(out_paragraphs),
        },
        "paragraphs": out_paragraphs,
    }