"""Understanding report seam: understand() / understand_file() / render_markdown()."""
from __future__ import annotations

from pathlib import Path

from docproc import paths
from docproc.io.detect import (
    DOCX, HTML, IMAGE, MARKDOWN, PDF_SCANNED, PDF_TEXT, UNKNOWN,
    ParseError,
    detect_file_type,
)
from docproc.io.parsers import extract_text
from docproc.nlp.fields import extract_fields
from docproc.nlp.keywords import extract_keywords
from docproc.nlp.structure import analyze_structure
from docproc.nlp.topics import extract_topics


def _router_text(text: str) -> dict:
    """Advisory label via the E0b SVM artifact; graceful when missing."""
    vec_path = paths.ROOT / "models" / "artifacts" / "text_vectorizer.joblib"
    model_path = paths.ROOT / "models" / "artifacts" / "text_model_svm.joblib"
    if not (vec_path.is_file() and model_path.is_file()):
        return {"label": "unavailable", "via": None}
    try:
        from docproc.preprocess.text import TextVectorizer

        vec = TextVectorizer.load(vec_path)
        import joblib

        model = joblib.load(model_path)
        pred = model.predict(vec.transform([text]))[0]
        return {"label": paths.class_names()[int(pred)], "via": "e0b_svm"}
    except Exception as exc:  # artifact corruption must not kill the report
        return {"label": "unavailable", "via": f"error: {exc}"}


def _router_image(image) -> dict:
    """Advisory label via the E1 CNN on a PIL page image; graceful when missing."""
    ckpt = paths.RUNS_DIR / "E1" / "best.keras"
    if not ckpt.is_file():
        return {"label": "unavailable", "via": None}
    try:
        import numpy as np
        import tensorflow as tf

        from docproc.preprocess.image import image_to_tensor, _model_size

        model = tf.keras.models.load_model(ckpt)
        x = image_to_tensor(image, _model_size("cnn"))[None, ...]
        prob = model.predict(x, verbose=0)
        idx = int(np.argmax(prob[0]))
        return {"label": paths.class_names()[idx],
                "confidence": round(float(prob[0][idx]), 4), "via": "e1_cnn"}
    except Exception as exc:
        return {"label": "unavailable", "via": f"error: {exc}"}


def understand(text: str, source: str = "inline", k_keywords: int = 10,
               k_topics: int | None = None) -> dict:
    """Full understanding record for a raw text string."""
    structure = analyze_structure(text)
    keywords = extract_keywords(text, k=k_keywords)
    topics = extract_topics(text, k=k_topics)
    router = _router_text(text) if text.strip() else {"label": "unavailable",
                                                      "via": "empty text"}
    fields = (extract_fields(text, router.get("label"))
              if router.get("label") not in (None, "unavailable")
              else {"doc_type": "generic", "fields": {}, "matched": 0})
    return {
        "source": source,
        "doc_type": router,
        "structure": structure,
        "keywords": keywords["keywords"],
        "topics": topics,
        "fields": fields,
    }


def understand_file(path, k_keywords: int = 10,
                    k_topics: int | None = None) -> dict:
    """Dispatch by file type per 02 §3; classification-only for images/scans."""
    p = Path(path)
    detection = detect_file_type(p)

    if detection.file_type in (PDF_SCANNED, IMAGE):
        if detection.file_type == IMAGE:
            from docproc.preprocess.image import file_to_pil

            page = file_to_pil(p)
        else:
            from docproc.io.render import render_pdf_pages

            pages = render_pdf_pages(p)
            if not pages:
                raise ParseError("pdf", p, "no pages rendered for routing")
            page = pages[0]
        router = _router_image(page)
        return {"source": str(p), "doc_type": router,
                "note": "classification-only input (no text layer; OCR out of scope)"}

    if detection.file_type in (PDF_TEXT, DOCX, MARKDOWN, HTML):
        text = extract_text(p)
    elif detection.file_type == UNKNOWN:
        raise ValueError(f"unsupported file type for understanding: {p}")
    else:
        raise ValueError(f"unroutable type '{detection.file_type}'")

    record = understand(text, source=str(p), k_keywords=k_keywords,
                        k_topics=k_topics)
    record["file_type"] = detection.file_type
    return record


def render_markdown(record: dict) -> str:
    """Deterministic Markdown rendering of an understanding record."""
    lines = [f"# Understanding — {record.get('source', 'inline')}", ""]
    dt = record.get("doc_type", {})
    lines += [f"- **doc_type:** {dt.get('label', 'unavailable')}"
              + (f" (via {dt['via']})" if dt.get("via") else ""), ""]
    if "structure" in record:
        s = record["structure"]["stats"]
        lines += ["## Structure", "",
                  f"- paragraphs: {s['paragraphs']} · sentences: {s['sentences']} · "
                  f"words: {s['words']} · unique: {s['unique_words']} · chars: {s['characters']}",
                  ""]
        for para in record["structure"]["paragraphs"]:
            lines.append(f"### ¶{para['index']} ({para['sentence_count']} sentences, "
                         f"{para['word_count']} words)")
            for i, sent in enumerate(para["sentences"], 1):
                lines.append(f"{i}. {sent}")
            lines.append("")
    if record.get("keywords"):
        lines += ["## Keywords (top %d)" % len(record["keywords"]), "",
                  "| term | score | count |", "|---|---|---|"]
        lines += [f"| {kw['term']} | {kw['score']} | {kw['count']} |"
                  for kw in record["keywords"]]
        lines.append("")
    if "topics" in record:
        t = record["topics"]
        lines += [f"## Topics (k={t.get('k')}, coherence={t.get('coherence')})", ""]
        for topic in t.get("topics", []):
            words = ", ".join(topic["top_words"])
            lines.append(f"- **T{topic['id']}:** {words}")
        mix = t.get("doc_topic_mixture") or []
        if mix:
            lines.append("")
            lines.append("mixture: " + ", ".join(f"T{i}={v:.3f}" for i, v in enumerate(mix)))
        lines.append("")
    if record.get("fields"):
        f = record["fields"]
        lines += [f"## Fields ({f.get('doc_type')})", ""]
        rows = [(k, v) for k, v in (f.get("fields") or {}).items()]
        if rows:
            lines += ["| field | value |", "|---|---|"]
            lines += [f"| {k} | {v if v is not None else '—'} |" for k, v in rows]
        else:
            lines.append("_no fields matched_")
        lines.append("")
    if record.get("note"):
        lines += [f"> {record['note']}", ""]
    return "\n".join(lines)