"""Understanding report seam: understand() / understand_file() / render_markdown()."""
from __future__ import annotations

from pathlib import Path

from docproc import paths
from docproc.io.detect import (
    DOCX,
    HTML,
    IMAGE,
    MARKDOWN,
    PDF_SCANNED,
    PDF_TEXT,
    UNKNOWN,
    ParseError,
    detect_file_type,
)
from docproc.io.parsers import extract_text
from docproc.nlp.fields import extract_fields
from docproc.nlp.keywords import extract_keywords
from docproc.nlp.structure import analyze_structure
from docproc.nlp.summary import summarize, summarize_extractive
from docproc.nlp.topics import extract_topics

# High-precision text cues per class — last-resort router when the SVM
# model is below the confidence threshold or artifacts are missing.
_ROUTER_CUES: dict[str, tuple[str, ...]] = {
    "invoice": ("total due", "invoice #", "inv-", "payment terms", "bill to"),
    "receipt": ("receipt", "payment method", "cash", "visa", "mastercard"),
    "letter": ("dear ", "sincerely"),
    "report": ("quarterly", "annual report", "project report", "abstract"),
    "article": ("published",),
    "form": ("office use only", "submission date", "reference no",
             "requested item"),
}


def _rule_cues(text: str) -> dict[str, int]:
    """Deterministic keyword-cue score per class (high-precision fallback)."""
    low = text.lower()
    return {cls: sum(1 for cue in cues if cue in low)
            for cls, cues in _ROUTER_CUES.items()}


def _router_confidence(model, X) -> float:
    """Deterministic confidence proxy: softmax over SVM one-vs-rest margins
    (LinearSVC has no predict_proba); predict_proba when available."""
    import numpy as np

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0].max())
    d = np.asarray(model.decision_function(X)).ravel()
    e = np.exp(d - d.max())
    return float((e / e.sum()).max())


def _router_threshold() -> float:
    try:
        return float(paths.pipeline_config().get("classification", {}).get(
            "confidence_threshold", 0.5))
    except Exception:
        return 0.5


def _router_text(text: str) -> dict:
    """Advisory label via the E0b SVM artifact, gated by confidence; rule
    cues as a deterministic fallback so low-confidence docs never guess (D3).
    """
    vec_path = paths.ROOT / "models" / "artifacts" / "text_vectorizer.joblib"
    model_path = paths.ROOT / "models" / "artifacts" / "text_model_svm.joblib"
    if not (vec_path.is_file() and model_path.is_file()):
        return {"label": "unavailable", "via": None}
    try:
        from docproc.preprocess.text import TextVectorizer

        vec = TextVectorizer.load(vec_path)
        import joblib

        model = joblib.load(model_path)
        X = vec.transform([text])
        pred = model.predict(X)[0]
        label = paths.class_names()[int(pred)]
        conf = _router_confidence(model, X)
        if conf >= _router_threshold():
            return {"label": label, "via": "e0b_svm",
                    "confidence": round(conf, 4)}
        cues = _rule_cues(text)
        best_cls, best_n = max(cues.items(), key=lambda kv: kv[1],
                               default=(None, 0))
        if best_n >= 1:
            return {"label": best_cls, "via": "rule_cues", "confidence": None}
        return {"label": None, "via": "low_confidence",
                "confidence": round(conf, 4)}
    except Exception as exc:
        return {"label": "unavailable", "via": f"error: {exc}"}


def _router_image(image) -> dict:
    """Advisory label via the E1 CNN on a PIL page image; graceful when missing."""
    ckpt = paths.RUNS_DIR / "E1" / "best.keras"
    if not ckpt.is_file():
        return {"label": "unavailable", "via": None}
    try:
        import numpy as np
        import tensorflow as tf

        from docproc.preprocess.image import _model_size, image_to_tensor

        model = tf.keras.models.load_model(ckpt)
        x = image_to_tensor(image, _model_size("cnn"))[None, ...]
        prob = model.predict(x, verbose=0)
        idx = int(np.argmax(prob[0]))
        return {"label": paths.class_names()[idx],
                "confidence": round(float(prob[0][idx]), 4), "via": "e1_cnn"}
    except Exception as exc:
        return {"label": "unavailable", "via": f"error: {exc}"}


def understand(text: str, source: str = "inline", k_keywords: int = 10,
               k_topics: int | None = None, summary_mode: str | None = None,
               summary_k: int | None = None) -> dict:
    """Full understanding record for a raw text string."""
    structure = analyze_structure(text)
    keywords = extract_keywords(text, k=k_keywords)
    topics = extract_topics(text, k=k_topics)
    router = _router_text(text) if text.strip() else {"label": "unavailable",
                                                      "via": "empty text"}
    fields = (extract_fields(text, router.get("label"))
              if router.get("label") not in (None, "unavailable")
              else extract_fields(text, None))  # generic schema, same contract
    record = {
        "source": source,
        "doc_type": router,
        "structure": structure,
        "keywords": keywords["keywords"],
        "topics": topics,
        "fields": fields,
    }
    try:
        record["summary"] = summarize(text, mode=summary_mode, k=summary_k)
    except NotImplementedError:
        # D2 graceful degradation: requested engine unavailable -> extractive.
        record["summary"] = summarize_extractive(text, k=summary_k)
        record["summary"]["engine_fallback"] = True
    return record


def understand_file(path, k_keywords: int = 10,
                    k_topics: int | None = None,
                    summary_mode: str | None = None,
                    summary_k: int | None = None) -> dict:
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
        from docproc.io.detect import UnsupportedFormatError

        raise UnsupportedFormatError(UNKNOWN, p, "no supported format detected")
    else:
        from docproc.io.detect import UnsupportedFormatError

        raise UnsupportedFormatError(
            detection.file_type, p,
            f"type '{detection.file_type}' has no understanding path")

    record = understand(text, source=str(p), k_keywords=k_keywords,
                        k_topics=k_topics, summary_mode=summary_mode,
                        summary_k=summary_k)
    record["file_type"] = detection.file_type
    return record


def render_markdown(record: dict) -> str:
    """Deterministic Markdown rendering of an understanding record."""
    lines = [f"# Understanding — {record.get('source', 'inline')}", ""]
    dt = record.get("doc_type", {})
    label = dt.get("label") or "unavailable"
    lines += [f"- **doc_type:** {label}"
              + (f" (via {dt['via']})" if dt.get("via") else ""), ""]
    sm = record.get("summary")
    if sm:
        engine = sm.get("engine", "extractive")
        if engine == "abstractive":
            head = f"## Summary (abstractive · {sm.get('model', '')})"
            lines += [head, "", sm.get("text", ""), ""]
        else:
            kept = len(sm.get("sentences", []))
            total = (sm.get("compression") or {}).get("original_sentences")
            head = f"## Summary ({engine}"
            if total:
                head += f" · kept {kept}/{total}"
            lines += [head + ")", ""]
            for i, s in enumerate(sm.get("sentences", []), 1):
                lines.append(f"{i}. {s['text']}")
            if sm.get("sentences"):
                lines.append("")
        if sm.get("engine_fallback"):
            lines += [("> engine fallback: requested engine unavailable, "
                       "extractive used"), ""]
    if "structure" in record:
        s = record["structure"]["stats"]
        lines += ["## Structure", "",
                  ("- paragraphs: "
                   f"{s['paragraphs']} · sentences: {s['sentences']} · "
                   f"words: {s['words']} · unique: {s['unique_words']} · chars: {s['characters']}"),
                  ""]
        for para in record["structure"]["paragraphs"]:
            lines.append(f"### ¶{para['index']} ({para['sentence_count']} sentences, "
                         f"{para['word_count']} words)")
            for i, sent in enumerate(para["sentences"], 1):
                lines.append(f"{i}. {sent}")
            lines.append("")
    if record.get("keywords"):
        lines += [f"## Keywords (top {len(record['keywords'])})", "",
                  "| term | score | count |", "|---|---|---|"]
        lines += [f"| {kw['term']} | {kw['score']} | {kw['count']} |"
                  for kw in record["keywords"]]
        lines.append("")
    if "topics" in record:
        t = record["topics"]
        lines += [f"## Topics (k={t.get('k')}, coherence={t.get('coherence')})", ""]
        for topic in t.get("topics", []):
            words = ", ".join(topic.get("top_words", []))
            label = topic.get("label")
            if label:
                lines.append(f"- **T{topic['id']}** (label: {label}): {words}")
            else:
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
        missing = f.get("missing_required")
        if missing:
            lines.append("")
            lines.append(f"> ⚠️ thiếu required field: {', '.join(missing)}")
        lines.append("")
    if record.get("note"):
        lines += [f"> {record['note']}", ""]
    return "\n".join(lines)