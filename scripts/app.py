
import csv
import html
import json
import re
import sys
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import streamlit as st
except ImportError:
    print("Missing streamlit. Install with: .venv/Scripts/pip install streamlit")
    sys.exit(1)

st.set_page_config(page_title="RCN Studio", page_icon="📄", layout="wide")

# ------------------------------------------------------------------ constants
# Palette "Blurple": deep-indigo canvas + Blurple/green/magenta.
# Light = background #f5f7ff, Dark = canvas #0a0d3a. Display font Space Grotesk.
BRAND = "#5865f2"          # Blurple —
BRAND_ON = "#ffffff"
GREEN_CTA = "#35ed7e"      # electric green — high-intent actions
MAGENTA = "#ec48bd"        # vibrant magenta — gradient feature panels
LINK_CYAN = "#00b0f4"      # inline link color on dark surfaces
CANVAS = "#0a0d3a"         # deep-indigo page canvas
SURFACE_INDIGO = "#1e2353" # raised indigo panel
SURFACE_ONYX = "#23272a"   # dark UI surface

CLASS_META = {
    "invoice": ("Invoice", "🧾", BRAND),           # Blurple
    "receipt": ("Receipt", "✅", GREEN_CTA),      # electric green
    "report": ("Report", "📊", MAGENTA),          # magenta
    "letter": ("Letter", "✉️", LINK_CYAN),         # link cyan
    "form": ("Form", "📋", "#a06cd5"),         # violet accent
    "article": ("Article", "📰", "#7c86c8"),      # muted indigo
}
ENGINE_BADGE = {
    "extractive": ("📋 Extracts the most important sentences", BRAND),
    "abstractive": ("✍️ Generates new text with a small local model", MAGENTA),
}
SUPPORTED = ["md", "txt", "html", "htm", "docx"]
DEMO_RECORD = {
    "source": "(sample data) hoa_don_mau.txt",
    "file_type": "txt",
    "doc_type": {"label": "invoice", "via": "e0b_svm", "confidence": 0.94},
    "structure": {"stats": {"characters": 1124, "words": 186,
                            "unique_words": 97, "sentences": 21,
                            "paragraphs": 6},
                  "paragraphs": [
                      {"index": 0, "sentence_count": 2, "word_count": 38,
                       "sentences": [
                           ("Công ty TNHH An Phát xin gửi hóa đơn giá trị gia "
                            "tăng số HD-2026-0841 ngày 20/08/2026."),
                           ("Hóa đơn áp dụng cho lô hàng giấy photo A4 theo "
                            "hợp đồng cung ứng văn phòng phẩm.")]},
                      {"index": 1, "sentence_count": 2, "word_count": 41,
                       "sentences": [
                           ("Tổng giá trị thanh toán sau thuế VAT 8% là "
                            "45.600.000 đồng."),
                           ("Hạn công nợ 30 ngày kể từ ngày xuất hóa đơn, vui "
                            "lòng thanh toán đúng hạn.")]},
                      {"index": 2, "sentence_count": 1, "word_count": 18,
                       "sentences": [
                           ("Mọi thắc mắc về khoản mục vui lòng liên hệ phòng "
                            "kế toán trong vòng 7 ngày làm việc.")]}]},
    "keywords": [{"term": "hóa đơn giá trị gia tăng", "score": 2.41, "count": 2},
                 {"term": "công ty TNHH An Phát", "score": 2.12, "count": 2},
                 {"term": "công nợ", "score": 1.87, "count": 1},
                 {"term": "giấy photo", "score": 1.74, "count": 1},
                 {"term": "thanh toán", "score": 1.66, "count": 2},
                 {"term": "VAT", "score": 1.58, "count": 2},
                 {"term": "45.600.000", "score": 1.51, "count": 1},
                 {"term": "hợp đồng", "score": 1.43, "count": 1},
                 {"term": "phòng kế toán", "score": 1.35, "count": 1},
                 {"term": "khoản mục", "score": 1.21, "count": 1}],
    "topics": {
        "k": 2,
        "selected_by": "argmax_umass",
        "coherence": -0.27,
        "coherence_curve": [{"k": 2, "umass": -0.27},
                            {"k": 3, "umass": -0.35},
                            {"k": 4, "umass": -0.42}],
        "topics": [
            {"id": 0, "top_words": ["hóa đơn", "công ty", "thanh toán",
                                    "công nợ", "đồng"], "label": "thanh toán"},
            {"id": 1, "top_words": ["giấy photo", "mặt hàng", "đơn giá",
                                    "số lượng", "VAT"], "label": "mặt hàng"},
        ],
        "doc_topic_mixture": [0.62, 0.38],
    },
    "fields": {
        "doc_type": "invoice",
        "fields": {
            "invoice_number": "HD-2026-0841",
            "date": "2026-08-20",
            "total_due": "45600000",
            "vendor": "Công ty TNHH An Phát",
            "buyer": "Công ty CP Minh Khoa",
        },
        "matched": 5,
        "missing_required": [],
    },
    "summary": {
        "engine": "extractive",
        "sentences": [
            {"text": ("Công ty TNHH An Phát xin gửi hóa đơn giá trị gia tăng "
                      "số HD-2026-0841 ngày 20/08/2026 cho lô hàng giấy photo "
                      "A4 theo hợp đồng cung ứng văn phòng phẩm."), "paragraph": 1,
             "score": 0.83},
            {"text": ("Tổng giá trị thanh toán sau thuế VAT 8% là "
                      "45.600.000 đồng, hạn công nợ 30 ngày kể từ ngày xuất "
                      "hóa đơn."), "paragraph": 4, "score": 0.91},
            {"text": ("Mọi thắc mắc về khoản mục vui lòng liên hệ phòng kế "
                      "toán trong vòng 7 ngày làm việc."), "paragraph": 6,
             "score": 0.41},
        ],
        "compression": {"original_sentences": 21, "kept": 3},
    },
}

DEMO_ABSTRACTIVE_TEXT = (
    "Hóa đơn số HD-2026-0841 do Công ty TNHH An Phát phát hành ngày 20/08/2026 "
    "cho Công ty CP Minh Khoa, ghi nhận lô hàng giấy photo A4 với tổng giá trị "
    "thanh toán 45.600.000 đồng sau thuế VAT 8%, hạn công nợ 30 ngày."
)


def available_checkpoints():
    """models/artifacts/ dirs that can actually run the abstractive engine
    (config + weights + tokenizer present) — UI model picker lists only these."""
    art = ROOT / "models" / "artifacts"
    if not art.is_dir():
        return []
    found = []
    for d in sorted(art.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name.endswith("_copy"):
            continue
        has_cfg = (d / "config.json").is_file()
        has_w = ((d / "model.safetensors").is_file()
                 or (d / "pytorch_model.bin").is_file())
        has_tok = ((d / "tokenizer.json").is_file()
                   or (d / "spiece.model").is_file()
                   or (d / "tokenizer_config.json").is_file())
        if has_cfg and has_w and has_tok:
            found.append(d.name)

    def rank(n):
        return 0 if n == "vit5_v1" else (1 if n == "vit5_soup_0.5_v1" else 2)

    return sorted(found, key=lambda n: (rank(n), n))


def demo_record(mode, k_sum, model=None):
    """Sample record following the sidebar choices (mode + sentence count),
    so every control does something while previewing the UI."""
    rec = json.loads(json.dumps(DEMO_RECORD))  # simple deep copy
    sm = rec["summary"]
    if mode == "abstractive":
        rec["summary"] = {"engine": "abstractive", "text": DEMO_ABSTRACTIVE_TEXT,
                          "model": model or "summarizer_mt5",
                          "compression": {"original_sentences": 21,
                                          "kept": None}}
    else:
        kept = sm["sentences"][:max(1, min(k_sum, len(sm["sentences"])))]
        sm["sentences"] = kept
        sm["compression"]["kept"] = len(kept)
    return rec


def inject_css(dark: bool):
    """Palette Blurple: deep-indigo canvas + Blurple/magenta/green accents
    (Discord-style design tokens) — display type Space Grotesk, body Inter."""
    if dark:
        bg_base = "#0a0d3a"                 # canvas (deep indigo)
        surface = "#1e2353"                 # surface-indigo (raised panel)
        onyx = "#23272a"                    # surface-onyx
        ink = "#ffffff"
        muted = "#8f96c9"
        hairline = "#2a2f63"
        sidebar_bg = "#0a0d3a"
        sidebar_txt = "#c3c8ea"
        body_extra = f"""
      section[data-testid="stSidebar"] {{
          background:{sidebar_bg};
          border-right:1px solid {hairline};
      }}
      section[data-testid="stSidebar"] *:not(button):not([data-variant]) {{
          color:{sidebar_txt} !important;
      }}
      .stTabs [data-baseweb="tab"] {{
          background:{surface}; border-radius:12px 12px 0 0;
          color:{muted} !important; font-weight:600; }}
      .stTabs [aria-selected="true"] {{ color:#ffffff !important; }}"""
    else:
        bg_base = "#f5f7ff"
        surface = "#ffffff"
        onyx = "#eceefc"
        ink = "#10143a"
        muted = "#5a6189"
        hairline = "#d7dbef"
        sidebar_bg = "#0a0d3a"
        sidebar_txt = "#c3c8ea"
        body_extra = f"""
      section[data-testid="stSidebar"] {{
          background:{sidebar_bg};
          border-right:1px solid {hairline};
      }}
      section[data-testid="stSidebar"] *:not(button):not([data-variant]) {{
          color:{sidebar_txt} !important;
      }}
      .stTabs [data-baseweb="tab"] {{
          background:#ececfc; border-radius:12px 12px 0 0;
          color:#5a6189 !important; font-weight:600; }}
      .stTabs [aria-selected="true"] {{ color:#5865f2 !important; }}"""
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
      :root {{
        --brand:#5865f2; --brand-on:#ffffff;
        --green:#35ed7e; --magenta:#ec48bd; --link:#00b0f4;
        --canvas:{bg_base}; --surface:{surface}; --onyx:{onyx};
        --ink:{ink}; --muted:{muted}; --hairline:{hairline};
      }}
      html, body, .stApp, div[data-testid="stAppViewContainer"] {{
          background: {bg_base};
          color: {ink};
          font-family: 'Inter', 'Segoe UI', sans-serif;
      }}
      /* Animated brand gradient mesh on the canvas (Blurple → magenta) */
      div[data-testid="stAppViewContainer"] {{
          background:
              radial-gradient(600px 320px at 12% -5%, rgba(88,101,242,.28), transparent 70%),
              radial-gradient(560px 300px at 88% 8%, rgba(236,72,189,.20), transparent 70%),
              radial-gradient(700px 380px at 50% 110%, rgba(0,176,244,.14), transparent 70%),
              {bg_base};
      }}
      {body_extra}
      section[data-testid="stSidebar"] > div {{ padding-top:.9rem; }}
      div[data-testid="stAppViewContainer"] .block-container {{ padding-top:.9rem; }}
      section[data-testid="stSidebar"] hr {{ border-color:{hairline}; }}
      h1, h2, h3, h4 {{
          font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
          letter-spacing:-.02em;
      }}
      /* ---- stat cards: Blurple fill, display number ---- */
      div[data-testid="stMetric"] {{
          background:{surface}; border:1px solid {hairline};
          border-radius:16px; padding:14px 16px;
          box-shadow:0 3px 18px rgba(69,42,124,.10);
          transition:transform .15s ease, box-shadow .15s ease; }}
      div[data-testid="stMetric"]:hover {{
          transform:translateY(-2px);
          box-shadow:0 6px 26px rgba(88,101,242,.25); }}
      div[data-testid="stMetric"] label {{
          color:{muted} !important; font-weight:600; font-size:.8rem; }}
      div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
          font-family:'Space Grotesk',sans-serif; font-weight:800;
          color:{ink}; }}
      /* ---- hero: brand gradient band ---- */
      .rcn-hero {{
          background:linear-gradient(120deg,#1e2353,#5865f2 55%,#ec48bd);
          color:#fff; padding:30px 34px; border-radius:40px;
          box-shadow:0 3px 68px rgba(69,42,124,.25); }}
      .rcn-hero h1 {{
          margin:0 0 6px 0; font-size:30px; font-weight:800;
          color:#fff !important; letter-spacing:-.02em; }}
      .rcn-hero p {{ margin:0; opacity:.92; font-size:15px; }}
      /* ---- chips: pill badges ---- */
      .rcn-chip {{
          display:inline-block; background:{surface};
          border:1px solid {hairline}; color:{ink};
          padding:6px 16px; margin:0 8px 10px 0; border-radius:50px;
          transition:transform .15s ease; font-weight:500; }}
      .rcn-chip:hover {{ transform:translateY(-1px);
          border-color:rgba(88,101,242,.6); }}
      /* ---- buttons: Blurple primary / green high-intent / ghost ---- */
      div[data-testid="stButton"] button,
      div[data-testid="stDownloadButton"] button,
      div[data-testid="stFormSubmitButton"] button {{
          border-radius:12px !important; font-weight:600; }}
      div[data-testid="stButton"] button[kind="primary"],
      div[data-testid="stDownloadButton"] button[kind="primary"] {{
          background:#5865f2 !important; }}
      div[data-testid="stButton"] button[kind="primary"]:hover,
      div[data-testid="stDownloadButton"] button[kind="primary"]:hover {{
          background:#4752c4 !important; }}
      div[data-testid="stButton"] button[kind="secondary"],
      div[data-testid="stDownloadButton"] button[kind="secondary"] {{
          background:{surface} !important; color:{ink} !important;
          border:1px solid {hairline} !important; }}
      /* Analyze CTA = electric green (highest intent) */
      div[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {{
          background:#35ed7e !important; color:#000000 !important;
          font-weight:700; }}
      div[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {{
          background:#2ad968 !important; }}
      /* ---- segmented control: surface chips, Blurple active ---- */
      div[data-testid="stButtonGroup"] {{ flex-wrap:nowrap !important; overflow-x:auto; }}
      div[data-testid="stButtonGroup"] [data-variant] {{
          border-radius:12px !important; flex:1 1 auto; white-space:nowrap;
          font-weight:600; }}
      /* ---- toggle ---- */
      div[data-testid="stToggle"] span[role="switch"] {{
          background:#5865f2 !important; }}
      /* ---- feature card grid ---- */
      .rcn-feature {{
          background:{surface}; border:1px solid {hairline};
          border-radius:16px; padding:22px; height:100%;
          transition:transform .15s ease, box-shadow .15s ease; }}
      .rcn-feature:hover {{
          transform:translateY(-3px);
          box-shadow:0 6px 26px rgba(88,101,242,.22); }}
      .rcn-feature .rcn-ico {{
          font-size:30px; margin-bottom:10px; display:inline-block;
          background:rgba(88,101,242,.16); border-radius:14px;
          padding:8px 10px; }}
      .rcn-feature h3 {{ margin:0 0 6px 0; font-size:18px; font-weight:700; }}
      .rcn-feature p {{ margin:0; font-size:13.5px; color:{muted}; }}
    </style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------- helpers
def to_markdown(rec):
    """Markdown download: use the core renderer when available, light fallback
    when the core is missing (sample-data mode can still download a file)."""
    try:
        from docproc.nlp.report import render_markdown

        return render_markdown(rec)
    except Exception:
        lines = ["# Understanding Report", "",
                 f"- Source: `{rec.get('source', '')}`"]
        dt = rec.get("doc_type") or {}
        if dt.get("label"):
            name, _, _ = label_meta(dt["label"])
            conf = dt.get("confidence")
            lines.append(f"- Document type: **{name}**"
                         + (f" (confidence {conf:.0%})" if conf else ""))
        totals = (rec.get("structure") or {}).get("stats") or {}
        if totals:
            lines.append(f"- Stats: {totals.get('words', 0)} words · "
                         f"{totals.get('sentences', 0)} sentences · "
                         f"{totals.get('paragraphs', 0)} paragraphs")
        kws = rec.get("keywords") or []
        if kws:
            terms = [k["term"] if isinstance(k, dict) else str(k) for k in kws]
            lines += ["", "## Keywords", ", ".join(terms)]
        fd = (rec.get("fields") or {}).get("fields") or {}
        if fd:
            lines += ["", "## Extracted fields", ""]
            lines += [f"- **{k}**: {v}" for k, v in fd.items()]
        sm = rec.get("summary") or {}
        body = sm.get("text") or " ".join(
            s["text"] for s in (sm.get("sentences") or []))
        if body:
            lines += ["", "## Summary", "", body]
        return "\n".join(lines)


def label_meta(label):
    name, icon, color = CLASS_META.get(label, (label or "Unknown",
                                               "❓", "#888888"))
    return name, icon, color


def type_card(doc_type):
    lbl = doc_type.get("label")
    conf = doc_type.get("confidence")
    via = doc_type.get("via") or ""
    name, icon, color = label_meta(lbl)
    sub = f"Confidence {conf:.0%}" if conf is not None else \
          {"rule_cues": "Identified by content rules",
           "low_confidence": "Low confidence — not guessing"}.get(
              via, "Document type could not be determined")
    name, sub = html.escape(str(name)), html.escape(str(sub))
    st.markdown(f"""
    <div style='background:{color};color:#fff;border-radius:12px;padding:18px 22px;
                display:flex;align-items:center;gap:14px'>
      <span style='font-size:34px'>{icon}</span>
      <div><div style='font-size:19px;font-weight:700'>{name}</div>
      <div style='opacity:.85;font-size:13px'>{sub}</div></div>
    </div>""", unsafe_allow_html=True)


def kw_cloud(keywords):
    """keywords: [{'term','score','count'}, ...] per the seam contract."""
    chips = []
    n = max(len(keywords), 1)
    for i, item in enumerate(keywords):
        term = item["term"] if isinstance(item, dict) else str(item)
        size = 17 - 7 * i // max(n - 1, 1)
        chips.append(f"<span class='rcn-chip' style='font-size:{size}px'>"
                     f"{html.escape(term)}</span>")
    st.markdown("".join(chips), unsafe_allow_html=True)
    st.caption(f"Top {len(keywords)} most distinctive phrases in this document.")


def kw_chart(keywords):
    """Horizontal bar chart: keyword salience (Altair, theme-aware color)."""
    import altair as alt
    import pandas as pd

    rows = []
    for item in keywords:
        if not isinstance(item, dict):
            continue
        rows.append({"Keyword": str(item.get("term", "?")),
                     "Salience": float(item.get("score", 0) or 0),
                     "Count": int(item.get("count", 0) or 0)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    chart = (alt.Chart(df)
             .mark_bar(cornerRadius=6)
             .encode(
                 x=alt.X("Salience:Q", title="Salience",
                         scale=alt.Scale(nice=True)),
                 y=alt.Y("Keyword:N", sort="-x", title=None),
                 tooltip=["Keyword", "Salience", "Count"],
                 color=alt.value(BRAND))
             .properties(height=max(160, 26 * len(df))))
    st.altair_chart(chart, width='stretch')
    st.caption("Salience = term frequency × how locally concentrated the term is "
               "(in-document TF-IDF) — frequent and locally focused terms rank higher.")


def topic_chart(topics, mixture):
    """Donut chart: topic share within the document."""
    import altair as alt
    import pandas as pd

    rows = []
    for i, t in enumerate(topics):
        share = mixture[i] if i < len(mixture) else None
        if share is None:
            continue
        rows.append({"Topic": t.get("label") or f"Topic {t.get('id', i) + 1}",
                     "Share": round(float(share) * 100, 1),
                     "Top words": ", ".join(t.get("top_words", [])[:4])})
    if not rows:
        return
    df = pd.DataFrame(rows)
    dark = st.session_state.get("dark", True)
    scheme = "blues" if dark else "tealblues"
    chart = (alt.Chart(df)
             .mark_arc(innerRadius=48, outerRadius=95)
             .encode(
                 theta=alt.Theta("Share:Q", stack=True),
                 color=alt.Color("Topic:N", scale=alt.Scale(scheme=scheme),
                                 legend=None),
                 tooltip=["Topic", "Share", "Top words"])
             .properties(height=210))
    st.altair_chart(chart, width='stretch')
    st.caption("Share = the fraction of the document assigned to that topic.")


def topic_bars(topics, mixture):
    for i, t in enumerate(topics):
        head = t.get("label") or f"Topic {t.get('id', i) + 1}"
        share = mixture[i] if i < len(mixture) else None
        pct = round((share or 0) * 100)
        st.markdown(f"**{html.escape(str(head))}**")
        track = SURFACE_ONYX if st.session_state.get("dark") else "#e3e6f7"
        st.markdown(
            f"<div style='background:{track};border-radius:8px;height:14px'>"
            f"<div style='background:{BRAND};width:{pct}%;height:14px;"
            f"border-radius:8px'></div></div>", unsafe_allow_html=True)
        left, right = st.columns([1, 4])
        left.caption(f"~{pct}% of the document")
        right.write(", ".join(t.get("top_words", [])[:8]))
        st.divider()
    st.caption("The number of topics is chosen automatically for the most separable groups.")


# ═══════════════════════════════════════════════════════════════════
# Algorithm charts (matplotlib, shared theme)
# ═══════════════════════════════════════════════════════════════════

INK = "#111827"
CYAN = "#06b6d4"
CYAN_LIGHT = "#67e8f9"
PAPER = "#f8fafc"
ACCENT = "#0284c7"
GRAY = "#6b7280"
PALETTE = ["#06b6d4", "#0284c7", "#0369a1", "#075985", "#0c4a6e", "#0891b2",
           "#0e7490", "#155e75", "#164e63", "#1e3a5f"]

def _algo_style():
    """Trả về style dict cho matplotlib, tôn trọng dark/light theme."""
    dark = st.session_state.get("dark", True)
    ink = "#ffffff" if dark else "#111827"
    paper = "#0a0d3a" if dark else "#f8fafc"
    grid = "#2a2f63" if dark else "#e5e7eb"
    return {
        "axes.edgecolor": ink, "axes.labelcolor": ink,
        "xtick.color": ink, "ytick.color": ink, "text.color": ink,
        "figure.facecolor": paper, "axes.facecolor": paper,
        "grid.color": grid, "grid.linestyle": "--", "grid.linewidth": 0.5,
        "legend.facecolor": paper, "legend.edgecolor": grid,
        "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
    }


def chart_cnn():
    """CNN Training Curves (từ runs/E1/history.csv)."""
    history_path = ROOT / "runs" / "E1" / "history.csv"
    if not history_path.exists():
        return None
    with open(history_path, encoding="utf-8") as fh:
        history = list(csv.DictReader(fh))
    epochs = [int(r["epoch"]) for r in history]
    acc = [float(r["accuracy"]) for r in history]
    val_acc = [float(r["val_accuracy"]) for r in history]

    with plt.rc_context(_algo_style()):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(epochs, acc, "o-", color=CYAN, linewidth=2, markersize=3, label="Train Accuracy")
        ax.plot(epochs, val_acc, "s-", color=ACCENT, linewidth=2, markersize=3, label="Val Accuracy")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title("CNN — Training curves (Architecture A, 64×64)")
        ax.legend(loc="lower right"); ax.grid(zorder=0)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
    return fig


def chart_svm_confusion():
    """Confusion Matrix (từ runs/E1/confusion_matrix.csv)."""
    cm_path = ROOT / "runs" / "E1" / "confusion_matrix.csv"
    if not cm_path.exists():
        return None
    with open(cm_path, encoding="utf-8") as fh:
        cm = list(csv.DictReader(fh))
    classes = list(cm[0].keys())[1:]
    n = len(classes)
    mat = np.zeros((n, n), dtype=int)
    for i, row in enumerate(cm):
        for j, cls in enumerate(classes):
            mat[i, j] = int(row[cls])

    with plt.rc_context(_algo_style()):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cmap = LinearSegmentedColormap.from_list("cyan", ["white", CYAN, ACCENT])
        ax.imshow(mat, cmap=cmap, aspect="auto")
        for i in range(n):
            for j in range(n):
                color = "white" if mat[i, j] > mat.max() * 0.5 else \
                        st.session_state.get("dark", True) and "#ffffff" or INK
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
        ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        total = mat.sum()
        correct = mat.diagonal().sum()
        ax.set_title(f"CNN Confusion Matrix ({correct}/{total} correct, {total} test)")
        plt.tight_layout()
    return fig


def chart_tfidf_algo(keywords):
    """TF-IDF bar chart (from real analysis results)."""
    if not keywords:
        return None
    terms = [k["term"] if isinstance(k, dict) else str(k) for k in keywords[:10]]
    scores = [k["score"] if isinstance(k, dict) else 0 for k in keywords[:10]]
    # Reverse so the longest bar is on top
    terms.reverse(); scores.reverse()

    with plt.rc_context(_algo_style()):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        colors = [CYAN if s < max(scores) else ACCENT for s in scores]
        ax.barh(terms, scores, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        for bar, s in zip(ax.patches, scores):
            ax.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{s:.3f}", va="center", fontsize=8, color=INK)
        ax.set_xlabel("TF-IDF Score"); ax.set_title("TF-IDF — Top keywords (in-document)")
        ax.grid(axis="x", zorder=0)
        ax.set_xlim(0, max(scores) * 1.2)
        plt.tight_layout()
    return fig


def chart_lda_coherence(topics_data):
    """UMass Coherence curve (from analysis results)."""
    curve = topics_data.get("coherence_curve") or []
    if len(curve) < 2:
        return None
    ks = [c["k"] for c in curve]
    coh = [c["umass"] for c in curve]
    best_k = topics_data.get("k")

    with plt.rc_context(_algo_style()):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(ks, coh, "o-", color=INK, linewidth=2, markersize=8,
                markerfacecolor=CYAN, markeredgecolor=INK, zorder=3)
        if best_k and best_k in ks:
            idx = ks.index(best_k)
            ax.plot(ks[idx], coh[idx], "o", color=ACCENT, markersize=12,
                    markeredgecolor=INK, linewidth=2, zorder=4)
            ax.annotate(f"k={best_k}", xy=(ks[idx], coh[idx]),
                        xytext=(ks[idx] + 0.5, coh[idx] + 0.3),
                        arrowprops={"arrowstyle": "->", "color": INK},
                        fontsize=10, fontweight="bold", color=ACCENT)
        ax.set_xlabel("Number of topics (k)"); ax.set_ylabel("UMass Coherence")
        ax.set_title("LDA — Choosing k with UMass Coherence")
        ax.set_xticks(ks); ax.grid(zorder=0)
        plt.tight_layout()
    return fig


def chart_pca_algo():
    """PCA Scree plot (from TF-IDF of 12 sample sentences)."""
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer

    samples = [
        "Hệ thống RCN phân tích tài liệu tiếng Việt bằng học máy.",
        "Pipeline 5 lớp xử lý tuần tự từ cấu trúc đến tóm tắt.",
        "TF-IDF trích xuất từ khóa quan trọng trong văn bản.",
        "LDA khám phá chủ đề ẩn dựa trên phân phối từ.",
        "CNN phân loại ảnh tài liệu quét vào 6 nhóm khác nhau.",
        "MMR tạo bản tóm tắt trích xuất không trùng lặp.",
        "Hệ thống hoạt động ngoại tuyến không cần kết nối Internet.",
        "PCA giảm chiều dữ liệu từ 5000 đặc trưng xuống 35 chiều.",
        "K-Means gom cụm từ khóa thành các nhóm chủ đề.",
        "Toàn bộ pipeline chạy trên máy tính cá nhân thông thường.",
        "Mô hình vit5-base tóm tắt tóm lược văn bản tiếng Việt.",
        "Đánh giá bằng ROUGE-1, ROUGE-2, ROUGE-L trên tập test.",
    ]
    vec = TfidfVectorizer(max_features=100)
    X = vec.fit_transform(samples).toarray()
    pca = PCA(random_state=42).fit(X)
    n_bars = min(10, len(pca.explained_variance_ratio_))
    var = pca.explained_variance_ratio_[:n_bars]
    cumsum = np.cumsum(var)

    with plt.rc_context(_algo_style()):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        x = np.arange(1, n_bars + 1)
        colors = [ACCENT if i == 0 else CYAN for i in range(n_bars)]
        ax.bar(x, var * 100, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        ax2 = ax.twinx()
        ax2.plot(x, cumsum * 100, "o-", color=INK, linewidth=2.5, markersize=5, zorder=4)
        ax2.set_ylabel("Cumulative %", color=INK)
        ax2.set_ylim(0, 105)
        ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance Explained (%)")
        ax.set_title("PCA — Explained variance (scree plot)")
        ax.set_xticks(x); ax.grid(axis="y", zorder=0)
        ax.text(1, var[0] * 100 + 1, f"{var[0]*100:.1f}%", ha="center", fontsize=9, fontweight="bold", color=ACCENT)
        plt.tight_layout()
    return fig


def chart_kmeans_algo():
    """K-Means elbow method (from TF-IDF of 12 sample sentences)."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    samples = [
        "Hệ thống RCN phân tích tài liệu tiếng Việt bằng học máy.",
        "Pipeline 5 lớp xử lý tuần tự từ cấu trúc đến tóm tắt.",
        "TF-IDF trích xuất từ khóa quan trọng trong văn bản.",
        "LDA khám phá chủ đề ẩn dựa trên phân phối từ.",
        "CNN phân loại ảnh tài liệu quét vào 6 nhóm khác nhau.",
        "MMR tạo bản tóm tắt trích xuất không trùng lặp.",
        "Hệ thống hoạt động ngoại tuyến không cần kết nối Internet.",
        "PCA giảm chiều dữ liệu từ 5000 đặc trưng xuống 35 chiều.",
        "K-Means gom cụm từ khóa thành các nhóm chủ đề.",
        "Toàn bộ pipeline chạy trên máy tính cá nhân thông thường.",
        "Mô hình vit5-base tóm tắt tóm lược văn bản tiếng Việt.",
        "Đánh giá bằng ROUGE-1, ROUGE-2, ROUGE-L trên tập test.",
    ]
    vec = TfidfVectorizer(max_features=100)
    X = vec.fit_transform(samples).toarray()
    X_scaled = StandardScaler().fit_transform(X)
    ks = range(1, min(11, len(samples) + 1))
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
                for k in ks]
    diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias) - 1)]
    elbow_k = diffs.index(max(diffs)) + 2 if diffs else 4

    with plt.rc_context(_algo_style()):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(list(ks), inertias, "o-", color=INK, linewidth=2.5, markersize=8,
                markerfacecolor=CYAN, markeredgecolor=INK, zorder=3)
        ax.axvline(x=elbow_k, color=ACCENT, linestyle="--", linewidth=2, alpha=0.8,
                   label=f"Elbow: k={elbow_k}")
        ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Inertia (WCSS)")
        ax.set_title("K-Means — Elbow method for choosing the cluster count")
        ax.set_xticks(list(ks)); ax.legend(); ax.grid(zorder=0)
        plt.tight_layout()
    return fig


# ML/DL algorithms used at each analysis step (shown in the UI).
ALGOS = [
    ("SVM", "e0b_svm", "Text classification",
     "TF-IDF → LinearSVC; GridSearchCV picks hyperparameters (5-fold).",
     "text_model_svm.joblib"),
    ("CNN", "e1_cnn", "Scanned-image classification",
     "Conv2D→MaxPool→Dense(softmax), trained on the RVL-CDIP subset.",
     "runs/E1/best.keras"),
    ("TF-IDF", "tfidf", "Keyword salience",
     "score = tf × idf (idf over paragraphs) — keywords & vectorization.",
     "text_vectorizer.joblib"),
    ("LDA", "lda", "Topic grouping",
     "Latent Dirichlet Allocation; k chosen by UMass coherence.",
     "sklearn.decomposition"),
    ("K-means + PCA", "km_pca", "Topic labels",
     "Clusters keywords → one representative label per topic group.",
     "sklearn.cluster"),
    ("MMR", "mmr", "Extractive summary",
     "score = λ·relevance − (1−λ)·redundancy; picks non-duplicate sentences.",
     "docproc.nlp.summary"),
    ("Seq2Seq T5", "t5", "Abstractive summary",
     "Transformer vit5-base fine-tuned on Vietnamese corpora (offline).",
     "vit5_v1"),
]


def algo_panel():
    """Grid of the project ML/DL algorithm cards (real core data)."""
    dark = st.session_state.get("dark", True)
    card_bg = SURFACE_INDIGO if dark else "#ffffff"
    border = SURFACE_ONYX if dark else "#d7dbef"
    sub = "#a3a9d8" if dark else "#5a6189"
    rows_html = []
    for name, key, task, desc, src in ALGOS:
        rows_html.append(f"""
        <div style='background:{card_bg};border:1px solid {border};
             border-radius:16px;padding:14px 16px;display:flex;
             gap:12px;align-items:flex-start'>
          <div style='min-width:74px;text-align:center'>
            <div style='background:rgba(88,101,242,.18);color:{BRAND_ON};
                 font-weight:700;border-radius:8px;padding:4px 6px;
                 font-size:13px'>{name}</div>
            <div style='font-size:10.5px;color:{sub};margin-top:3px'>{task}</div>
          </div>
          <div style='font-size:13px;line-height:1.55;color:{sub}'>
            {desc}<br>
            <code style='font-size:11px;color:{LINK_CYAN}'>{src}</code>
          </div>
        </div>""")
    st.markdown("<div style='display:grid;grid-template-columns:1fr 1fr;"
                "gap:10px'>" + "".join(rows_html) + "</div>",
                unsafe_allow_html=True)


def demo_cta(help_text="No document handy? Preview the UI with sample data:",
             cta_key: str | None = None):
    """Shared CTA button: switch to the sample-data demo (no core needed)."""
    st.caption(help_text)
    key = cta_key or f"cta_demo_{abs(hash(help_text)) % 100000}"
    if st.button("🧪 View with sample data", key=key):
        st.session_state["rec"] = demo_record(None, 3)
        st.rerun()


def highlight_text(text: str, keywords: list, dark: bool) -> str:
    """Highlight keyphrases (uni+bi-gram, case-insensitive)."""
    terms = [k["term"] if isinstance(k, dict) else str(k) for k in keywords]
    if not text or not terms:
        return html.escape(text or "")
    pattern = re.compile(
        "(" + "|".join(re.escape(t) for t in
                       sorted(terms, key=len, reverse=True)) + ")",
        re.IGNORECASE)
    bg = "rgba(88,101,242,.22)" if dark else "rgba(88,101,242,.15)"
    out, last = [], 0
    for m in pattern.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        out.append(f"<mark style='background:{bg};color:inherit;"
                   f"border-radius:4px;padding:0 2px'>"
                   f"{html.escape(m.group(0))}</mark>")
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


# ------------------------------------------------------------------ sidebar --
_THEME_FILE = ROOT / ".rcn_theme"


def _load_theme() -> bool:
    try:
        return _THEME_FILE.read_text().strip() != "light"
    except Exception:
        return True   # dark by default


if "dark" not in st.session_state:
    st.session_state["dark"] = _load_theme()
inject_css(st.session_state["dark"])

with st.sidebar:
    st.header("📄 RCN Studio")
    st.caption("Every document, one clear view — fully on your machine.")
    st.divider()

    dark_new = st.toggle("🌙 Dark theme",
                         value=st.session_state["dark"],
                         key="theme_toggle",
                         help="Switch light/dark (remembered across restarts)")
    if dark_new != st.session_state["dark"]:
        st.session_state["dark"] = dark_new
        try:
            _THEME_FILE.write_text("dark" if dark_new else "light")
        except Exception:  # noqa: S110  # best-effort theme persistence
            pass
        st.rerun()

    src = st.radio("Data source",
                   ["🧪 Sample data (preview the UI)",
                    "📄 Your document",
                    "📁 Folder (batch)"])
    use_demo = src.startswith("🧪")
    use_batch = src.startswith("📁")

    up, pasted = None, ""
    folder_path = ""
    if use_batch:
        folder_path = st.text_input(
            "Folder to scan",
            value=str(ROOT / "datasets" / "text"),
            help="Every text file in the folder (recursively) is scanned "
                 "into a table + CSV.")
    elif not use_demo:
        up = st.file_uploader("Drag & drop your document here", type=SUPPORTED)
        st.caption("TXT · MD · HTML · DOCX · PDF (incl. scanned) · images")
        pasted = st.text_area("...or paste text:", height=130,
                              placeholder="Paste the content you want analyzed here")

    st.markdown("---")
    st.markdown("### ⚙️ Summary options")
    mode_lbl = st.segmented_control(
        "Mode", ["Extract sentences", "Generate new text"],
        default="Extract sentences",
        help="“Generate new text” needs the small local model; if it is missing "
             "it falls back to sentence extraction.")
    MODE_MAP = {"Extract sentences": "extractive",
                "Generate new text": "abstractive"}
    k_sum = st.slider("Summary sentences", 1, 7, 3)

    ckpts = available_checkpoints()
    model_lbl = None
    if mode_lbl == "Generate new text":
        if ckpts:
            default_i = ckpts.index("vit5_v1") if "vit5_v1" in ckpts else 0
            model_lbl = st.selectbox(
                "Summarizer model", ckpts, index=default_i,
                help="Local checkpoint that writes the new text. Only models "
                     "that load on this machine are listed; vit5_v1 is the "
                     "default.")
        else:
            st.caption("⚠️ No local summarizer model found under "
                       "models/artifacts/ — will fall back to sentence "
                       "extraction.")

    can_go = (use_demo
              or use_batch
              or (not use_demo and not use_batch
                  and (up is not None or bool(pasted.strip()))))
    go = st.button("▶️  Analyze", type="primary", width="stretch",
                   key="analyze", disabled=not can_go,
                   shortcut="Ctrl+Enter",
                   help="Nothing to analyze yet? Pick a file, paste text or enter a "
                        "folder first. Ctrl+Enter runs it fast.")


def skeleton_block(rows: int = 3, width_pct: tuple = (72, 95, 60)):
    """Shimmer skeleton instead of a spinner — feels faster."""
    bars = "".join(
        f"<div style='height:13px;border-radius:7px;margin-bottom:11px;"
        f"width:{width_pct[min(i, len(width_pct) - 1)]}%;"
        f"background:linear-gradient(90deg,#1e2353 25%,#2a3168 50%,"
        f"#1e2353 75%);background-size:200% 100%;"
        f"animation:rcn-shimmer 1.3s infinite'></div>"
        for i in range(rows))
    st.markdown(f"<style>@keyframes rcn-shimmer{{0%{{background-position:"
                f"200% 0}}100%{{background-position:-200% 0}}}}</style>"
                f"<div>{bars}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------- analysis ---
if go:
    if use_demo:
        st.session_state["rec"] = demo_record(MODE_MAP[mode_lbl], k_sum,
                                              model_lbl)
        st.toast("Showing sample data 🧪", icon="🧪")
    elif use_batch:
        fdir = Path(folder_path)
        if not fdir.is_dir():
            st.sidebar.warning("Folder does not exist — check the path.")
        files = sorted(p for p in fdir.rglob("*")
                       if p.is_file()
                       and not p.name.endswith((".understanding.json",
                                                ".understanding.md"))
                       and p.suffix.lower() in
                       {".txt", ".md", ".markdown", ".html", ".htm",
                        ".docx", ".pdf"})
        if not files:
            st.sidebar.warning("No supported files found in this folder.")
        else:
            from docproc.nlp.report import understand_file

            rows, errs = [], 0
            prog = st.progress(0.0, text="Scanning folder...")
            for i, fp in enumerate(files):
                err_txt = ""
                try:
                    r = understand_file(fp, summary_mode=None,
                                        summary_k=None)
                    doc_lbl = (r.get("doc_type") or {}).get("label")
                except Exception as exc:
                    errs += 1
                    doc_lbl = None
                    err_txt = f"{type(exc).__name__}: {exc}"[:90]
                rows.append({
                    "File": fp.name,
                    "Type": label_meta(doc_lbl)[0] if doc_lbl else "— error —",
                    "Words": ((r.get("structure") or {}).get("stats", {})
                           .get("words", 0)) if doc_lbl else 0,
                    "Sentences": ((r.get("structure") or {}).get("stats", {})
                            .get("sentences", 0)) if doc_lbl else 0,
                    "Keywords": len(r.get("keywords") or []) if doc_lbl else 0,
                    "Fields": (r.get("fields") or {}).get(
                        "matched", 0) if doc_lbl else 0,
                    "Note": err_txt,
                })
                prog.progress((i + 1) / len(files),
                              text=f"{i + 1}/{len(files)}: {fp.name}")
            st.session_state["batch_rows"] = rows
            st.session_state["batch_errs"] = errs
            st.toast(f"Done — {len(files)} files"
                     + (f" ({errs} errors)" if errs else ""), icon="📁")
    elif up is None and not pasted.strip():
        st.sidebar.warning("Upload a file or paste text first.")
    else:
        # Show the skeleton IMMEDIATELY before the seam runs (abstractive can take 5-25s)
        ph = st.empty()
        with ph.container():
            st.markdown("**Analyzing document...**")
            skeleton_block(rows=5)
        c1, c2, c3 = st.columns(3)
        for c in (c1, c2, c3):
            with c:
                st.markdown(
                    "<div style='height:74px;border-radius:12px;"
                    "background:linear-gradient(90deg,#1e2353 25%,#2a3168 50%,"
                    "#1e2353 75%);background-size:200% 100%;"
                    "animation:rcn-shimmer 1.3s infinite'></div>",
                    unsafe_allow_html=True)
        tmp_path = None
        try:
            from docproc.nlp.report import understand, understand_file

            if up is not None:
                # the seam needs a real path on disk -> spill the upload to a temp file
                # (keep the suffix so magic-bytes + extension detect the right kind)
                suffix = Path(up.name or "upload.bin").suffix or ".bin"
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix) as tmp:
                    tmp.write(up.getvalue())
                    tmp_path = Path(tmp.name)
                rec = understand_file(tmp_path,
                                      summary_mode=MODE_MAP[mode_lbl],
                                      summary_k=k_sum,
                                      summary_checkpoint=model_lbl)
                rec["source"] = up.name  # real name instead of the temp path
            else:
                rec = understand(pasted, source="(pasted text)",
                                 summary_mode=MODE_MAP[mode_lbl],
                                 summary_k=k_sum,
                                 summary_checkpoint=model_lbl)
            st.session_state["rec"] = rec
            # Session history: keep the 8 most recent analyses
            hist = st.session_state.setdefault("history", [])
            hist.insert(0, {
                "source": str(rec.get("source", "")),
                "time": datetime.now().astimezone().strftime("%H:%M:%S"),
                "label": (rec.get("doc_type") or {}).get("label") or "?",
                "rec": rec,
            })
            del hist[8:]
        except Exception as exc:
            st.session_state.pop("rec", None)
            st.error("Could not process this document.")
            st.code(f"{type(exc).__name__}: {exc}", language=None)
            st.info("💡 You can still preview the UI with **Sample data** "
                    "in the sidebar.")
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            ph.empty()  # kết quả render bên dưới — không để skeleton treo mãi

rec = st.session_state.get("rec")
is_demo = bool(rec) and str(rec.get("source", "")).startswith("(sample data)")

# ------------------------------------------------------------ batch result --
batch_rows = st.session_state.get("batch_rows")
if batch_rows:
    st.markdown(f"""<div class='rcn-hero'><h1>Folder scan results</h1>
      <p>{len(batch_rows)} files · {st.session_state.get('batch_errs', 0)} errors
      — press “Analyze” to rescan.</p></div>""", unsafe_allow_html=True)
    st.write("")
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(batch_rows[0].keys()))
    writer.writeheader()
    writer.writerows(batch_rows)
    ctbl, ccsv = st.columns([3, 1])
    with ctbl:
        st.dataframe(batch_rows, hide_index=True, width='stretch')
    with ccsv:
        st.download_button("Download CSV", buf.getvalue().encode("utf-8-sig"),
                           "batch_understanding.csv", "text/csv",
                           width='stretch',
                           type="primary", icon=":material/table:",
                           help="Folder scan results as CSV (Excel-compatible)")
        if st.button("Clear batch results", width="stretch",
                     type="secondary", icon=":material/delete:"):
            for k in ("batch_rows", "batch_errs"):
                st.session_state.pop(k, None)
            st.rerun()
    st.stop()

# ------------------------------------------------------------------ result --
history = st.session_state.get("history") or []
if history and not is_demo:
    with st.expander(f"🕘 Session history ({len(history)} analyses)"):
        for i, h in enumerate(history):
            c1, c2, c3, c4 = st.columns([4, 2, 3, 1])
            c1.write(f"**{html.escape(Path(h['source']).name or h['source'])}**")
            c2.caption(h["time"])
            name_h, _, _ = label_meta(h.get("label"))
            c3.caption(name_h)
            if c4.button("View", key=f"hist_{i}", type="secondary",
                         icon=":material/visibility:",
                         help="Reopen this analysis result"):
                st.session_state["rec"] = h["rec"]
                st.session_state.pop("batch_rows", None)
                st.rerun()

if rec:
    md_txt = to_markdown(rec)
    js_bytes = json.dumps(rec, ensure_ascii=False, indent=2).encode("utf-8")
    raw_name = Path(str(rec.get("source", "report"))).stem or "report"
    name = re.sub(r"[^\w\-]+", "_", raw_name).strip("_")[:60] or "report"

    if is_demo:
        st.markdown(
            "<div style='background:rgba(236,72,189,.14);border:1px solid "
            "rgba(236,72,189,.45);color:#ec48bd;padding:8px 16px;"
            "border-radius:10px;font-size:13px'>"
            "🧪 <b>Viewing sample data</b> — the UI shows a prebuilt record, "
            "no analysis engine was called.</div>",
            unsafe_allow_html=True)
        st.write("")
        hero_sub = ""
    else:
        hero_sub = ("Everything here is computed on this machine — same engine as "
                    "the CLI, identical JSON/Markdown output.")
    st.markdown(f"""<div class='rcn-hero'><h1>Analysis results</h1>
      <p><b>Every document, one clear view.</b>{' ' + hero_sub if hero_sub else ''}</p></div>""",
                unsafe_allow_html=True)
    st.write("")
    d1, d2 = st.columns([1, 1])
    d1.download_button("Download JSON", js_bytes, f"{name}.understanding.json",
                       "application/json", width="stretch",
                       type="primary", icon=":material/data_object:",
                       help="The full analysis record as JSON")
    d2.download_button("Download Markdown", md_txt.encode("utf-8"),
                       f"{name}.understanding.md", "text/markdown",
                       width="stretch", type="secondary",
                       icon=":material/article:",
                       help="Markdown report, identical to the CLI")
    st.write("")

    if rec.get("note"):  # ảnh / PDF scan -> chỉ phân loại
        st.info("This is an image or scanned page: the system only detects the document "
                "type — it does not read the text (project scope).")
        type_card(rec.get("doc_type", {}))
        st.stop()

    tab_sm, tab_ov, tab_kw, tab_tx, tab_tp, tab_fd, tab_al = st.tabs(
        ["📝 Summary", "📊 Overview",
         f"🔑 Keywords ({len(rec.get('keywords') or [])})",
         "📃 Source text", "🧩 Topics",
         f"🗂 Extracted fields ({len((rec.get('fields') or {}).get('fields') or {})})",
         "🧠 Algorithms"],
        default="📝 Summary")

    with tab_ov:
        c1, c2 = st.columns([1, 1])
        with c1:
            type_card(rec.get("doc_type", {}))
        with c2:
            totals = (rec.get("structure") or {}).get("stats", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("Words", f"{totals.get('words', 0):,}")
            m2.metric("Sentences", totals.get("sentences", 0))
            m3.metric("Paragraphs", totals.get("paragraphs", 0))
        st.caption(f"Source: `{rec.get('source')}`"
                   + (f" · file type {str(rec.get('file_type')).upper()}"
                      if rec.get("file_type") else ""))

        # ---- secondary metric row: reading density + matched fields ----
        words = totals.get("words", 0)
        sents = totals.get("sentences", 0)
        uniq = totals.get("unique_words", 0)
        m_a, m_b, m_c, m_d = st.columns(4)
        m_a.metric("Unique words", f"{uniq:,}"
                   + (f" ({uniq / words:.0%})" if words else ""))
        m_b.metric("Words / sentence",
                   f"{words / sents:.1f}" if sents else "—")
        read_min = words / 200  # ~200 từ/phút đọc trung bình
        m_c.metric("Reading time",
                   f"{read_min:.1f} min" if read_min >= 1
                   else f"{read_min * 60:.0f} sec")
        fd_ov = rec.get("fields") or {}
        flds = fd_ov.get("fields") or {}
        m_d.metric("Fields matched",
                   f"{fd_ov.get('matched', 0)}/{len(flds)}"
                   if flds else "—")

        # ---- paragraph length distribution ----
        paras = (rec.get("structure") or {}).get("paragraphs") or []
        if len(paras) >= 2:
            st.write("")
            st.markdown("**Paragraph length (words):**")
            st.bar_chart(
                {f"¶{p.get('index', i) + 1}": p.get("word_count", 0)
                 for i, p in enumerate(paras)},
                height=220, x_label="paragraph", y_label="words")

    with tab_kw:
        kws = rec.get("keywords") or []
        if kws:
            kw_cloud(kws)
            st.write("")
            kw_chart(kws)
        else:
            st.markdown("### 🔑")
            st.write("**Nothing stands out.** No distinctive phrases were found "
                     "in this document.")
            demo_cta("Longer documents produce clearer keywords.")

    with tab_tx:
        # Rebuild the source from L1 (exact sentence/paragraph order the seam saw)
        paras = ((rec.get("structure") or {}).get("paragraphs") or [])
        if paras:
            full_text = "\n\n".join(
                " ".join(p.get("sentences", [])) for p in paras)
            st.markdown(
                f"<div style='font-size:15.5px;line-height:1.9'>"
                f"{highlight_text(full_text, rec.get('keywords') or [], st.session_state.get('dark', True))}"
                f"</div>", unsafe_allow_html=True)
        else:
            st.info("No source text to show (this document was classification-only).")

    with tab_tp:
        tp = rec.get("topics") or {}
        if tp.get("topics"):
            c_donut, c_bars = st.columns([1, 2])
            with c_donut:
                topic_chart(tp["topics"], tp.get("doc_topic_mixture") or [])
            with c_bars:
                topic_bars(tp["topics"], tp.get("doc_topic_mixture") or [])
            curve = tp.get("coherence_curve") or []
            if len(curve) >= 2:
                st.divider()
                c_curve, c_formula = st.columns([3, 2])
                with c_curve:
                    st.markdown("**Topic coherence vs. number of topics (k):**")
                    st.line_chart(
                        data={c["k"]: c["umass"] for c in curve},
                        x_label="number of topics (k)", y_label="coherence")
                    st.caption(f"Peak at k={tp.get('k')} "
                               f"({tp.get('selected_by')}) → that topic count is chosen.")
                with c_formula:
                    f_bg = SURFACE_INDIGO if st.session_state.get(
                        "dark", True) else "#f5f7ff"
                    st.markdown(
                        f"<div style='background:{f_bg};'"
                        "border:1px solid rgba(88,101,242,.40);border-radius:16px;"
                        "padding:12px 14px;font-size:12.5px;line-height:1.7>"
                        "<b>UMass coherence</b> — do words inside one topic tend to "
                        "co-occur in the same paragraph?<br><br>"
                        "<code>log( D(wₜ & wᵢ) / D(wₜ) )</code><br>"
                        "<span style='opacity:.75'>D(w) = number of paragraphs with w; "
                        "wₜ, wᵢ = the two top words of the topic.</span><br><br>"
                        "Closer to 0 is more coherent → pick the highest k.</div>"
                        ,
                        unsafe_allow_html=True)
        else:
            st.markdown("### 🧩")
            st.write("**Not enough material.** The text is too short to separate "
                     "into topics.")
            demo_cta("Topic separation needs at least 3 content paragraphs — or preview:",
                     cta_key="cta_topics")

    with tab_fd:
        fd = rec.get("fields") or {}
        fields = fd.get("fields") or {}
        if fields:
            st.dataframe(
                [{"Field": k,
                  "Value": ", ".join(map(str, v)) if isinstance(v, list)
                  else str(v)}
                 for k, v in fields.items()],
                hide_index=True, width="stretch")
            ok, tot = fd.get("matched", 0), len(fields)
            if ok == tot and tot:
                st.success(f"Matched all {ok}/{tot} fields for type "
                           f"\"{label_meta(fd.get('doc_type'))[0]}\".")
            else:
                st.caption(f"Matched {ok}/{tot} fields for type "
                           f"\"{label_meta(fd.get('doc_type'))[0]}\".")
            for miss in fd.get("missing_required") or []:
                st.warning(f"⚠️ Missing required field: **{miss}**")
        else:
            st.markdown("### 🗂")
            st.write("**No fields extracted.** This document carries no signs of "
                     "numbers, dates or currency.")
            demo_cta("Invoices and receipts extract best — or try:")

    with tab_sm:
        sm = rec.get("summary") or {}
        engine = sm.get("engine")
        badge, color = ENGINE_BADGE.get(engine, (engine or "?", "#888"))
        st.markdown(f"<span style='background:{color};color:#fff;padding:6px 14px;"
                    f"border-radius:16px;font-size:14px'>{badge}</span>",
                    unsafe_allow_html=True)
        st.write("")
        sentences = sm.get("sentences")
        if engine == "abstractive":
            body = sm.get("text") or "(the model produced no text)"
            esc_body = html.escape(body)
            st.markdown(f"""><div style='font-size:16px;line-height:1.65'>"""
                        f"""{esc_body}</div>""", unsafe_allow_html=True)
            st.code(body, language=None)  # có sẵn nút copy của st.code
        elif sentences:
            body = " ".join(s["text"] for s in sentences)
            esc_body = html.escape(body)
            st.markdown(f"""><div style='font-size:16px;line-height:1.65'>"""
                        f"""{esc_body}</div>""", unsafe_allow_html=True)
            st.code(body, language=None)  # copy 1 chạm
            with st.expander("🔎 Show each selected sentence"):
                for s in sentences:
                    st.markdown(f"> ¶{s.get('paragraph', '?')} — "
                                f"{html.escape(s['text'])}")
            comp = sm.get("compression") or {}
            orig, kept = comp.get("original_sentences"), comp.get("kept")
            foot = (f"Compression: kept {kept}/{orig} sentences."
                    if kept is not None else f"From {orig} original sentences.")
            st.caption(foot)
        else:
            st.markdown("### 📝")
            st.write("**Nothing to summarize.** The content is not long enough "
                     "to surface key sentences.")
            demo_cta("Paste more content, or preview:")
        if sm.get("engine_fallback"):
            st.warning("⚠️ The abstractive model is not installed — fell back to "
                       "sentence extraction; the result is still complete.")

    with tab_al:
            st.markdown("**A visual chart for each algorithm in the pipeline:**")
            st.write("")

            # CNN
            st.markdown("#### 🖼️ CNN — Image classification")
            fig_cnn = chart_cnn()
            if fig_cnn:
                st.pyplot(fig_cnn)
                plt.close(fig_cnn)
            else:
                st.caption("⚠️ No CNN training data (missing runs/E1/history.csv).")

            st.divider()

            # SVM / Confusion Matrix
            st.markdown("#### 📊 Text classification — Confusion matrix")
            fig_svm = chart_svm_confusion()
            if fig_svm:
                st.pyplot(fig_svm)
                plt.close(fig_svm)
            else:
                st.caption("⚠️ No confusion-matrix data (missing runs/E1/confusion_matrix.csv).")

            st.divider()

            # TF-IDF
            st.markdown("#### 🔑 TF-IDF — Keywords")
            kws = rec.get("keywords") or []
            fig_tfidf = chart_tfidf_algo(kws)
            if fig_tfidf:
                st.pyplot(fig_tfidf)
                plt.close(fig_tfidf)
            else:
                st.caption("⚠️ No keywords to chart.")

            st.divider()

            # LDA Coherence
            st.markdown("#### 🧩 LDA — UMass coherence")
            tp = rec.get("topics") or {}
            fig_lda = chart_lda_coherence(tp)
            if fig_lda:
                st.pyplot(fig_lda)
                plt.close(fig_lda)
            else:
                st.caption("⚠️ Not enough topic data (needs ≥3 paragraphs).")

            st.divider()

            # PCA
            st.markdown("#### 📉 PCA — Dimensionality reduction")
            fig_pca = chart_pca_algo()
            if fig_pca:
                st.pyplot(fig_pca)
                plt.close(fig_pca)

            st.divider()

            # K-Means
            st.markdown("#### 🎯 K-Means — Clustering")
            fig_km = chart_kmeans_algo()
            if fig_km:
                st.pyplot(fig_km)
                plt.close(fig_km)

            st.divider()
            st.caption("All algorithms run offline on this machine — TF-IDF, LDA, PCA, K-Means, CNN, SVM, MMR, T5.")

else:
    st.markdown("<div class='rcn-hero'><h1>RCN STUDIO</h1></div>",
                unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    feats = [
        ("📊", "Structure at a glance",
         "Words, sentences, paragraphs — know instantly if a document is dense or light."),
        ("🔑", "Keywords & topics",
         "Distinctive phrases plus topic shares with readable labels."),
        ("🗂", "Structured fields",
         "Numbers, dates, money, parties... cleanly extracted into a table."),
    ]
    for col, (icon, title, desc) in zip((c1, c2, c3), feats):
        with col:
            st.markdown(
                f"<div class='rcn-feature'><span class='rcn-ico'>{icon}</span>"
                f"<h3>{title}</h3><p>{desc}</p></div>",
                unsafe_allow_html=True)
    st.write("")
    st.info("💡 Tip: pick **🧪 Sample data** in the sidebar to preview the UI "
            "with zero preparation.", icon="👈")
    st.write("")
    st.markdown("#### 🧠 ML/DL algorithms")
    algo_panel()
