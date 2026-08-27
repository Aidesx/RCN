"""RCN Studio - web UI cho việc hiểu tài liệu (tách rời core).

Run (from RCN/):
    .venv/Scripts/python -m streamlit run scripts/app.py

Hai nguồn dữ liệu:
  - "Du lieu mau": record demo dong san trong file -> xem duoi UI khong can core.
  - "Tai lieu cua ban": goi seam thuc understand()/understand_file() khi core san sang.
Khong sua bat ky file core nao; ket qua JSON/Markdown giong het CLI.
"""
import csv
import html
import json
import re
import sys
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import streamlit as st
except ImportError:
    print("Thieu streamlit. Cai: .venv/Scripts/pip install streamlit")
    sys.exit(1)

st.set_page_config(page_title="RCN Studio", page_icon="📄", layout="wide")

# ------------------------------------------------------------------ constants
# Palette "Blurple" (Discord-style): deep-indigo canvas + Blurple/green/magenta.
# Light = nền #f5f7ff, Dark = canvas #0a0d3a. Display font Space Grotesk.
BRAND = "#5865f2"          # Blurple — Discord brand
BRAND_ON = "#ffffff"
GREEN_CTA = "#35ed7e"      # electric green — high-intent actions
MAGENTA = "#ec48bd"        # vibrant magenta — gradient feature panels
LINK_CYAN = "#00b0f4"      # inline link color on dark surfaces
CANVAS = "#0a0d3a"         # deep-indigo page canvas
SURFACE_INDIGO = "#1e2353" # raised indigo panel
SURFACE_ONYX = "#23272a"   # dark UI surface

CLASS_META = {
    "invoice": ("Hóa đơn", "🧾", BRAND),           # Blurple
    "receipt": ("Biên lai", "✅", GREEN_CTA),      # electric green
    "report": ("Báo cáo", "📊", MAGENTA),          # magenta
    "letter": ("Thư từ", "✉️", LINK_CYAN),         # link cyan
    "form": ("Biểu mẫu", "📋", "#a06cd5"),         # violet accent
    "article": ("Bài viết", "📰", "#7c86c8"),      # muted indigo
}
ENGINE_BADGE = {
    "extractive": ("📋 Trích xuất các câu quan trọng nhất", BRAND),
    "abstractive": ("✍️ Sinh đoạn văn mới bằng model nhỏ", MAGENTA),
}
SUPPORTED = ["md", "txt", "html", "htm", "docx", "pdf", "png", "jpg", "jpeg"]

# ------------------------------------------------------------ demo record ---
DEMO_RECORD = {
    "source": "(dữ liệu mẫu) hoa_don_mau.txt",
    "file_type": "txt",
    "doc_type": {"label": "invoice", "via": "e0b_svm", "confidence": 0.94},
    "structure": {"stats": {"characters": 1124, "words": 186,
                            "unique_words": 97, "sentences": 21,
                            "paragraphs": 6},
                  "paragraphs": [
                      {"index": 0, "sentence_count": 2, "word_count": 38,
                       "sentences": [
                           "Công ty TNHH An Phát xin gửi hóa đơn giá trị gia "
                           "tăng số HD-2026-0841 ngày 20/08/2026.",
                           "Hóa đơn áp dụng cho lô hàng giấy photo A4 theo "
                           "hợp đồng cung ứng văn phòng phẩm."]},
                      {"index": 1, "sentence_count": 2, "word_count": 41,
                       "sentences": [
                           "Tổng giá trị thanh toán sau thuế VAT 8% là "
                           "45.600.000 đồng.",
                           "Hạn công nợ 30 ngày kể từ ngày xuất hóa đơn, vui "
                           "lòng thanh toán đúng hạn."]},
                      {"index": 2, "sentence_count": 1, "word_count": 18,
                       "sentences": [
                           "Mọi thắc mắc về khoản mục vui lòng liên hệ phòng "
                           "kế toán trong vòng 7 ngày làm việc."]}]},
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
            {"text": "Công ty TNHH An Phát xin gửi hóa đơn giá trị gia tăng "
                     "số HD-2026-0841 ngày 20/08/2026 cho lô hàng giấy photo "
                     "A4 theo hợp đồng cung ứng văn phòng phẩm.", "paragraph": 1,
             "score": 0.83},
            {"text": "Tổng giá trị thanh toán sau thuế VAT 8% là "
                     "45.600.000 đồng, hạn công nợ 30 ngày kể từ ngày xuất "
                     "hóa đơn.", "paragraph": 4, "score": 0.91},
            {"text": "Mọi thắc mắc về khoản mục vui lòng liên hệ phòng kế "
                     "toán trong vòng 7 ngày làm việc.", "paragraph": 6,
             "score": 0.41},
        ],
        "compression": {"original_sentences": 21, "kept": 3},
    },
}

# Bản tóm tắt abstractive giả lập cho chế độ mẫu (để review được cả 2 badge)
DEMO_ABSTRACTIVE_TEXT = (
    "Hóa đơn số HD-2026-0841 do Công ty TNHH An Phát phát hành ngày 20/08/2026 "
    "cho Công ty CP Minh Khoa, ghi nhận lô hàng giấy photo A4 với tổng giá trị "
    "thanh toán 45.600.000 đồng sau thuế VAT 8%, hạn công nợ 30 ngày."
)


def demo_record(mode, k_sum):
    """Record mẫu tuân theo lựa chọn trên sidebar (chế độ + số câu),
    để mọi control đều có tác dụng khi review giao diện."""
    rec = json.loads(json.dumps(DEMO_RECORD))  # deep copy đơn giản
    sm = rec["summary"]
    if mode == "abstractive":
        rec["summary"] = {"engine": "abstractive", "text": DEMO_ABSTRACTIVE_TEXT,
                          "model": "summarizer_mt5",
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
        marquee_colors = ("linear-gradient(90deg,#5865f2,#ec48bd)",
                          "rgba(255,255,255,.9)")
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
        marquee_colors = ("linear-gradient(90deg,#5865f2,#ec48bd)",
                          "rgba(255,255,255,.9)")
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
      /* ---- marquee band (Blurple, Discord-style) ---- */
      .rcn-marquee {{
          background:linear-gradient(90deg,#5865f2,#ec48bd);
          border-radius:40px; padding:14px 0; overflow:hidden;
          white-space:nowrap; position:relative; }}
      .rcn-marquee span {{
          display:inline-block; padding-left:100%;
          animation:rcn-scroll 22s linear infinite;
          font-family:'Space Grotesk',sans-serif; font-weight:800;
          font-size:20px; color:#fff; letter-spacing:.04em; }}
      .rcn-marquee:hover span {{ animation-play-state:paused; }}
      @keyframes rcn-scroll {{ 0% {{ transform:translateX(0); }}
          100% {{ transform:translateX(-100%); }} }}
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
    """Markdown download: dung renderer that cua core khi co, fallback nhe
    khi chua co core (che do du lieu mau van tai file ve duoc)."""
    try:
        from docproc.nlp.report import render_markdown

        return render_markdown(rec)
    except Exception:
        lines = [f"# Báo cáo hiểu tài liệu", "",
                 f"- Nguồn: `{rec.get('source', '')}`"]
        dt = rec.get("doc_type") or {}
        if dt.get("label"):
            name, _, _ = label_meta(dt["label"])
            conf = dt.get("confidence")
            lines.append(f"- Loại tài liệu: **{name}**"
                         + (f" (độ tin cậy {conf:.0%})" if conf else ""))
        totals = (rec.get("structure") or {}).get("stats") or {}
        if totals:
            lines.append(f"- Thống kê: {totals.get('words', 0)} từ · "
                         f"{totals.get('sentences', 0)} câu · "
                         f"{totals.get('paragraphs', 0)} đoạn")
        kws = rec.get("keywords") or []
        if kws:
            terms = [k["term"] if isinstance(k, dict) else str(k) for k in kws]
            lines += ["", "## Từ khóa", ", ".join(terms)]
        fd = (rec.get("fields") or {}).get("fields") or {}
        if fd:
            lines += ["", "## Trường dữ liệu", ""]
            lines += [f"- **{k}**: {v}" for k, v in fd.items()]
        sm = rec.get("summary") or {}
        body = sm.get("text") or " ".join(
            s["text"] for s in (sm.get("sentences") or []))
        if body:
            lines += ["", "## Tóm tắt", "", body]
        return "\n".join(lines)


def label_meta(label):
    name, icon, color = CLASS_META.get(label, (label or "Không xác định",
                                               "❓", "#888888"))
    return name, icon, color


def type_card(doc_type):
    lbl = doc_type.get("label")
    conf = doc_type.get("confidence")
    via = doc_type.get("via") or ""
    name, icon, color = label_meta(lbl)
    sub = f"Độ tin cậy {conf:.0%}" if conf is not None else \
          {"rule_cues": "Xác định theo quy tắc nội dung",
           "low_confidence": "Độ tin cậy thấp — không đoán mò"}.get(
              via, "Chưa xác định được loại tài liệu")
    name, sub = html.escape(str(name)), html.escape(str(sub))
    st.markdown(f"""
    <div style='background:{color};color:#fff;border-radius:12px;padding:18px 22px;
                display:flex;align-items:center;gap:14px'>
      <span style='font-size:34px'>{icon}</span>
      <div><div style='font-size:19px;font-weight:700'>{name}</div>
      <div style='opacity:.85;font-size:13px'>{sub}</div></div>
    </div>""", unsafe_allow_html=True)


def kw_cloud(keywords):
    """keywords: [{'term','score','count'}, ...] đúng hợp đồng của seam."""
    chips = []
    n = max(len(keywords), 1)
    for i, item in enumerate(keywords):
        term = item["term"] if isinstance(item, dict) else str(item)
        size = 17 - 7 * i // max(n - 1, 1)
        chips.append(f"<span class='rcn-chip' style='font-size:{size}px'>"
                     f"{html.escape(term)}</span>")
    st.markdown("".join(chips), unsafe_allow_html=True)
    st.caption(f"Top {len(keywords)} cụm từ đặc trưng nhất của tài liệu.")


def kw_chart(keywords):
    """Bar chart ngang: độ nổi bật từng cụm từ khóa (Altair, màu theo theme)."""
    import altair as alt
    import pandas as pd

    rows = []
    for item in keywords:
        if not isinstance(item, dict):
            continue
        rows.append({"Từ khóa": str(item.get("term", "?")),
                     "Độ nổi bật": float(item.get("score", 0) or 0),
                     "Số lần": int(item.get("count", 0) or 0)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    dark = st.session_state.get("dark", True)
    chart = (alt.Chart(df)
             .mark_bar(cornerRadius=6)
             .encode(
                 x=alt.X("Độ nổi bật:Q", title="Độ nổi bật",
                         scale=alt.Scale(nice=True)),
                 y=alt.Y("Từ khóa:N", sort="-x", title=None),
                 tooltip=["Từ khóa", "Độ nổi bật", "Số lần"],
                 color=alt.value(BRAND))
             .properties(height=max(160, 26 * len(df))))
    st.altair_chart(chart, width='stretch')
    st.caption("Độ nổi bật = tần suất (tf) × độ hiếm theo đoạn (idf) — "
               "xuất hiện nhiều + tập trung cục bộ thì càng đặc trưng.")


def topic_chart(topics, mixture):
    """Donut chart: tỷ trọng từng nhóm chủ đề trong tài liệu."""
    import altair as alt
    import pandas as pd

    rows = []
    for i, t in enumerate(topics):
        share = mixture[i] if i < len(mixture) else None
        if share is None:
            continue
        rows.append({"Chủ đề": t.get("label") or f"Chủ đề {t.get('id', i) + 1}",
                     "Tỷ trọng": round(float(share) * 100, 1),
                     "Từ tiêu biểu": ", ".join(t.get("top_words", [])[:4])})
    if not rows:
        return
    df = pd.DataFrame(rows)
    dark = st.session_state.get("dark", True)
    scheme = "blues" if dark else "tealblues"
    chart = (alt.Chart(df)
             .mark_arc(innerRadius=48, outerRadius=95)
             .encode(
                 theta=alt.Theta("Tỷ trọng:Q", stack=True),
                 color=alt.Color("Chủ đề:N", scale=alt.Scale(scheme=scheme),
                                 legend=None),
                 tooltip=["Chủ đề", "Tỷ trọng", "Từ tiêu biểu"])
             .properties(height=210))
    st.altair_chart(chart, width='stretch')
    st.caption("Tỷ trọng = tỷ lệ nội dung tài liệu thuộc về nhóm chủ đề đó.")


def topic_bars(topics, mixture):
    for i, t in enumerate(topics):
        head = t.get("label") or f"Chủ đề {t.get('id', i) + 1}"
        share = mixture[i] if i < len(mixture) else None
        pct = int(round((share or 0) * 100))
        st.markdown(f"**{html.escape(str(head))}**")
        track = SURFACE_ONYX if st.session_state.get("dark") else "#e3e6f7"
        st.markdown(
            f"<div style='background:{track};border-radius:8px;height:14px'>"
            f"<div style='background:{BRAND};width:{pct}%;height:14px;"
            f"border-radius:8px'></div></div>", unsafe_allow_html=True)
        left, right = st.columns([1, 4])
        left.caption(f"~{pct}% tài liệu")
        right.write(", ".join(t.get("top_words", [])[:8]))
        st.divider()
    st.caption("Số nhóm chủ đề được chọn tự động sao cho các nhóm tách bạch nhất.")


# Các thuật toán ML/DL được dùng trong từng bước phân tích (hiển thị UI).
ALGOS = [
    ("SVM", "e0b_svm", "Phân loại văn bản",
     "TF-IDF → LinearSVC; GridSearchCV chọn siêu tham số 5-fold.",
     "text_model_svm.joblib"),
    ("CNN", "e1_cnn", "Phân loại ảnh scan",
     "Conv2D→MaxPool→Dense(softmax), train trên RVL-CDIP.",
     "runs/E1/best.keras"),
    ("TF-IDF", "tfidf", "Độ nổi bật từ khóa",
     "score = tf × idf (idf theo đoạn) — từ khóa & vector hóa.",
     "text_vectorizer.joblib"),
    ("LDA", "lda", "Nhóm chủ đề",
     "Latent Dirichlet Allocation; k chọn bằng UMass coherence.",
     "sklearn.decomposition"),
    ("K-means + PCA", "km_pca", "Nhãn chủ đề",
     "Gom cụm từ khóa → 1 từ đại diện cho mỗi nhóm chủ đề.",
     "sklearn.cluster"),
    ("MMR", "mmr", "Tóm tắt trích xuất",
     "score = λ·relevance − (1−λ)·redundancy; chọn câu không trùng.",
     "docproc.nlp.summary"),
    ("Seq2Seq T5", "t5", "Tóm tắt sinh",
     "Transformer vit5-base fine-tune trên XLSum-VI (offline).",
     "vit5_finetuned"),
]


def algo_panel():
    """Lưới card thuật toán ML/DL của project (đúng dữ liệu core thật)."""
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


def demo_cta(help_text="Chưa có tài liệu? Xem ngay bằng dữ liệu mẫu:",
             cta_key: str | None = None):
    """Nút CTA dùng chung: bật bản demo không cần core."""
    st.caption(help_text)
    key = cta_key or f"cta_demo_{abs(hash(help_text)) % 100000}"
    if st.button("🧪 Xem với dữ liệu mẫu", key=key):
        st.session_state["rec"] = demo_record(None, 3)
        st.rerun()


def highlight_text(text: str, keywords: list, dark: bool) -> str:
    """Bôi nền các keyphrase (uni+bi-gram, không phân biệt hoa/thường)."""
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
        return True   # mặc định dark


if "dark" not in st.session_state:
    st.session_state["dark"] = _load_theme()
inject_css(st.session_state["dark"])

with st.sidebar:
    st.header("📄 RCN Studio")
    st.caption("Mọi tài liệu, một cái nhìn rõ ràng — chạy hoàn toàn trên máy.")
    st.divider()

    dark_new = st.toggle("🌙 Giao diện tối",
                         value=st.session_state["dark"],
                         key="theme_toggle",
                         help="Đổi sáng/tối (ghi nhớ cả khi khởi động lại)")
    if dark_new != st.session_state["dark"]:
        st.session_state["dark"] = dark_new
        try:
            _THEME_FILE.write_text("dark" if dark_new else "light")
        except Exception:
            pass
        st.rerun()

    src = st.radio("Nguồn dữ liệu",
                   ["🧪 Dữ liệu mẫu (xem thử giao diện)",
                    "📄 Tài liệu của bạn",
                    "📁 Thư mục (batch)"])
    use_demo = src.startswith("🧪")
    use_batch = src.startswith("📁")

    up, pasted = None, ""
    folder_path = ""
    if use_batch:
        folder_path = st.text_input(
            "Thư mục cần quét",
            value=str(ROOT / "datasets" / "text"),
            help="Mọi tệp văn bản trong thư mục (kể cả thư mục con) "
                 "được quét và gộp thành bảng + CSV.")
    elif not use_demo:
        up = st.file_uploader("Kéo-thả tài liệu vào đây", type=SUPPORTED)
        st.caption("TXT · MD · HTML · DOCX · PDF (kể cả scan) · ảnh")
        pasted = st.text_area("...hoặc dán văn bản:", height=130,
                              placeholder="Dán nội dung cần phân tích vào đây")

    st.markdown("---")
    st.markdown("### ⚙️ Tuỳ chọn tóm tắt")
    mode_lbl = st.segmented_control(
        "Chế độ", ["Tự động", "Trích xuất câu", "Sinh đoạn văn mới"],
        default="Tự động",
        help="“Sinh đoạn văn mới” cần model nhỏ cài sẵn trong máy; "
             "thiếu thì tự quay về trích xuất câu.")
    MODE_MAP = {"Tự động": None, "Trích xuất câu": "extractive",
                "Sinh đoạn văn mới": "abstractive"}
    k_sum = st.slider("Số câu tóm tắt", 1, 7, 3)

    can_go = (use_demo
              or use_batch
              or (not use_demo and not use_batch
                  and (up is not None or bool(pasted.strip()))))
    go = st.button("▶️  Phân tích", type="primary", width="stretch",
                   key="analyze", disabled=not can_go,
                   shortcut="Ctrl+Enter",
                   help="Chưa đủ dữ liệu? Chọn file/dán văn bản/nhập thư mục "
                        "trước. Ctrl+Enter để chạy nhanh.")


def skeleton_block(rows: int = 3, width_pct: tuple = (72, 95, 60)):
    """Skeleton shimmer thay spinner — cảm giác tải nhanh hơn."""
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
        st.session_state["rec"] = demo_record(MODE_MAP[mode_lbl], k_sum)
        st.toast("Đang hiển thị dữ liệu mẫu 🧪", icon="🧪")
    elif use_batch:
        fdir = Path(folder_path)
        if not fdir.is_dir():
            st.sidebar.warning("Thư mục không tồn tại — kiểm tra lại đường dẫn.")
        files = sorted(p for p in fdir.rglob("*")
                       if p.is_file()
                       and not p.name.endswith((".understanding.json",
                                                ".understanding.md"))
                       and p.suffix.lower() in
                       {".txt", ".md", ".markdown", ".html", ".htm",
                        ".docx", ".pdf"})
        if not files:
            st.sidebar.warning("Không có tệp nào loại hỗ trợ trong thư mục.")
        else:
            from docproc.nlp.report import understand_file

            rows, errs = [], 0
            prog = st.progress(0.0, text="Đang quét thư mục...")
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
                    "Tệp": fp.name,
                    "Loại": label_meta(doc_lbl)[0] if doc_lbl else "— lỗi —",
                    "Từ": ((r.get("structure") or {}).get("stats", {})
                           .get("words", 0)) if doc_lbl else 0,
                    "Câu": ((r.get("structure") or {}).get("stats", {})
                            .get("sentences", 0)) if doc_lbl else 0,
                    "Từ khóa": len(r.get("keywords") or []) if doc_lbl else 0,
                    "Trường khớp": (r.get("fields") or {}).get(
                        "matched", 0) if doc_lbl else 0,
                    "Ghi chú": err_txt,
                })
                prog.progress((i + 1) / len(files),
                              text=f"{i + 1}/{len(files)}: {fp.name}")
            st.session_state["batch_rows"] = rows
            st.session_state["batch_errs"] = errs
            st.toast(f"Xong {len(files)} tệp"
                     + (f" ({errs} lỗi)" if errs else ""), icon="📁")
    elif up is None and not pasted.strip():
        st.sidebar.warning("Hãy tải tệp hoặc dán văn bản trước.")
    else:
        # Skeleton hiện NGAY trước khi seam chạy (abstractive có thể mất 5-25s)
        st.markdown("**Đang phân tích tài liệu...**")
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
                # seam cần đường dẫn thật trên đĩa -> đổ upload ra file tạm
                # (giữ hậu tố để magic-bytes + extension nhận đúng loại)
                suffix = Path(up.name or "upload.bin").suffix or ".bin"
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix) as tmp:
                    tmp.write(up.getvalue())
                    tmp_path = Path(tmp.name)
                rec = understand_file(tmp_path,
                                      summary_mode=MODE_MAP[mode_lbl],
                                      summary_k=k_sum)
                rec["source"] = up.name  # tên thật thay đường dẫn tạm
            else:
                rec = understand(pasted, source="(văn bản dán)",
                                 summary_mode=MODE_MAP[mode_lbl],
                                 summary_k=k_sum)
            st.session_state["rec"] = rec
            # Lịch sử phiên: giữ tối đa 8 lần phân tích gần nhất
            hist = st.session_state.setdefault("history", [])
            hist.insert(0, {
                "source": str(rec.get("source", "")),
                "time": datetime.now().strftime("%H:%M:%S"),
                "label": (rec.get("doc_type") or {}).get("label") or "?",
                "rec": rec,
            })
            del hist[8:]
        except Exception as exc:
            st.session_state.pop("rec", None)
            st.error("Không xử lý được tài liệu này.")
            st.code(f"{type(exc).__name__}: {exc}", language=None)
            st.info("💡 Có thể xem giao diện trước bằng **Dữ liệu mẫu** "
                    "trên thanh bên.")
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

rec = st.session_state.get("rec")
is_demo = bool(rec) and str(rec.get("source", "")).startswith("(dữ liệu mẫu)")

# ------------------------------------------------------------ batch result --
batch_rows = st.session_state.get("batch_rows")
if batch_rows:
    st.markdown(f"""<div class='rcn-hero'><h1>Kết quả quét thư mục</h1>
      <p>{len(batch_rows)} tệp · {st.session_state.get('batch_errs', 0)} lỗi
      — bấm “Phân tích” để quét lại.</p></div>""", unsafe_allow_html=True)
    st.write("")
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(batch_rows[0].keys()))
    writer.writeheader()
    writer.writerows(batch_rows)
    ctbl, ccsv = st.columns([3, 1])
    with ctbl:
        st.dataframe(batch_rows, hide_index=True, width='stretch')
    with ccsv:
        st.download_button("Tải CSV", buf.getvalue().encode("utf-8-sig"),
                           "batch_understanding.csv", "text/csv",
                           width='stretch',
                           type="primary", icon=":material/table:",
                           help="Bảng kết quả quét thư mục (CSV, Excel-compatible)")
        if st.button("Xóa kết quả batch", width="stretch",
                     type="secondary", icon=":material/delete:"):
            for k in ("batch_rows", "batch_errs"):
                st.session_state.pop(k, None)
            st.rerun()
    st.stop()

# ------------------------------------------------------------------ result --
history = st.session_state.get("history") or []
if history and not is_demo:
    with st.expander(f"🕘 Lịch sử phiên ({len(history)} lần phân tích)"):
        for i, h in enumerate(history):
            c1, c2, c3, c4 = st.columns([4, 2, 3, 1])
            c1.write(f"**{html.escape(Path(h['source']).name or h['source'])}**")
            c2.caption(h["time"])
            name_h, _, _ = label_meta(h.get("label"))
            c3.caption(name_h)
            if c4.button("Xem", key=f"hist_{i}", type="secondary",
                         icon=":material/visibility:",
                         help="Mở lại kết quả của lần phân tích này"):
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
            "🧪 <b>Đang xem dữ liệu mẫu</b> — giao diện dùng bản ghi đóng sẵn, "
            "chưa gọi bộ máy phân tích.</div>",
            unsafe_allow_html=True)
        st.write("")
        hero_sub = ""
    else:
        hero_sub = ("Mọi số liệu sinh trên chính máy này — cùng bộ máy với "
                    "dòng lệnh, kết quả JSON/Markdown giống hệt.")
    st.markdown(f"""<div class='rcn-hero'><h1>Kết quả phân tích</h1>
      <p><b>Mọi tài liệu, một cái nhìn rõ ràng.</b>{' ' + hero_sub if hero_sub else ''}</p></div>""",
                unsafe_allow_html=True)
    st.write("")
    d1, d2 = st.columns([1, 1])
    d1.download_button("Tải JSON", js_bytes, f"{name}.understanding.json",
                       "application/json", width="stretch",
                       type="primary", icon=":material/data_object:",
                       help="Toàn bộ bản ghi phân tích (định dạng JSON)")
    d2.download_button("Tải Markdown", md_txt.encode("utf-8"),
                       f"{name}.understanding.md", "text/markdown",
                       width="stretch", type="secondary",
                       icon=":material/article:",
                       help="Báo cáo dạng Markdown, giống hệt dòng lệnh")
    st.write("")

    if rec.get("note"):  # ảnh / PDF scan -> chỉ phân loại
        st.info("Đây là ảnh/trang scan: hệ thống chỉ xác định loại tài liệu, "
                "không đọc chữ (theo phạm vi dự án).")
        type_card(rec.get("doc_type", {}))
        st.stop()

    tab_sm, tab_ov, tab_kw, tab_tx, tab_tp, tab_fd, tab_al = st.tabs(
        ["📝 Tóm tắt", "📊 Tổng quan",
         f"🔑 Từ khóa ({len(rec.get('keywords') or [])})",
         "📃 Văn bản gốc", "🧩 Chủ đề",
         f"🗂 Trường dữ liệu ({len((rec.get('fields') or {}).get('fields') or {})})",
         "🧠 Thuật toán"],
        default="📝 Tóm tắt")

    with tab_ov:
        c1, c2 = st.columns([1, 1])
        with c1:
            type_card(rec.get("doc_type", {}))
        with c2:
            totals = (rec.get("structure") or {}).get("stats", {})
            m1, m2, m3 = st.columns(3)
            m1.metric("Số từ", f"{totals.get('words', 0):,}")
            m2.metric("Số câu", totals.get("sentences", 0))
            m3.metric("Số đoạn", totals.get("paragraphs", 0))
        st.caption(f"Nguồn: `{rec.get('source')}`"
                   + (f" · loại tệp {str(rec.get('file_type')).upper()}"
                      if rec.get("file_type") else ""))

        # ---- hàng metrics phụ: mật độ đọc + trường khớp ----
        words = totals.get("words", 0)
        sents = totals.get("sentences", 0)
        uniq = totals.get("unique_words", 0)
        m_a, m_b, m_c, m_d = st.columns(4)
        m_a.metric("Từ độc nhất", f"{uniq:,}"
                   + (f" ({uniq / words:.0%})" if words else ""))
        m_b.metric("Từ / câu",
                   f"{words / sents:.1f}" if sents else "—")
        read_min = words / 200  # ~200 từ/phút đọc trung bình
        m_c.metric("Thời gian đọc",
                   f"{read_min:.1f} phút" if read_min >= 1
                   else f"{read_min * 60:.0f} giây")
        fd_ov = rec.get("fields") or {}
        flds = fd_ov.get("fields") or {}
        m_d.metric("Trường khớp",
                   f"{fd_ov.get('matched', 0)}/{len(flds)}"
                   if flds else "—")

        # ---- phân bố độ dài đoạn ----
        paras = (rec.get("structure") or {}).get("paragraphs") or []
        if len(paras) >= 2:
            st.write("")
            st.markdown("**Độ dài từng đoạn (số từ):**")
            st.bar_chart(
                {f"¶{p.get('index', i) + 1}": p.get("word_count", 0)
                 for i, p in enumerate(paras)},
                height=220, x_label="đoạn", y_label="số từ")

    with tab_kw:
        kws = rec.get("keywords") or []
        if kws:
            kw_cloud(kws)
            st.write("")
            kw_chart(kws)
        else:
            st.markdown("### 🔑")
            st.write("**Yên lặng quá.** Tài liệu này không có cụm từ nào "
                     "nổi bật cả.")
            demo_cta("Tài liệu dài hơn sẽ cho từ khóa rõ vẽ hơn.")

    with tab_tx:
        # Văn bản gốc dựng lại từ L1 (đúng thứ tự câu/đoạn seam đã phân tích)
        paras = ((rec.get("structure") or {}).get("paragraphs") or [])
        if paras:
            full_text = "\n\n".join(
                " ".join(p.get("sentences", [])) for p in paras)
            st.markdown(
                f"<div style='font-size:15.5px;line-height:1.9'>"
                f"{highlight_text(full_text, rec.get('keywords') or [], st.session_state.get('dark', True))}"
                f"</div>", unsafe_allow_html=True)
        else:
            st.info("Không có văn bản gốc để hiển thị (tài liệu chỉ phân loại).")

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
                    st.markdown("**Độ gắn kết nhóm chủ đề theo số nhóm (k):**")
                    st.line_chart(
                        data={c["k"]: c["umass"] for c in curve},
                        x_label="số nhóm (k)", y_label="độ gắn kết")
                    st.caption(f"Điểm cao nhất k={tp.get('k')} "
                               f"({tp.get('selected_by')}) → nhóm được chọn.")
                with c_formula:
                    f_bg = SURFACE_INDIGO if st.session_state.get(
                        "dark", True) else "#f5f7ff"
                    st.markdown(
                        "<div style='background:%s;"
                        "border:1px solid rgba(88,101,242,.40);border-radius:16px;"
                        "padding:12px 14px;font-size:12.5px;line-height:1.7'>"
                        "<b>Độ gắn kết UMass</b> — các từ trong cùng một nhóm "
                        "có thường xuất hiện chung trong cùng một đoạn không.<br><br>"
                        "<code>log( D(wₜ & wᵢ) / D(wₜ) )</code><br>"
                        "<span style='opacity:.75'>D(w) = số đoạn chứa từ w; "
                        "wₜ, wᵢ = 2 từ đứng đầu nhóm.</span><br><br>"
                        "Càng gần 0 càng gắn kết → chọn k cao nhất.</div>"
                        % f_bg,
                        unsafe_allow_html=True)
        else:
            st.markdown("### 🧩")
            st.write("**Chưa đủ chất liệu.** Văn bản ngắn quá nên chưa tách "
                     "ra được nhóm chủ đề.")
            demo_cta("Cần tối thiểu 3 đoạn có nội dung — hoặc xem thử:",
                     cta_key="cta_topics")

    with tab_fd:
        fd = rec.get("fields") or {}
        fields = fd.get("fields") or {}
        if fields:
            st.dataframe(
                [{"Trường": k,
                  "Giá trị": ", ".join(map(str, v)) if isinstance(v, list)
                  else str(v)}
                 for k, v in fields.items()],
                hide_index=True, width="stretch")
            ok, tot = fd.get("matched", 0), len(fields)
            if ok == tot and tot:
                st.success(f"Khớp đầy đủ {ok}/{tot} trường theo loại "
                           f"\"{label_meta(fd.get('doc_type'))[0]}\".")
            else:
                st.caption(f"Khớp {ok}/{tot} trường theo loại "
                           f"\"{label_meta(fd.get('doc_type'))[0]}\".")
            for miss in fd.get("missing_required") or []:
                st.warning(f"⚠️ Thiếu trường bắt buộc: **{miss}**")
        else:
            st.markdown("### 🗂")
            st.write("**Không bóc được trường nào.** Tài liệu này không mang "
                     "dấu hiệu số hiệu, ngày hay tiền tệ.")
            demo_cta("Hóa đơn/biên lai sẽ cho kết quả tốt nhất — hoặc thử:")

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
            body = sm.get("text") or "(model không sinh ra nội dung)"
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
            with st.expander("🔎 Xem từng câu được chọn"):
                for s in sentences:
                    st.markdown(f"> ¶{s.get('paragraph', '?')} — "
                                f"{html.escape(s['text'])}")
            comp = sm.get("compression") or {}
            orig, kept = comp.get("original_sentences"), comp.get("kept")
            foot = (f"Nén: giữ {kept}/{orig} câu."
                    if kept is not None else f"Từ {orig} câu gốc.")
            st.caption(foot)
        else:
            st.markdown("### 📝")
            st.write("**Không có gì để tóm.** Nội dung chưa đủ dài để rút "
                     "ra câu quan trọng.")
            demo_cta("Dán thêm nội dung, hoặc xem thử:")
        if sm.get("engine_fallback"):
            st.warning("⚠️ Model sinh đoạn chưa cài — đã tự chuyển sang "
                       "trích xuất câu, kết quả vẫn đầy đủ.")

    with tab_al:
        st.markdown("**Các thuật toán ML/DL đang chạy trong project:**")
        algo_panel()
        st.write("")
        st.caption("SVM + CNN: phân loại tài liệu · TF-IDF/LDA/K-means: hiểu "
                   "nội dung · MMR/T5: tóm tắt — toàn bộ chạy offline trên máy.")

    with st.expander("🔧 Chi tiết kỹ thuật (cho demo/bảo vệ)"):
        st.json(rec, expanded=False)
else:
    st.markdown("""<div class='rcn-hero'><h1>RCN STUDIO</h1>
      <p><b>Mọi tài liệu, một cái nhìn rõ ràng.</b> Nạp tài liệu hoặc dán văn
      bản ở thanh bên rồi bấm “Phân tích”. Chưa sẵn sàng? Chọn “Dữ liệu mẫu”
      để xem thử toàn bộ giao diện.</p>
      </div>""", unsafe_allow_html=True)
    st.write("")
    st.markdown("<div class='rcn-marquee'><span>"
                "PHÂN LOẠI &nbsp;·&nbsp; TỪ KHÓA &nbsp;·&nbsp; CHỦ ĐỀ "
                "&nbsp;·&nbsp; TRƯỜNG DỮ LIỆU &nbsp;·&nbsp; TÓM TẮT "
                "&nbsp;·&nbsp; 100% OFFLINE &nbsp;·&nbsp; "
                "PHÂN LOẠI &nbsp;·&nbsp; TỪ KHÓA &nbsp;·&nbsp; CHỦ ĐỀ "
                "&nbsp;·&nbsp; TRƯỜNG DỮ LIỆU &nbsp;·&nbsp; TÓM TẮT "
                "&nbsp;·&nbsp; 100% OFFLINE &nbsp;·&nbsp; "
                "</span></div>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    feats = [
        ("📊", "Hiểu cấu trúc",
         "Đếm từ – câu – đoạn, biết ngay tài liệu dày hay mỏng."),
        ("🔑", "Từ khóa & chủ đề",
         "Cụm từ đặc trưng + nhóm chủ đề chiếm tỷ trọng, có nhãn dễ đọc."),
        ("🗂", "Bóc trường dữ liệu",
         "Số hiệu, ngày, tiền, bên mua/bán... thành bảng gọn gàng."),
    ]
    for col, (icon, title, desc) in zip((c1, c2, c3), feats):
        with col:
            st.markdown(
                f"<div class='rcn-feature'><span class='rcn-ico'>{icon}</span>"
                f"<h3>{title}</h3><p>{desc}</p></div>",
                unsafe_allow_html=True)
    st.write("")
    st.info("💡 Mẹo: chọn **🧪 Dữ liệu mẫu** ở thanh bên để xem thử giao diện "
            "mà không cần chuẩn bị gì.", icon="👈")
    st.write("")
    st.markdown("#### 🧠 Thuật toán ML/DL")
    algo_panel()

st.markdown("---")
st.caption("RCN Studio · Mọi tài liệu, một cái nhìn rõ ràng · đầu ra "
           "JSON/Markdown giống hệt dòng lệnh.")
