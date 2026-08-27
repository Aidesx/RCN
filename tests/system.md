# RCN — Test Suite Map

> Tổ chức tests theo module tương ứng với `src/docproc/`. Mỗi thư mục con = 1 thành phần hệ thống.

## Cấu trúc

```
tests/
├── system.md                   ← file này
├── fixtures/                   ← dữ liệu mẫu dùng chung (PDF, DOCX, ảnh...)
├── golden/                     ← mảng numpy đóng băng (preprocessing)
│
├── io/                         ← ingestion: detect file type + parse text + render pages
│   ├── test_detect.py          magic bytes + probe PDF scan → phân loại
│   ├── test_parsers.py         trích text từ PDF/DOCX/MD/HTML (golden-tested)
│   └── test_render.py          PDF → ảnh + embedded images
│
├── preprocess/                 ← chuẩn bị input cho model
│   ├── test_image.py           resize bicubic → tensor (golden: khớp numpy từng pixel)
│   └── test_text.py            TF-IDF vectorizer wrapper (save/load joblib)
│
├── models/                     ← định nghĩa kiến trúc model
│   └── test_cnn.py             Architecture A: Conv2D→MaxPool→Dense→Softmax
│
├── training/                   ← hạ tầng huấn luyện
│   ├── test_data.py            registry dataset 2-arm (64×64 / 224×224)
│   └── test_harness.py         seeded fit + snapshot config + EarlyStopping
│
├── evaluation/                 ← đo lường
│   ├── test_metrics.py         accuracy, macro-F1, confusion matrix, acceptance gate
│   └── test_eval_report.py     frozen-test report + learning curves
│
├── nlp/                        ← trái tim "hiểu tài liệu" (L1–L5)
│   ├── test_structure.py       L1: word → sentence → paragraph + stats
│   ├── test_keywords.py        L2: top-k keyphrases (in-doc TF-IDF, uni+bigram)
│   ├── test_topics.py          L3: LDA + UMass coherence k-selection
│   ├── test_fields.py          L4: regex schema per class (invoice/receipt...)
│   ├── test_summary.py         L5: extractive MMR + abstractive seq2seq + fallback
│   └── test_report.py          seam understand() + render_markdown()
│
├── text_classifier/            ← router phân loại văn bản
│   └── test_baseline.py        GridSearchCV SVM/RF + artifact dump
│
├── dataset/                    ← dữ liệu
│   └── test_module.py          manifest + split 70/15/15 + leak check
│
└── ui/                         ← kiểm thử giao diện (Streamlit AppTest)
    ├── test_smoke.py           render không crash + demo mode + dán text thật
    ├── test_features.py        highlight, coherence chart, lịch sử, batch+CSV
    └── test_e2e_core.py        E2E: 4 luồng core thật (extractive/abstractive/upload/batch)
```

## Chạy

```bash
# Từ thư mục gốc RCN/
python -m pytest -q              # toàn bộ ~186 test
python -m pytest tests/io/ -q    # chỉ ingestion
python -m pytest tests/ui/ -q    # chỉ UI smoke/E2E
```

## Quy ước

- **Deterministic**: mọi test có seed 42, cùng input → cùng output
- **Golden-tested**: file I/O đối chiếu với `.expected.txt`; tensor ảnh khớp `.npy`
- **UI tests**: dùng `streamlit.testing.v1.AppTest`, không cần browser thật
- **Fallback-safe**: test abstractive tự skip nếu thiếu checkpoint