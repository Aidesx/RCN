# RCN — Narrow AI Document Processing Project

> Document Understanding without OCR

## Mục tiêu dự án (Project Objective)

Xây dựng một CLI application **hiểu tài liệu** nhận text / file / folder và:

1. **Hiểu nội dung theo tầng** (mục tiêu chính):
   - **L1 Cấu trúc** ✅ — word → sentence → paragraph kèm thống kê (`docproc.nlp.analyze_structure`)
   - **L2 Từ khóa** ✅ — top-k keyphrases bằng TF-IDF trong văn bản (`nlp/keywords.py`, uni+bigram, lọc stopwords song ngữ)
   - **L3 Chủ đề** ✅ — LDA (seed 42), chọn k bằng UMass coherence (`nlp/topics.py`)
   - **L4 Trường dữ liệu** ✅ — regex schemas theo loại tài liệu: số HĐ, ngày, tổng tiền, bên mua/bán… (`nlp/fields.py`)
2. **Phân loại loại tài liệu** (router): 6 nhóm `invoice/receipt/report/letter/form/article` — CNN cho ảnh/scan, TF-IDF+SVM cho text; nhãn phụ trợ, thiếu artifact → `unavailable` không fail.
3. **Trích xuất nội dung** từ PDF-text/DOCX/Markdown/HTML bằng parser deterministic.
4. **Xuất báo cáo hiểu tài liệu**: `understand()` → JSON + Markdown + terminal (`nlp/report.py` + CLI `scripts/understand_text.py`).

**Ranh giới phạm vi (không đổi):** không OCR; không LLM/API ngoài — ưu tiên kiến thức khóa học; scanned page chỉ được *phân loại*, không đọc chữ; MobileNetV2/fusion chuyển **FUTURE**. Kế hoạch chi tiết: `docs/design/01..06` (bộ v2, 2026-08-23).

## Input / Output

| Input | Cách xử lý |
|---|---|
| PDF (text layer) | Trích text deterministic (text branch) |
| PDF (scanned) | Render trang → image branch (phân loại, **không OCR**) |
| DOCX / Markdown / HTML | Parse → trích text |
| PNG / JPG | Image branch trực tiếp |

| Output | Định dạng |
|---|---|
| Structured data | JSON (category, confidence, per-page, extracted text) |
| Report | Markdown |
| Summary | Terminal text |

## Hiện trạng (Status)

- [x] Design phase hoàn tất (spec chi tiết: `01-knowledge-audit` → `06-project-specification`, xem `docs/design/`)
- [x] Repository scaffold + môi trường Python (`.venv`, TF 2.21, scikit-learn, PyMuPDF, pytest...) + README/ARCHITECTURE
- [x] Dataset v0: **700 trang** — 500 RVL-CDIP (100/class × letter, form, report, article, invoice) + 200 SROIE receipts — nguồn ngoài, `EXTENSION`, provenance từng ảnh tại `datasets/raw/PROVENANCE.csv` (cột `source`: rvlcdip/sroie); split manifest 70/15/15 theo document đã build (0 leak, đủ 6 class); **spot-check nhãn đã duyệt bởi user (2026-08-19)**
- [x] Stage 3 preprocessing: `src/docproc/preprocess/` (image 64×64/224×224 deterministic + TF-IDF wrapper) — golden tests, **18/18 pass**
- [x] **Stages 4–6: Self-built CNN (Architecture A) train + evaluate** — E1 best val_accuracy 60% (epoch 27/30); frozen test: accuracy **57.1%**, macro-F1 **0.514**, majority baseline 28.6% → **gate PASS** (+28.6 pts, ≥0.50 F1). Artifacts: `runs/E1/` (weights `best.keras`, history, metrics, confusion matrix, learning curves)
- [x] Stage 7 Document I/O: `src/docproc/io/` — file detection (magic bytes + scanned-PDF probe), parsers (PDF/DOCX/MD/HTML), renderer + embedded-image extraction, structured error contracts — golden tests
- [x] Architecture deepening: `docproc/paths` (một nguồn duy nhất cho layout/config), dataset module với registry 2 arms (cnn 64×64 / finetune 224×224), `evaluation.report` dùng chung cho mọi experiment — manifest rebuild **byte-identical**, E1 tái lập chính xác
- [x] Stage 8 Text baseline (E0b): corpus tổng hợp 360 docs (`datasets/text/`) + TF-IDF+SVM/RF GridSearchCV — test acc **1.000** / macro-F1 **1.000** vs baseline 16.7% → gate PASS; artifacts `models/artifacts/text_*` + `runs/E0b/`
- [x] **Understanding pipeline v2 (L1+L2+L3+L4 + router + report + CLI)** — keyphrases TF-IDF, LDA+UMass, `understand()` seam, CLI file/folder/demo — test suite **167/167 pass**
- [ ] FUTURE: MobileNetV2 fine-tune (E3/E4) · fusion E5 · QA/tóm tắt

## Cấu trúc

```
src/docproc/{io, preprocess, models, training, evaluation, nlp}
configs/            # mọi hyperparameter + fields.yaml (schema override)
scripts/            # CLI chính (understand_text.py) + tooling dataset/train/experiments
docs/               # design set snapshot (01..06) + manifest-schema.md
datasets/           # raw/ text/ splits/ (gitignored)
models/artifacts/   # router artifacts: vectorizer, SVM, CNN checkpoint refs (gitignored)
runs/               # per-experiment logs + metrics (gitignored)
requirements.txt    # pin đúng phiên bản venv đã kiểm chứng
```

*Kiểm chứng chất lượng: **167/167 testcases pass** (`python -m pytest -q`). Cài đặt môi trường: `pip install -r requirements.txt`.*

*File này sẽ được cập nhật dần theo tiến độ implementation.*