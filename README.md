# RCN — Narrow AI Document Processing Project

> Document Understanding without OCR

## Mục tiêu dự án (Project Objective)

Xây dựng một CLI application nhận file/folder tài liệu và:

1. **Phân loại trang tài liệu** vào 6 nhóm: `invoice`, `receipt`, `report`, `letter`, `form`, `article` bằng **CNN tự xây** (kiến trúc Ch9 template) — đây là thành phần ML chính.
2. **Trích xuất nội dung** từ tài liệu text-based (PDF text layer, DOCX, Markdown, HTML) bằng parser deterministic — không dùng ML.
3. **So sánh 3 nhánh model** trên cùng một dataset:
   - Self-built CNN (64×64, tự xây, là model chính)
   - MobileNetV2 fine-tuned (224×224, so sánh — `EXTENSION`)
   - TF-IDF + SVM/RF text baseline (so sánh)
4. **Xuất kết quả có cấu trúc**: JSON + Markdown report + summary ở terminal.

**Ranh giới phạm vi (không đổi):** không OCR — scanned PDF được *phân loại như ảnh*, không đọc chữ; không layout/table extraction; >60 testcases; CLI chạy offline.

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

- [x] Design phase hoàn tất (spec chi tiết: `01-knowledge-audit` → `06-project-specification` trong `D:\Working\SIC\`)
- [x] Repository scaffold + môi trường Python (`.venv`, TF 2.21, scikit-learn, PyMuPDF, pytest...) + README/ARCHITECTURE
- [x] Dataset v0: **700 trang** — 500 RVL-CDIP (100/class × letter, form, report, article, invoice) + 200 SROIE receipts — nguồn ngoài, `EXTENSION`, provenance từng ảnh tại `datasets/raw/PROVENANCE.csv` (cột `source`: rvlcdip/sroie); split manifest 70/15/15 theo document đã build (0 leak, đủ 6 class); **spot-check nhãn đã duyệt bởi user (2026-08-19)**
- [x] Stage 3 preprocessing: `src/docproc/preprocess/` (image 64×64/224×224 deterministic + TF-IDF wrapper) — golden tests, **18/18 pass**
- [x] **Stages 4–6: Self-built CNN (Architecture A) train + evaluate** — E1 best val_accuracy 60% (epoch 27/30); frozen test: accuracy **57.1%**, macro-F1 **0.514**, majority baseline 28.6% → **gate PASS** (+28.6 pts, ≥0.50 F1). Artifacts: `runs/E1/` (weights `best.keras`, history, metrics, confusion matrix, learning curves)
- [x] Stage 7 Document I/O: `src/docproc/io/` — file detection (magic bytes + scanned-PDF probe), parsers (PDF/DOCX/MD/HTML), renderer + embedded-image extraction, structured error contracts — golden tests
- [x] Architecture deepening: `docproc/paths` (một nguồn duy nhất cho layout/config), dataset module với registry 2 arms (cnn 64×64 / finetune 224×224), `evaluation.report` dùng chung cho mọi experiment — manifest rebuild **byte-identical**, E1 tái lập chính xác
- [ ] Implementation (theo `06-project-specification.md` §8: Stage 0 → Stage 13) — test suite hiện tại **92/92 pass**

## Cấu trúc

```
src/docproc/{io, preprocess, models, training, inference, pipeline, evaluation, cli}
configs/            # mọi hyperparameter, không hard-code
datasets/           # raw/ pages/ splits/ text/ (gitignored)
models/artifacts/   # weights, checkpoints, vectorizer (gitignored)
runs/               # log + metrics từng experiment (gitignored)
docs/               # tài liệu dự án
```

*Test suite (>92 testcases) nằm ngoài repo tại `D:\Working\SIC\rcn-tests\` — chạy bằng `python -m pytest -q` từ thư mục RCN (quyết định của chủ project, spec 06 v2.7).*

*File này sẽ được cập nhật dần theo tiến độ implementation.*