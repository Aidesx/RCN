# docproc — Kiến trúc chi tiết `src/`

> Tài liệu tham chiếu code trong `src/docproc/`. Đồng bộ với `ARCHITECTURE.md` gốc và spec `docs/design/06-project-specification.md`. Cập nhật 2026-08-24.

## Luồng dữ liệu tổng quát

```
file/txt đầu vào
   │
   ├─ io.detect_file_type()      nhận diện loại (magic bytes + probe PDF scan)
   ├─ io.extract_text()          trích text theo định dạng
   │
   ├─ report._router_text/_image gán nhãn loại tài liệu (phụ trợ)
   │
   ├─ structure.analyze_structure()   L1 word → câu → đoạn
   ├─ keywords.extract_keywords()     L2 top-k keyphrase TF-IDF
   ├─ topics.extract_topics()         L3 LDA + UMass (+ K-Means đặt nhãn)
   ├─ fields.extract_fields()         L4 regex trường dữ liệu theo nhãn
   ├─ summary.summarize()             L5 tóm tắt (extractive | abstractive + fallback)
   │
   └─ report.understand()        SEAM DUY NHẤT → record dict
      report.render_markdown()   → Markdown cho người đọc
```

## `paths.py` — nguồn sự thật về đường dẫn & config

| Hàm | Tác dụng |
|---|---|
| `load_config(name)` | Nạp `configs/<name>.yaml` |
| `dataset_config()` / `pipeline_config()` | Config dataset / pipeline |
| `class_names()` | 6 lớp chuẩn: article, form, invoice, letter, receipt, report |

Mọi module khác **không hardcode đường dẫn** — luôn đi qua đây.

## `io/` — ingestion tài liệu

| File | Nội dung chính | Vai trò |
|---|---|---|
| `detect.py` | `detect_file_type()` — magic bytes + sniff text format + đếm char PDF; `Detection`; lỗi có cấu trúc `DocumentIOError` / `ParseError` / `UnsupportedFormatError` | Quyết định đi nhánh nào (text / ảnh / scan) |
| `parsers.py` | `extract_text()` điều phối `_extract_pdf/docx/markdown/html` | Trích text deterministic, không OCR |
| `render.py` | `render_pdf_pages()` (PDF→ảnh, dpi mặc định), `extract_embedded_images_pdf()` | Chỉ để *phân loại* trang scan, không đọc chữ |

## `nlp/` — trái tim "hiểu tài liệu" (L1–L4 + báo cáo)

| File | Tầng | Thuật toán |
|---|---|---|
| `structure.py` | L1 | `analyze_structure()`: tách đoạn → câu (`_split_sentences`) → từ, trả stats + danh sách ¶ |
| `keywords.py` | L2 | `extract_keywords()`: TF-IDF trong-văn-bản trên uni+bigram, lọc stopwords (`stopwords.txt`, song ngữ) + số |
| `topics.py` | L3 | `extract_topics()`: **LDA** (seed 42), chọn k bằng **UMass coherence** (k=3..10); `_cluster_representatives()`: **PCA + K-Means** chọn keyphrase đại diện làm nhãn topic dễ đọc |
| `fields.py` | L4 | `extract_fields(text, doc_type)`: regex schema theo lớp (số HĐ, ngày → ISO, tổng tiền, bên mua/bán...); override qua `configs/fields.yaml`; chuẩn hóa giá trị (`_normalize`) |
| `summary.py` | L5 | `summarize_extractive()`: chấm điểm câu bằng keyphrase L2 + MMR khử trùng lặp (deterministic); `summarize_abstractive()`: sinh đoạn mới bằng checkpoint nhỏ <7B tại `models/artifacts/summarizer_mt5/` (mặc định mT5 XLSum VI+EN; `finetuned_checkpoint` ưu tiên) — thiếu model/deps ⇒ `NotImplementedError`, caller fallback extractive (D2); model load có `lru_cache`. Config: `configs/summary.yaml` |
| `report.py` | Ghép | **Seam duy nhất**: `understand(text)` / `understand_file(path)` → record `{source, doc_type, structure, keywords, topics, fields, summary, timing}`; `render_markdown(record)`; router text `_router_text()` load artifacts SVM, router ảnh `_router_image()` gọi CNN |

Nguyên tắc: **router chỉ là nhãn phụ trợ** — thiếu artifact/model → `"unavailable"`, pipeline L1–L4 vẫn chạy, không fail.

## `preprocess/` — chuẩn bị input cho model

| File | Vai trò |
|---|---|
| `image.py` | Decode RGB → resize **bicubic** → float32 [0,1]. Hai arm kích thước: `cnn_tensor()` 64×64×3, `finetune_tensor()` 224×224×3 (FUTURE). Deterministic hoàn toàn |
| `text.py` | `TextVectorizer`: wrapper TF-IDF (uni+bigram, stopwords), `save()/load()` joblib |

## `models/` — định nghĩa kiến trúc router (chưa train)

| File | Nội dung |
|---|---|
| `cnn.py` | `build_model()` dựng **Architecture A** đúng theo `configs/cnn.yaml`: Conv(32,5)-Pool-Conv(64,5)-Pool-Flatten-Dense(256)-Dropout(0.5)-Softmax(C); Adam lr=0.001, sparse CE |
| `text_classifier.py` | `train_baseline()`: GridSearchCV (LinearSVC vs RandomForest, 5-fold f1_macro) → dump artifact vào `../../models/artifacts/`; `evaluate_baseline()`: test + majority baseline + acceptance gate |

Phân biệt quan trọng:
- Thư mục này = **bản thiết kế** (code kiến trúc).
- `../../models/artifacts/` (ngoài src) = **model đã học xong** (`text_vectorizer.joblib`, `text_model_svm.joblib`) mà CLI load lúc chạy.
- CNN checkpoint đã train nằm ở `runs/E1/best.keras`.

## `training/` — hạ tầng huấn luyện dùng chung

| File | Vai trò |
|---|---|
| `data.py` | Registry dataset 2 arms (cnn 64×64 / finetune 224×224): `read_manifest()`, `load_split_arrays()`, `make_datasets()` từ `datasets/splits/manifest.csv` |
| `harness.py` | `run_training()`: set seed 42, snapshot config đầy đủ vào `runs/<tên>/config.yaml`, fit + EarlyStopping/ModelCheckpoint, ghi `history.csv` + `metrics.json` |

## `evaluation/` — đo lường dùng chung mọi experiment

| File | Vai trò |
|---|---|
| `metrics.py` | `compute_metrics()` accuracy/macro-F1/confusion; `majority_class_index*()` baseline đa số; `acceptance_gate()` chốt PASS/FAIL so với baseline |
| `report.py` | `report_run()`: frozen-test trên split, ghi metrics.json + confusion CSV + learning curves PNG (`save_curves`) |

## Quy ước xuyên suốt

1. **Deterministic / seeded**: cùng input ⇒ cùng output (seed 42 mọi nơi).
2. **Một seam duy nhất** cho CLI/test: `nlp.report.understand()`.
3. **Understanding-first**: L1–L3 là sản phẩm chính; router phụ trợ.
4. **Không OCR, không LLM/API cloud** khi chạy; model seq2seq nhỏ (<7B) chạy local offline được phép (07-summary).
5. Mọi hyperparameter sống trong `configs/`, không hardcode trong code.

Kiểm chứng: bộ test 186/186 pass (`python -m pytest -q` từ thư mục gốc).
