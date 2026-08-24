# RCN — Architecture (v2, understanding-first)

> Đồng bộ với `06-project-specification.md` v4 (SOURCE OF TRUTH). Cập nhật 2026-08-23.

## Pipeline

```
Input: text / .txt / .md / .html / .docx / PDF-text        [ảnh & PDF scan → classify-only]
   │
   ├─ [io] detect (magic bytes + scanned probe)             ✅
   ├─ [io] extract text per format                          ✅
   │
   ├─ [nlp] L1 structure: word→sentence→paragraph           ✅  nlp/structure.py
   ├─ [nlp] L2 keywords: top-k keyphrases (in-doc TF-IDF)   ✅  nlp/keywords.py
   ├─ [nlp] L3 topics: LDA + UMass coherence k-selection    ✅  nlp/topics.py
   ├─ [nlp] L4 fields: regex schemas per class              ✅  nlp/fields.py
   │
   ├─ [router] doc-type label                               ♻️ CNN best.keras (ảnh) · SVM joblib (text)
   │
   └─ [nlp/report] understand() → JSON + Markdown + summary ✅  nlp/report.py
```

## Nguyên tắc

- **Understanding-first**: L1/L2/L3 là sản phẩm; router chỉ gán nhãn phụ trợ (thiếu artifact → `"unavailable"`, không fail).
- **Deterministic / seeded**: cùng input ⇒ cùng report (LDA seed 42; mọi hàm thuần).
- **Course-direct**: TF-IDF, LDA (Ch7), rules — không LLM, không OCR, offline.
- **Một seam duy nhất** cho test/CLI: `nlp.report.understand()`.

## Thành phần đã có

| Module | Vai trò mới |
|---|---|
| `io/detect·parsers·render` | ingestion cho hiểu tài liệu |
| `nlp/structure.py` | L1 |
| `models/cnn.py` + `runs/E1/best.keras` | router ảnh |
| `models/text_classifier.py` + SVM artifacts | router text |
| `training/`, `evaluation/` | bảo trì router artifacts |

## FUTURE (ngoài phạm vi hiện tại)

MobileNetV2 fine-tune (E3/E4) · fusion E5 · QA/tóm tắt.
