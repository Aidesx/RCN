# RCN — Architecture (temporary)

> Kiến trúc tạm thời, tóm tắt từ `06-project-specification.md` (§4/§5 — SOURCE OF TRUTH). Sẽ được cập nhật khi implementation tiến triển. Mọi thay đổi phải theo luật §14 của spec.

## Pipeline (7 bước)

```
[Ingestion] file / folder
      │
[1] File detection ──► format (PDF|DOCX|MD|HTML|IMG) — magic bytes + extension (no ML)
      │
[2] Scanned check (PDF): text-layer probe (<100 non-whitespace chars = scanned; configurable)
      ├── text present ─► [3a] Text parsing ─► raw text ─► [4a] TF-IDF ─► text classifier (SVM/RF) [E0b]
      └── no text ─────► [3b] Page rendering / image decode ─► page images
                              │
                              ▼
                      [4b] Preprocessing (64×64 | 224×224; per-model normalization)
                              │
                              ▼
                 [5] ML: Architecture A (CNN 64×64) | Architecture B (MobileNetV2 224×224)
                              │
                              ▼
                 [6] Post-processing: confidence threshold + page majority vote + routing
                              │
                              ▼
                 [7] Output: JSON + Markdown report + CLI summary
```

- **Embedded images** (từ DOCX/HTML, cấu hình được) → [3b] → image branch.
- **No OCR** ở bất kỳ bước nào. Scanned page = phân loại ảnh, không đọc chữ.
- Fusion (Architecture C) **không nằm trong pipeline mặc định** — chỉ dùng nếu E5 chứng minh thắng model đơn tốt nhất.

## ML Components

### A. Self-built CNN (model chính)
- Input 64×64×3; Conv2D 32×5×5 → MaxPool → Conv2D 64×5×5 → MaxPool → Flatten → Dense 256 → Dropout 0.5 → Dense 6 softmax (Ch9 template)
- Adam 1e-3, batch 32, ≤30 epochs, ES patience 5–7 (val accuracy), checkpoint theo val accuracy
- Loss: SparseCategoricalCrossentropy

### B. MobileNetV2 fine-tuned (so sánh, `EXTENSION`)
- Input 224×224×3, ImageNet weights
- Phase 1: backbone frozen, head GAP→Dropout→Dense 6, lr 1e-3, ≤10 epochs (E3)
- Phase 2: unfreeze top 30–50%, lr 1e-4, ≤15 epochs (E4)

### C. Text baseline (E0b)
- TF-IDF + LinearSVC / RandomForest, GridSearchCV 5-fold

## Dataset & Split

- 6 classes: invoice, receipt, report, letter, form, article
- Min 300 pages (≥50/class); recommended 900–1500; folder-per-class labels
- Split 70/15/15 **theo document** (chống leakage); test split frozen
- Majority baseline = class phổ biến nhất trong TRAINING split, đo trên frozen test

## Experiments & Metrics

- E0 baselines → E1 CNN → E2 CNN+aug (optional, `EXTENSION`) → E3 frozen → E4 fine-tuned → E5 fusion (optional)
- Metrics: accuracy (primary), macro-F1, per-class P/R/F1, confusion matrix; latency/size secondary; **không CER/WER**
- Acceptance E1: ≥ majority baseline + 10 pts **và** macro-F1 ≥ 0.50

## Gates

dataset validated → training works → baseline established → I/O verified → E4 evaluated → comparison complete → E2E inference → CLI works → **>60 testcases pass**