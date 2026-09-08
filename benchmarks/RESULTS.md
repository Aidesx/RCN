# Benchmark — Đánh giá model tóm tắt (RCN v1)

Bộ benchmark **chính thức của repo**: dữ liệu đánh giá được đóng gói sẵn (`benchmarks/data/`),
script tái chạy được (`benchmarks/run_benchmark.py`), kết quả versioned (`benchmarks/results/`).

## Dữ liệu đánh giá

- `benchmarks/data/eval_vi_20260908.jsonl` — **12 bài báo Việt** (250–900 từ) + reference sapo,
  chọn ngẫu nhiên seed 42. **5 bài (id 3, 4, 6, 8, 10) bị gắn cờ `clean: false`** vì trùng tập
  train của `vit5-base-vietnews` (bằng chứng: model đó đạt R2 = 1.0000 chính xác trên 5 bài —
  học thuộc, không tóm tắt). Bảng chính thức chỉ tính trên **7 bài sạch**.

## Phương pháp

- Model trong `models/artifacts/`, chạy qua đúng seam của app (AutoTokenizer → fallback
  `T5TokenizerFast(tokenizer_file=...)`), đúng cấu hình demo `configs/summary.yaml`
  (beam 4, length_penalty 1.2, max 128 token, input ≤1024 token), CPU.
- Chỉ số: **ROUGE-2** (word F1) · **copy5%** = % summary là nguyên văn 5-gram của nguồn
  (đo "copy-paste") · **num-hall%** = % số liệu trong summary không có trong nguồn (bịa số) ·
  lặp trigram · tốc độ s/bài. **Wilcoxon signed-rank** (paired từng bài) so với `vit5_v1`.

## Kết quả chính thức — 7 bài sạch (xem `results/2026-09-08/results.md`)

| Hạng | Model | Loại | ROUGE-2 | copy5% | num-hall% | s/bài |
|---|---|---|---|---|---|---|
| 1 | `vit5_soup_0.7_v1` | Model soup | **0.289** | 52 | 0 | 9.7 |
| 2 | `vit5_soup_0.7_v2` | Model soup | 0.266 | 46 | 0 | 11.0 |
| 3 | **`vit5_v1`** ⭐ | Tự fine-tune 127k cặp VI | 0.255 | **44** | **0** | **8.7** |
| 4 | `vit5_soup_0.5_v3` | Model soup | 0.238 | 55 | 0 | 11.8 |
| 5 | `vit5_soup_0.5_v1` | Soup (v1⊕VietNews @0.5) | 0.231 | 48 | 0 | 11.8 |
| 6 | `summarizer_mt5` | mT5 đa ngôn ngữ (zero-shot) | 0.211 | **39** | 0 | 24.6 |
| 7 | `vit5_base_vietnews` | Model cộng đồng (HF) | 0.135 | 48 | 0 | 8.1 |
| 8 | `vit5_v2` | Tự fine-tune (overnight) | 0.100 | 89 | 4 | 16.2 |
| 9 | `vit5_base_original` | Gốc chưa train (đối chứng) | 0.089 | 12 | **25** | 22.2 |

*Nhóm còn lại (chamdenti, trong269, antechai, NishiKyen, giaPhu, soup 0.3/0.5 phụ): R2 0.08–0.16 — `per_doc.csv`.*

## Đọc kết quả

- **Nhóm đầu tương đương thống kê** (p = 0.27–0.73 với vit5_v1): soup_0.7_v1 ≈ soup_0.7_v2 ≈ vit5_v1.
- **Fine-tune tạo khác biệt thật**: v1 R2 0.255 vs base chưa train 0.089 (**+186%**), bịa số 25% → **0%**.
- **vit5_v1 sinh nhiều hơn chép**: copy5 44% — thấp nhất nhóm đầu (tóm tắt diễn đạt lại, không ghép câu gốc).
- Soup nghiêng về bám nguồn (copy5 46–55%); mT5 zero-shot paraphrase nhiều nhất (39%) nhưng chậm ~3×.
- Các model cộng đồng tải về (trong269, antechai, giaPhu...) không cạnh tranh được với model tự train.

## Khuyến nghị (demo)

- **Mặc định: `vit5_v1`** — điểm thuộc nhóm đầu (không kém soup có ý nghĩa), hồ sơ train đầy đủ,
  copy thấp = tóm tắt thật, 0% bịa số, nhanh, có bản fp16.
- Kể chuyện Model Soup bằng `vit5_soup_0.5_v1` (recipe có hồ sơ); đối chứng "chưa train"
  bằng `vit5_base_original`; định hướng v2 bằng `summarizer_mt5` (EN+VI).

## Tái chạy

```bash
# từ thư mục RCN/
.venv/Scripts/python.exe benchmarks/run_benchmark.py
# kết quả mới ghi vào benchmarks/results/<ngày>/ — so sánh với bản 2026-09-08
```
