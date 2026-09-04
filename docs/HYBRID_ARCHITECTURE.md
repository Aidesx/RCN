# Hybrid Summarizer — Kiến trúc & Training (ĐÃ CẬP NHẬT)

> Phiên bản mới nhất: **v5** — Flash Attention + vocab prune + fully-frozen encoder.
> Standalone tại `D:\Working\SIC\hybrid_deploy\`, backup các phiên tại `D:\Working\SIC\hybrid_backups\`.

## Tổng quan

```
INPUT TEXT → mT5 Encoder (FROZEN) → Custom Decoder (trainable) → LM Head → SUMMARY
```

- **Encoder**: `google/mt5-base`, 12 layer, **hoàn toàn frozen** (không unfreeze — xem "Bugs đã sửa").
- **Decoder**: 6 layer custom, khởi tạo từ **layer chẵn** (0,2,4,6,8,10) của mT5 decoder (chiến lược DistilBERT).

## Dữ liệu

| Dataset | Số pairs |
|---------|----------|
| thefirst | 936,842 |
| thesecond | 1,430,490 |
| thethird | 696,161 |
| **Tổng** | **3,063,493** |

Không có validation split (train loop tự bỏ qua val khi `Val=0`).

---

## 1. Encoder: mT5-base (frozen)

```
google/mt5-base (pretrained, 101 ngôn ngữ)
├── 12 layers T5 — TẤT CẢ FROZEN
├── d_model = 768, num_heads = 12, FFN = 2048
└── chạy dưới torch.no_grad() trong training loop
```

## 2. Decoder: Custom (6 layers)

Mỗi `DecoderLayer`:
```
RMSNorm → Self-Attention (GQA + RoPE + Flash) → +
RMSNorm → Cross-Attention (GQA + Flash)      → +
RMSNorm → SwiGLU FFN                        → +
RMSNorm (output)
```

| Thành phần | Chi tiết |
|-----------|----------|
| Self/Cross attention | **Flash Attention (SDPA)** — `flash=True` trên RTX 5090 |
| GQA | 12 Q heads / 4 KV heads (repeat 3×) |
| RoPE | θ=10000, max_seq=2048 |
| FFN | SwiGLU: `w3(silu(w1(x)) * w2(x))`, dim 2048 |
| Norm | RMSNorm (eps 1e-6) |
| Embedding | `Embedding(vocab_size, 768, padding_idx=0)`, **weight-tied** với LM head |

### Khởi tạo weight từ mT5 (layer chẵn)

| Our Component | mT5 Source | Mapping |
|--------------|-----------|---------|
| `self_attn.q_proj` | `SelfAttention.q` (layer chẵn) | Copy |
| `self_attn.k_proj` | `SelfAttention.k` | 12→4 heads (avg groups 3) |
| `self_attn.v_proj` | `SelfAttention.v` | 12→4 heads (avg groups 3) |
| `self_attn.o_proj` | `SelfAttention.o` | Copy |
| `cross_attn_q` | `EncDecAttention.q` | Copy |
| `cross_attn_k/v` | `EncDecAttention.k/v` | 12→4 heads (avg) |
| `cross_attn_o` | `EncDecAttention.o` | Copy |
| `ffn.w1/w2/w3` | `DenseReluDense.wi_0/wi_1/wo` | Gate/Up/Down |
| embedding + lm_head | mT5 decoder embed / lm_head | Copy hoặc **map theo vocab đã prune** |

## 3. LM Head

`Linear(768, vocab_size, bias=False)`, **weight-tied** với token embedding.

## 4. Vocab Prune (250K → ~50K)

`prune_vocab.py`:
1. Đếm tần suất token trên toàn bộ data (original tokenizer).
2. Giữ `keep_top=50000` token phổ biến nhất (+ min_freq=3).
3. Lưu `kept_orig_indices.pt` — **map pruned index → original index** (để copy embedding đúng).
4. Pre-tokenize: **source dùng original tokenizer**, **target dùng pruned tokenizer**.

> **QUAN TRỌNG**: source phải giữ index vocab GỐC (encoder dùng embedding 250K), target dùng index vocab PRUNED (decoder/lm_head dùng vocab ~50K). Bug này đã fix trong v5.

## 5. Training

```bash
nohup python3 train.py \
  --data_dir ~/hybrid_train/pruned \
  --out output_final \
  --epochs 3 \
  --bf16 \
  --batch 128 --grad_accum 2 \
  --tokenizer ~/hybrid_train/pruned/tokenizer \
  --embedding_map ~/hybrid_train/pruned/kept_orig_indices.pt \
  > train.log 2>&1 &
```

| Tham số | Giá trị |
|---------|---------|
| Precision | **BF16** (FP16 = NaN, FP32 = OOM ở batch lớn) |
| Batch | 32 (full vocab) / **128–256** (pruned vocab) |
| Effective batch | 256 |
| Epochs | **3** (đủ hội tụ với 3M pairs) |
| LR | 8e-4, CosineAnnealing + warmup 500 |
| Optimizer | **AdamW fused** |
| Loss | CE + **label_smoothing 0.1**, ignore_index=-100 |
| Speed opts | torch.compile (reduce-overhead) + matmul_precision=high + prefetch_factor=4 + persistent_workers |

## 6. Generation

| Tham số | Giá trị |
|---------|---------|
| num_beams | 4 |
| max_new_tokens | 128 |
| min_new_tokens | **25** |
| repetition_penalty | **1.0 (tắt)** |
| no_repeat_ngram_size | **4** |
| length_penalty | 1.0 |
| early_stopping | True |

## 7. Thời gian train (RTX 5090, 31 GB)

| Cấu hình | 1 epoch | 3 epochs |
|----------|---------|----------|
| Full vocab, batch 32 | ~2.5h | ~8h |
| **Pruned vocab, batch 128** | ~1h | **~3h** |

## 8. Bugs đã tìm & sửa (lịch sử)

| # | Bug | Fix |
|---|-----|-----|
| 1 | `(1-mask)*-inf` = NaN ở cross-attention | Dùng boolean mask |
| 2 | FP16 = NaN trên GPU | Chuyển sang BF16 |
| 3 | Encoder unfreeze 2 layer "chết" (train loop bọc `no_grad()`) | Frozen hoàn toàn |
| 4 | Vocab prune sai index (source tokenize bằng tokenizer prune + embedding copy theo index) | Source dùng tokenizer gốc + embedding map |
| 5 | `scheduler.step()` trước `optimizer.step()` | Đổi thứ tự |
| 6 | Val=0 gây chia 0 | Bỏ qua validation khi không có val |
| 7 | mT5 init lấy 6 layer đầu (mất nửa phổ) | Layer chẵn 0,2,4,6,8,10 |

## 9. File structure (hybrid_deploy/)

```
hybrid_deploy/
├── hybrid_summarizer.py   # Kiến trúc model
├── train.py               # Training loop
├── prune_vocab.py         # Vocab prune + pre-tokenize
├── infer.py               # Inference
├── quick_data.py          # Sinh dữ liệu test nhanh
├── setup.sh               # Cài môi trường máy thuê
└── requirements.txt
```

Backup: `hybrid_backups/` (v0_rcn → v5_flashattn_vocabfix), CHANGELOG.md ghi lịch sử.

## 10. Máy thuê CKEY

```bash
ssh root@n2.ckey.vn -p 3211   # RTX 5090, 31 GB, ~12K/h
```