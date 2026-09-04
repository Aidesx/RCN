"""
Hybrid Summarizer: mT5 Encoder (frozen) + Custom Decoder (trainable).

Kiến trúc:
- Encoder: mT5-base (frozen, giữ multilingual knowledge)
- Decoder: Custom với RoPE, GQA, SwiGLU, RMSNorm, Flash Attention

VRAM: ~4GB (fits GTX 1660 Ti 6GB)
Trainable: ~150M params (decoder + lm_head)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MT5EncoderModel, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════
# RoPE (Rotary Position Embedding)
# ═══════════════════════════════════════════════════════════════════════
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device):
        if self._seq_len_cached is not None and self._seq_len_cached >= seq_len:
            return
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

    def forward(self, x: torch.Tensor, offset: int = 0):
        seq_len = x.shape[2]  # x: (B, heads, seq, head_dim)
        self._update_cache(seq_len + offset, x.device)
        cos = self._cos_cached[offset : offset + seq_len]  # (seq, head_dim)
        sin = self._sin_cached[offset : offset + seq_len]
        # Reshape for broadcasting: (1, 1, seq, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ═══════════════════════════════════════════════════════════════════════
# RMSNorm (thay LayerNorm — nhanh hơn)
# ═══════════════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm).to(dtype)


# ═══════════════════════════════════════════════════════════════════════
# GQA (Grouped-Query Attention) với RoPE
# ═══════════════════════════════════════════════════════════════════════
class GQAAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        num_kv_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        current_kv = (k, v) if use_cache else None

        # RoPE
        cos, sin = self.rotary(q, offset=past_kv[0].shape[2] if past_kv is not None else 0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV heads for GQA
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        return out, current_kv


# ═══════════════════════════════════════════════════════════════════════
# SwiGLU FFN (thay ReLU — chất lượng tốt hơn)
# ═══════════════════════════════════════════════════════════════════════
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int = 768, ffn_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ═══════════════════════════════════════════════════════════════════════
# Decoder Layer
# ═══════════════════════════════════════════════════════════════════════
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 768, num_heads: int = 12, num_kv_heads: int = 4,
                 ffn_dim: int = 3072, dropout: float = 0.1, max_seq_len: int = 2048,
                 rope_theta: float = 10000.0):
        super().__init__()
        self.self_attn = GQAAttention(d_model, num_heads, num_kv_heads, dropout, max_seq_len, rope_theta)
        self.cross_attn_q = nn.Linear(d_model, num_heads * (d_model // num_heads), bias=False)
        self.cross_attn_k = nn.Linear(d_model, num_kv_heads * (d_model // num_heads), bias=False)
        self.cross_attn_v = nn.Linear(d_model, num_kv_heads * (d_model // num_heads), bias=False)
        self.cross_attn_o = nn.Linear(num_heads * (d_model // num_heads), d_model, bias=False)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.ffn = SwiGLUFFN(d_model, ffn_dim, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.norm4 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        past_kv_self: Optional[tuple] = None,
        past_kv_cross: Optional[tuple] = None,
        use_cache: bool = False,
    ):
        B, T, C = x.shape

        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, new_kv_self = self.self_attn(x, self_mask, past_kv_self, use_cache)
        x = residual + self.dropout(attn_out)

        # Cross-attention (q from decoder, k/v from encoder)
        residual = x
        x_norm = self.norm2(x)
        q = self.cross_attn_q(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.cross_attn_k(encoder_out).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.cross_attn_v(encoder_out).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV for GQA
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if cross_mask is not None:
            attn_weights = attn_weights + cross_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        cross_out = torch.matmul(attn_weights, v)
        cross_out = cross_out.transpose(1, 2).contiguous().view(B, T, C)
        cross_out = self.cross_attn_o(cross_out)
        x = residual + self.dropout(cross_out)

        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ffn(x))

        # Final norm
        x = self.norm4(x)

        return x, (new_kv_self, None) if use_cache else None


# ═══════════════════════════════════════════════════════════════════════
# Custom Decoder
# ═══════════════════════════════════════════════════════════════════════
class CustomDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        num_kv_heads: int = 4,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, num_kv_heads, ffn_dim, dropout, max_seq_len, rope_theta)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(x)

        self_mask = self._causal_mask(T, x.device)

        cross_mask = None
        if encoder_attention_mask is not None:
            # Fix: (1-mask) * -inf = 0 * -inf = NaN
            # Use mask == 0 to avoid NaN
            cross_mask = torch.where(
                encoder_attention_mask[:, None, None, :].bool(),
                0.0,
                float("-inf")
            )

        new_kvs = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_self = past_key_values[i][0] if past_key_values is not None else None
            past_cross = past_key_values[i][1] if past_key_values is not None else None
            x, layer_kv = layer(x, encoder_hidden_states, self_mask, cross_mask,
                               past_self, past_cross, use_cache)
            if use_cache and layer_kv is not None:
                new_kvs.append(layer_kv)

        x = self.norm(x)
        return x, new_kvs


# ═══════════════════════════════════════════════════════════════════════
# Hybrid Summarizer (mT5 Encoder + Custom Decoder)
# ═══════════════════════════════════════════════════════════════════════
class HybridSummarizer(nn.Module):
    """mT5 encoder (frozen) + custom decoder (trainable) for summarization."""

    def __init__(
        self,
        mt5_model_name: str = "google/mt5-base",
        decoder_layers: int = 6,
        decoder_heads: int = 12,
        decoder_kv_heads: int = 4,
        decoder_ffn_dim: int = 2048,  # Match mT5 FFN dim for weight transfer
        decoder_dropout: float = 0.1,
        decoder_max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        # Load mT5 encoder
        self.encoder = MT5EncoderModel.from_pretrained(mt5_model_name)
        d_model = self.encoder.config.d_model

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Load tokenizer for vocab size
        tokenizer = AutoTokenizer.from_pretrained(mt5_model_name)
        vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id or 0

        # Custom decoder
        self.decoder = CustomDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            num_kv_heads=decoder_kv_heads,
            ffn_dim=decoder_ffn_dim,
            dropout=decoder_dropout,
            max_seq_len=decoder_max_seq_len,
            rope_theta=rope_theta,
            pad_token_id=self.pad_token_id,
        )

        # LM head (shares weights with decoder embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.decoder.token_embedding.weight  # Weight tying

        # ── CẢI TIẾN 1: Khởi tạo decoder từ mT5 decoder ─────────────────
        self._init_decoder_from_mt5(mt5_model_name)

        # ── CẢI TIẾN 2: Unfreeze 2 layer cuối encoder ──────────────────
        self._unfreeze_encoder_layers(num_layers=2)

        # ── CẢI TIẾN 3: Coverage mechanism ─────────────────────────────
        self.coverage_lambda = 0.5  # Trọng số coverage loss

        self.config = self.encoder.config  # For HF compatibility

    def _init_decoder_from_mt5(self, mt5_model_name: str):
        """Khởi tạo decoder embedding + attention weights từ mT5 decoder."""
        from transformers import MT5ForConditionalGeneration
        try:
            mt5_full = MT5ForConditionalGeneration.from_pretrained(mt5_model_name)
            mt5_decoder = mt5_full.decoder
            mt5_lm_head = mt5_full.lm_head

            # Copy token embedding
            if hasattr(mt5_decoder, 'embed_tokens'):
                self.decoder.token_embedding.weight.data.copy_(
                    mt5_decoder.embed_tokens.weight.data[:self.decoder.token_embedding.weight.shape[0]]
                )

            # Copy LM head
            if hasattr(mt5_lm_head, 'weight'):
                self.lm_head.weight.data.copy_(
                    mt5_lm_head.weight.data[:self.lm_head.weight.shape[0]]
                )

            # Copy decoder layer weights where dimensions match (first 6 layers)
            for i in range(min(self.decoder.layers.__len__(), len(mt5_decoder.block))):
                mt5_layer = mt5_decoder.block[i]
                our_layer = self.decoder.layers[i]

                # Self-attention Q, K, V, O (same dims: 768)
                if hasattr(mt5_layer.layer[0], 'SelfAttention'):
                    sa = mt5_layer.layer[0].SelfAttention
                    our_layer.self_attn.q_proj.weight.data.copy_(sa.q.weight.data)
                    # K, V: mT5 has full heads, we have fewer KV heads
                    # Average across head groups
                    k_w = sa.k.weight.data.view(12, 64, 768)
                    our_layer.self_attn.k_proj.weight.data.copy_(
                        k_w.view(4, 3, 64, 768).mean(dim=1).reshape(4*64, 768)
                    )
                    v_w = sa.v.weight.data.view(12, 64, 768)
                    our_layer.self_attn.v_proj.weight.data.copy_(
                        v_w.view(4, 3, 64, 768).mean(dim=1).reshape(4*64, 768)
                    )
                    our_layer.self_attn.o_proj.weight.data.copy_(sa.o.weight.data)

                # Cross-attention Q (decoder side)
                if hasattr(mt5_layer.layer[1], 'EncDecAttention'):
                    ca = mt5_layer.layer[1].EncDecAttention
                    our_layer.cross_attn_q.weight.data.copy_(ca.q.weight.data)
                    k_w = ca.k.weight.data.view(12, 64, 768)
                    our_layer.cross_attn_k.weight.data.copy_(
                        k_w.view(4, 3, 64, 768).mean(dim=1).reshape(4*64, 768)
                    )
                    v_w = ca.v.weight.data.view(12, 64, 768)
                    our_layer.cross_attn_v.weight.data.copy_(
                        v_w.view(4, 3, 64, 768).mean(dim=1).reshape(4*64, 768)
                    )
                    our_layer.cross_attn_o.weight.data.copy_(ca.o.weight.data)

                # FFN: mT5 uses DenseReluDense (wi_0, wi_1, wo)
                # We use SwiGLU (w1, w2, w3)
                # wi_0 = gate, wi_1 = up, wo = down
                if hasattr(mt5_layer.layer[2], 'DenseReluDense'):
                    ffn = mt5_layer.layer[2].DenseReluDense
                    our_layer.ffn.w1.weight.data.copy_(ffn.wi_0.weight.data)  # gate
                    our_layer.ffn.w2.weight.data.copy_(ffn.wi_1.weight.data)  # up
                    our_layer.ffn.w3.weight.data.copy_(ffn.wo.weight.data)    # down

            del mt5_full
            print("  ✅ Decoder initialized from mT5 decoder weights")
        except Exception as e:
            print(f"  ⚠️ Could not init from mT5 decoder: {e}")

    def _unfreeze_encoder_layers(self, num_layers: int = 2):
        """Unfreeze last N encoder layers for task adaptation."""
        blocks = self.encoder.encoder.block
        total_layers = len(blocks)
        count = 0
        for i, block in enumerate(blocks):
            if i >= total_layers - num_layers:
                for param in block.parameters():
                    param.requires_grad = True
                    count += 1
        print(f"  ✅ Unfrozen last {num_layers} encoder layers ({count} params)")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Encoder
        with torch.no_grad():
            encoder_out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        # Decoder
        decoder_out, _ = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=attention_mask,
        )

        logits = self.lm_head(decoder_out)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ce_loss

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        repetition_penalty: float = 1.2,
        min_new_tokens: int = 8,
        length_penalty: float = 1.0,
        **kwargs,
    ):
        """Beam search generation with repetition + length penalty."""
        B = input_ids.shape[0]
        device = input_ids.device
        vocab_size = self.lm_head.weight.shape[0]
        eos_token_id = self.encoder.config.eos_token_id
        pad_token_id = self.pad_token_id

        # Encoder
        with torch.no_grad():
            encoder_out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            # Expand for beam search
            encoder_out = encoder_out.unsqueeze(1).expand(B, num_beams, -1, -1).reshape(B * num_beams, -1, encoder_out.shape[-1])
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).expand(B, num_beams, -1).reshape(B * num_beams, -1)

        # Initialize beams
        beam_scores = torch.zeros(B, num_beams, device=device)
        beam_scores[:, 1:] = -1e9  # Only first beam active initially
        beam_scores = beam_scores.view(-1)

        decoder_input_ids = torch.full((B * num_beams, 1), pad_token_id, dtype=torch.long, device=device)
        done = torch.zeros(B * num_beams, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            if done.all():
                break

            decoder_out, _ = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_out,
                encoder_attention_mask=attention_mask,
            )
            logits = self.lm_head(decoder_out[:, -1:, :]).squeeze(1)  # (B*N, vocab)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(decoder_input_ids.shape[0]):
                    seen_tokens = set(decoder_input_ids[i].tolist())
                    for token_id in seen_tokens:
                        if logits[i, token_id] > 0:
                            logits[i, token_id] /= repetition_penalty
                        else:
                            logits[i, token_id] *= repetition_penalty

            # Length penalty
            if step < min_new_tokens:
                logits[:, eos_token_id] = -1e9

            scores = F.log_softmax(logits, dim=-1)
            next_scores = scores + beam_scores.unsqueeze(1)

            # Reshape to (B, num_beams * vocab)
            next_scores = next_scores.view(B, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=-1)

            # Reconstruct beam indices and token indices
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # Select top beams
            beam_scores = torch.full((B, num_beams), -1e9, device=device)
            beam_tokens = torch.zeros(B, num_beams, dtype=torch.long, device=device)
            beam_indices = torch.zeros(B, num_beams, dtype=torch.long, device=device)

            for b in range(B):
                beam_idx = 0
                for i in range(2 * num_beams):
                    if beam_idx >= num_beams:
                        break
                    prev_beam = next_indices[b, i].item()
                    token = next_tokens[b, i].item()
                    score = next_scores[b, i].item()
                    global_beam = b * num_beams + prev_beam

                    if done[global_beam]:
                        continue

                    beam_scores[b, beam_idx] = score
                    beam_tokens[b, beam_idx] = token
                    beam_indices[b, beam_idx] = prev_beam
                    beam_idx += 1

            # Update decoder inputs
            new_decoder_inputs = []
            for b in range(B):
                for n in range(num_beams):
                    prev_beam = beam_indices[b, n].item()
                    prev_ids = decoder_input_ids[b * num_beams + prev_beam]
                    new_ids = torch.cat([prev_ids, beam_tokens[b, n].unsqueeze(0)])
                    new_decoder_inputs.append(new_ids)
            decoder_input_ids = torch.stack(new_decoder_inputs)

            # Update done flags
            done = beam_tokens.view(-1) == eos_token_id

        # Return best beam for each batch (score normalized by length penalty)
        lengths = (decoder_input_ids != pad_token_id).sum(dim=1).float()  # (B*N,)
        adjusted = beam_scores / (lengths.view(B, num_beams) ** length_penalty)
        best_beams = adjusted.argmax(dim=-1)
        output = []
        for b in range(B):
            best_idx = b * num_beams + best_beams[b].item()
            output.append(decoder_input_ids[best_idx])
        return torch.stack(output)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for decoder layers (saves ~30% VRAM)."""
        def ckpt_forward(layer, x, encoder_out, self_mask, cross_mask, *args):
            def custom_forward(x, encoder_out):
                return layer(x, encoder_out, self_mask, cross_mask, *args)
            return torch.utils.checkpoint.checkpoint(custom_forward, x, encoder_out, use_reentrant=False)
        
        for layer in self.decoder.layers:
            layer._ckpt_forward = ckpt_forward
        self._gradient_checkpointing = True

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def save_pretrained(self, path: str):
        """Save decoder + lm_head weights."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save({
            "decoder": self.decoder.state_dict(),
            "lm_head": self.lm_head.state_dict(),
        }, os.path.join(path, "pytorch_model.bin"))
        self.encoder.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, mt5_model_name: str = "google/mt5-base", **kwargs):
        """Load model from checkpoint."""
        import os
        model = cls(mt5_model_name=mt5_model_name, **kwargs)
        ckpt = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        model.decoder.load_state_dict(ckpt["decoder"])
        model.lm_head.load_state_dict(ckpt["lm_head"])
        return model