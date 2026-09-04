"""
Train Hybrid Summarizer (mT5 Encoder + Custom Decoder).

VRAM-optimized for GTX 1660 Ti 6GB:
- Encoder frozen → ~1.2GB
- Decoder trainable → ~600MB
- Activations + optimizer → ~3GB
- Total: ~5GB / 6GB

Usage:
  python scripts/train_hybrid.py --train thefirst_train.jsonl --val thefirst_val.jsonl --out hybrid_v1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.docproc.models.hybrid_summarizer import HybridSummarizer
from transformers import AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
BASE_MODEL = "google/mt5-base"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 120
BATCH_SIZE = 4
GRAD_ACCUM = 4  # Effective batch = 16
EPOCHS = 3
LR = 3e-4
WARMUP_STEPS = 500
SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════
class SummarizationDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_input_len: int, max_target_len: int):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.examples = []

        skipped = 0
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            text = r.get("text", "")
            summary = r.get("summary", "")
            if not text or not summary:
                skipped += 1
                continue
            self.examples.append((text, summary))

        print(f"  {path.name}: {len(self.examples):,} examples ({skipped} skipped)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        text, summary = self.examples[i]
        src = self.tokenizer(
            text, truncation=True, max_length=self.max_input_len,
            return_tensors="pt", padding=False,
        )
        tgt = self.tokenizer(
            summary, truncation=True, max_length=self.max_target_len,
            return_tensors="pt", padding=False,
        )
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": tgt["input_ids"].squeeze(0),
        }


def collate_fn(batch, pad_token_id: int):
    """Dynamic padding per batch."""
    max_src = max(x["input_ids"].shape[0] for x in batch)
    max_tgt = max(x["labels"].shape[0] for x in batch)

    input_ids = torch.full((len(batch), max_src), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_src), dtype=torch.long)
    labels = torch.full((len(batch), max_tgt), -100, dtype=torch.long)

    for i, x in enumerate(batch):
        sl = x["input_ids"].shape[0]
        tl = x["labels"].shape[0]
        input_ids[i, :sl] = x["input_ids"]
        attention_mask[i, :sl] = x["attention_mask"]
        labels[i, :tl] = x["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════
def train(args):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    pad_token_id = tokenizer.pad_token_id or 0

    # Model
    print("\nBuilding Hybrid Summarizer...")
    model = HybridSummarizer(
        mt5_model_name=BASE_MODEL,
        decoder_layers=6,
        decoder_heads=12,
        decoder_kv_heads=4,
        decoder_ffn_dim=2048,
        decoder_dropout=0.1,
        decoder_max_seq_len=2048,
        rope_theta=10000.0,
    )
    model = model.to(device)

    params = model.count_parameters()
    print(f"  Total:    {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen:    {params['frozen']:,}")

    # Dataset
    print("\nLoading datasets...")
    train_ds = SummarizationDataset(Path(args.train), tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
    val_ds = SummarizationDataset(Path(args.val), tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        num_workers=2, pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Output dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining: {EPOCHS} epochs, batch={BATCH_SIZE}×{GRAD_ACCUM}=effective {BATCH_SIZE*GRAD_ACCUM}")
    print(f"  Steps: {total_steps}")
    print(f"  LR: {LR} → cosine decay")
    print(f"  Output: {out_dir}")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Teacher forcing: decoder input = labels shifted right
            decoder_input_ids = batch["labels"].clone()
            decoder_input_ids[decoder_input_ids == -100] = pad_token_id

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=decoder_input_ids,
                labels=batch["labels"],
            )

            loss = out["loss"] / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (step + 1) % 200 == 0:
                avg_loss = epoch_loss / (step + 1)
                elapsed = time.time() - epoch_start
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"Time: {elapsed:.0f}s")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                decoder_input_ids = batch["labels"].clone()
                decoder_input_ids[decoder_input_ids == -100] = pad_token_id
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    labels=batch["labels"],
                )
                val_loss += out["loss"].item()

        val_loss /= len(val_loader)
        epoch_time = time.time() - epoch_start
        print(f"\n  ✅ Epoch {epoch} done | Train Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Time: {epoch_time:.0f}s\n")

        # Save checkpoint
        ckpt_dir = out_dir / f"checkpoint-epoch-{epoch}"
        model.save_pretrained(str(ckpt_dir))
        print(f"  💾 Saved: {ckpt_dir}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = out_dir / "best"
            model.save_pretrained(str(best_dir))
            print(f"  🏆 Best model: {best_dir} (val_loss={best_val_loss:.4f})")

    print(f"\n{'=' * 60}")
    print(f"Training complete! Best val_loss: {best_val_loss:.4f}")
    print(f"Model: {out_dir}/best/")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Hybrid Summarizer")
    ap.add_argument("--train", required=True, help="Training JSONL file")
    ap.add_argument("--val", required=True, help="Validation JSONL file")
    ap.add_argument("--out", default="hybrid_v1", help="Output directory")
    args = ap.parse_args()
    train(args)