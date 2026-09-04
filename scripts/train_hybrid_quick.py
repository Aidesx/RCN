"""Train Hybrid Summarizer — optimized for RTX 5090 31GB.
Features: FP16 AMP, early stopping, ROUGE eval, checkpoint resume.

Usage:
  python train.py --data_dir ./data --out output --epochs 5
  python train.py --resume output/checkpoint-3  # resume from epoch 3
"""
import argparse, json, sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_summarizer import HybridSummarizer
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer

SEED = 42
MAX_INPUT = 512
MAX_TARGET = 120
EARLY_STOP_PATIENCE = 3

def compute_rouge(preds: list[str], refs: list[str]) -> dict:
    """Compute ROUGE-1/2/L scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for p, r in zip(preds, refs):
            s = scorer.score(r, p)
            for k in scores:
                scores[k].append(s[k].fmeasure)
        return {k: sum(v)/len(v) for k, v in scores.items()}
    except ImportError:
        return {}

def generate_summary(model, tokenizer, text: str, device) -> str:
    """Generate summary for a single text."""
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(device)
    with torch.no_grad():
        enc = model.encoder(**inp).last_hidden_state
        dec_ids = torch.full((1, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
        for _ in range(MAX_TARGET):
            d, _ = model.decoder(dec_ids, enc)
            nxt = model.lm_head(d[:, -1:]).argmax(-1)
            dec_ids = torch.cat([dec_ids, nxt], dim=1)
            if nxt.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(dec_ids[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data", help="Directory with train_*/val_*.jsonl")
    ap.add_argument("--out", default="output")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500, help="Warmup steps")
    ap.add_argument("--fp16", action="store_true", default=True, help="Mixed precision (default: ON)")
    ap.add_argument("--no_fp16", action="store_true", help="Disable FP16")
    ap.add_argument("--grad_checkpoint", action="store_true", help="Gradient checkpointing")
    ap.add_argument("--resume", default=None, help="Resume from checkpoint dir")
    ap.add_argument("--eval_steps", type=int, default=2000, help="ROUGE eval every N steps")
    args = ap.parse_args()

    use_fp16 = args.fp16 and not args.no_fp16
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory
    print(f"VRAM: {vram / 1024**3:.1f} GB")
    print(f"FP16: {'ON' if use_fp16 else 'OFF'}")

    # ── Data ───────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_files = sorted(data_dir.glob("*train*.jsonl"))
    val_files = sorted(data_dir.glob("*val*.jsonl"))
    
    if not train_files:
        print(f"ERROR: No train files in {data_dir}")
        sys.exit(1)
    
    print(f"\nLoading {len(train_files)} train files...")
    train_pairs = []
    for f in train_files:
        lines = [json.loads(l) for l in f.read_text(encoding="utf-8").split("\n") if l.strip()]
        train_pairs.extend(lines)
        print(f"  {f.name}: {len(lines):,}")
    
    val_pairs = []
    for f in val_files:
        lines = [json.loads(l) for l in f.read_text(encoding="utf-8").split("\n") if l.strip()]
        val_pairs.extend(lines)
        print(f"  {f.name}: {len(lines):,}")
    
    print(f"Total: Train={len(train_pairs):,}, Val={len(val_pairs):,}")

    # ── Model ──────────────────────────────────────────────────────────
    print("\nBuilding Hybrid Summarizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    
    if args.resume:
        model = HybridSummarizer.from_pretrained(args.resume)
        print(f"  Resumed from {args.resume}")
    else:
        model = HybridSummarizer(mt5_model_name="google/mt5-base")
    
    model = model.to(device)
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()
        print("  Gradient checkpointing: ON")
    
    params = model.count_parameters()
    print(f"  Trainable: {params['trainable']:,} / Total: {params['total']:,}")

    # ── Dataset ────────────────────────────────────────────────────────
    class DS(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]

    def collate(batch):
        src = tokenizer([x["text"] for x in batch], truncation=True, max_length=MAX_INPUT,
                        padding=True, return_tensors="pt")
        tgt = tokenizer([x["summary"] for x in batch], truncation=True, max_length=MAX_TARGET,
                        padding=True, return_tensors="pt")
        labels = tgt["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": src["input_ids"], "attention_mask": src["attention_mask"], "labels": labels}

    nw = min(4, os.cpu_count() or 1)
    train_dl = DataLoader(DS(train_pairs), args.batch, shuffle=True, collate_fn=collate, num_workers=nw, pin_memory=True)
    val_dl = DataLoader(DS(val_pairs), args.batch, shuffle=False, collate_fn=collate, num_workers=nw, pin_memory=True)

    # ── Optimizer ──────────────────────────────────────────────────────
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs // args.grad_accum
    sched = CosineAnnealingLR(opt, T_max=total_steps)
    scaler = GradScaler() if use_fp16 else None
    effective_batch = args.batch * args.grad_accum
    print(f"\nTraining: {args.epochs} epochs, {len(train_dl)} batches/epoch")
    print(f"  Batch: {args.batch} × {args.grad_accum} = {effective_batch}, Steps: {total_steps}")
    print(f"  LR: {args.lr:.0e} → 0 (cosine), Warmup: {args.warmup} steps")

    # ── Train ──────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    best_val = float("inf")
    patience_counter = 0
    global_step = 0
    start = time.time()
    start_epoch = 1

    if args.resume:
        # Find last epoch from checkpoint name
        ckpt_name = Path(args.resume).name
        if "checkpoint-" in ckpt_name:
            start_epoch = int(ckpt_name.split("-")[-1]) + 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        opt.zero_grad()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            dec_in = labels.clone()
            dec_in[dec_in == -100] = tokenizer.pad_token_id

            if use_fp16:
                with autocast(device_type="cuda"):
                    with torch.no_grad():
                        enc_out = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                    dec_out, _ = model.decoder(input_ids=dec_in, encoder_hidden_states=enc_out, encoder_attention_mask=mask)
                    logits = model.lm_head(dec_out)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
                loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * args.grad_accum
            else:
                with torch.no_grad():
                    enc_out = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                dec_out, _ = model.decoder(input_ids=dec_in, encoder_hidden_states=enc_out, encoder_attention_mask=mask)
                logits = model.lm_head(dec_out)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
                loss = loss / args.grad_accum
                loss.backward()
                epoch_loss += loss.item() * args.grad_accum

            if (batch_idx + 1) % args.grad_accum == 0:
                if use_fp16:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1

            if (batch_idx + 1) % 500 == 0:
                avg = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - start
                lr = sched.get_last_lr()[0]
                print(f"  Epoch {epoch} | {batch_idx+1}/{len(train_dl)} | Loss: {avg:.4f} | LR: {lr:.2e} | {elapsed:.0f}s")

            # ROUGE eval during training
            if args.eval_steps > 0 and global_step > 0 and global_step % args.eval_steps == 0:
                model.eval()
                rouge_preds, rouge_refs = [], []
                with torch.no_grad():
                    for i, batch in enumerate(val_dl):
                        if i >= 10:  # Only 10 batches for speed
                            break
                        for text, summary in zip(
                            [val_pairs[j]["text"] for j in range(i*args.batch, min((i+1)*args.batch, len(val_pairs)))],
                            [val_pairs[j]["summary"] for j in range(i*args.batch, min((i+1)*args.batch, len(val_pairs)))]
                        ):
                            pred = generate_summary(model, tokenizer, text, device)
                            rouge_preds.append(pred)
                            rouge_refs.append(summary)
                
                rouge = compute_rouge(rouge_preds, rouge_refs)
                if rouge:
                    print(f"  📊 ROUGE-L: {rouge.get('rougeL', 0):.4f} | ROUGE-1: {rouge.get('rouge1', 0):.4f} | ROUGE-2: {rouge.get('rouge2', 0):.4f}")
                model.train()

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                dec_in = labels.clone()
                dec_in[dec_in == -100] = tokenizer.pad_token_id
                enc_out = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                dec_out, _ = model.decoder(input_ids=dec_in, encoder_hidden_states=enc_out, encoder_attention_mask=mask)
                logits = model.lm_head(dec_out)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                val_loss += F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100).item()

        val_loss /= len(val_dl)
        elapsed = time.time() - start
        print(f"\n  ✅ Epoch {epoch} | Train: {epoch_loss/len(train_dl):.4f} | Val: {val_loss:.4f} | {elapsed:.0f}s\n")

        # ── Checkpoint ──────────────────────────────────────────────────
        ckpt_dir = out_dir / f"checkpoint-{epoch}"
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        torch.save({
            'epoch': epoch,
            'optimizer': opt.state_dict(),
            'scheduler': sched.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'global_step': global_step,
            'best_val': best_val,
        }, ckpt_dir / "trainer_state.pt")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            model.save_pretrained(str(out_dir / "best"))
            tokenizer.save_pretrained(str(out_dir / "best"))
            print(f"  🏆 New best val_loss: {best_val:.4f}")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break

    print(f"\n{'='*50}")
    print(f"Training complete! Best val_loss: {best_val:.4f}")
    print(f"Model: {out_dir}/best/")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()