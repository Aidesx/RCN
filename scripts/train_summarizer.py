"""Fine-tune a small (<7B) seq2seq summarizer on XLSum pairs — the ONE
trainer for every host: local CPU, local GPU, or Colab T4 (same file, same
flags).

Usage:
  python scripts/train_summarizer.py \
      --train <train.jsonl> --val <val.jsonl> \
      [--out DIR] [--epochs N] [--batch-size N] [--lr F] \
      [--base-model MODEL_OR_DIR]

On Colab: upload this file + the jsonl pair files, then run the same command
in a cell prefixed with `!`. After training, point
configs/summary.yaml -> abstractive.finetuned_checkpoint at the output dir
copied under models/artifacts/.

Version-tolerant: works on transformers 4.x and 5.x by probing signatures.
"""
import argparse
import inspect
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

BASE_MODEL = "VietAI/vit5-base"
MAX_IN_TOKENS = 512
MAX_OUT_TOKENS = 120


def load_tokenizer(base_model: str):
    """AutoTokenizer first; vit5-base on transformers 5.x crashes in the slow
    T5 path, so fall back to loading tokenizer.json (local dir or Hub)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(base_model)
    except Exception:
        from transformers import T5TokenizerFast

        local = Path(base_model) / "tokenizer.json"
        if local.exists():
            print("AutoTokenizer failed; using local tokenizer.json directly")
            return T5TokenizerFast(tokenizer_file=str(local),
                                   model_max_length=512)
        from huggingface_hub import hf_hub_download

        f = hf_hub_download(base_model, "tokenizer.json")
        print("AutoTokenizer failed; using Hub tokenizer.json directly")
        return T5TokenizerFast(tokenizer_file=f, model_max_length=512)


class PairDataset(Dataset):
    """Tokenized {text -> summary} pairs; collator pads dynamically per batch."""

    def __init__(self, path: Path, tokenizer):
        self.examples = []
        skipped = 0
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            enc = tokenizer(r["text"], truncation=True,
                            max_length=MAX_IN_TOKENS)
            dec = tokenizer(r["summary"], truncation=True,
                            max_length=MAX_OUT_TOKENS)
            labels = [t if t != tokenizer.pad_token_id else -100
                      for t in dec["input_ids"]]
            if not any(v != -100 for v in labels):
                skipped += 1
                continue
            self.examples.append((enc["input_ids"], enc["attention_mask"],
                                  labels))
        print(f"{path.name}: {len(self.examples)} examples "
              f"({skipped} skipped empty)", flush=True)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, mask, labels = self.examples[i]
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(mask),
                "labels": torch.tensor(labels)}


def build_training_args(out_dir: Path, args):
    from transformers import Seq2SeqTrainingArguments

    cuda_ok = torch.cuda.is_available()
    vram_gb = 0
    gpu_name = ""
    is_16xx = False
    is_low_vram = False
    if cuda_ok:
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            gpu_name = props.name
            is_16xx = "1660" in gpu_name or "1650" in gpu_name
            is_low_vram = vram_gb < 8
        except Exception:
            pass

    # Auto-tune for GTX 16xx / 6GB: no Tensor Cores -> fp16 slower, use fp32
    # Larger cards: fp16 helps. WDDM overhead mitigated by pin_memory + workers.
    if is_16xx:
        use_fp16 = False
    else:
        use_fp16 = cuda_ok

    grad_accum = args.grad_accum if args.grad_accum is not None else (2 if (is_low_vram and cuda_ok) else 1)
    workers = args.workers if args.workers is not None else (2 if cuda_ok else 0)

    kwargs = {
        "output_dir": str(out_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": 0.05,
        "fp16": use_fp16,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "logging_steps": 25,
        "seed": 42,
        "dataloader_num_workers": workers,
        "dataloader_pin_memory": bool(cuda_ok and workers > 0),
        "report_to": "none",
        "optim": "adamw_torch",
    }
    # Reduce CPU/GPU sync stalls on WDDM (GeForce display + compute)
    if cuda_ok:
        kwargs["dataloader_prefetch_factor"] = 2
        kwargs["dataloader_persistent_workers"] = workers > 0
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    # transformers >=4.46 renamed evaluation_strategy -> eval_strategy; some
    # versions expose generation_max_length / generation_num_beams.
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    if "generation_max_length" in params:
        kwargs["generation_max_length"] = MAX_OUT_TOKENS
    if "generation_num_beams" in params:
        kwargs["generation_num_beams"] = 2
    # drop any arg this version no longer accepts (e.g. warmup_ratio in 5.x)
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return Seq2SeqTrainingArguments(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="vit5_finetuned")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--grad-accum", type=int, default=None, help="gradient accumulation steps (auto: 2 on <8GB GPU)")
    ap.add_argument("--workers", type=int, default=None, help="dataloader workers (auto: 2 on GPU, 0 on CPU)")
    ap.add_argument("--throttle-ms", type=int, default=0,
                    help="sleep ms after each optimizer step; caps GPU util "
                         "so the host stays usable (e.g. 4000 ≈ 80% util)")
    ap.add_argument("--resume", default=None,
                    help="'auto' = continue from newest checkpoint under --out")
    args = ap.parse_args()

    from transformers import (
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    train_ds = PairDataset(Path(args.train), tokenizer)
    val_ds = PairDataset(Path(args.val), tokenizer)

    targs = build_training_args(out_dir, args)
    trainer_kwargs = {
        "model": model,
        "args": targs,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": DataCollatorForSeq2Seq(tokenizer, model=model),
    }
    tr_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in tr_params:   # transformers >= 4.46 / 5.x
        trainer_kwargs["processing_class"] = tokenizer
    else:                                 # older API
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    if args.throttle_ms > 0:
        import time as _time

        from transformers import TrainerCallback

        throttle_ms = args.throttle_ms

        class _ThrottleCB(TrainerCallback):
            """Post-step sleep + cuda sync: caps GPU util so the host stays
            responsive for other work (desktop apps etc)."""

            def on_step_end(self, args, state, control, **kw):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    _time.sleep(throttle_ms / 1000.0)

        trainer.add_callback(_ThrottleCB())
        print(f"throttle: {throttle_ms} ms/step "
              f"(GPU util target ~80%)", flush=True)

    resume_ckpt = None
    if args.resume == "auto":
        import re as _re
        cks = sorted(
            Path(args.out).glob("runs_ft/checkpoint-*"),
            key=lambda p: int(_re.findall(r"\d+", p.name)[-1]) if _re.search(r"\d+", p.name) else -1,
        )
        # only resume from checkpoints whose optimizer state exists (complete save)
        cks = [c for c in cks if (c / "trainer_state.json").exists()]
        resume_ckpt = str(cks[-1]) if cks else None
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final = out_dir / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"saved final checkpoint -> {final}")


if __name__ == "__main__":
    main()
