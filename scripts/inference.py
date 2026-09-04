"""Inference with trained Hybrid Summarizer.
Usage: python inference.py --model output/best --text "Văn bản cần tóm tắt..."
      python inference.py --model output/best --file input.txt
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_summarizer import HybridSummarizer
import torch
from transformers import AutoTokenizer

MAX_INPUT = 512
MAX_TARGET = 120

def summarize(model, tokenizer, text: str, device) -> str:
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
    ap.add_argument("--model", required=True, help="Path to model checkpoint")
    ap.add_argument("--text", default=None, help="Text to summarize")
    ap.add_argument("--file", default=None, help="File with text to summarize")
    ap.add_argument("--beam", type=int, default=1, help="Beam size (1=greedy, 4=beam)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.model}...")
    model = HybridSummarizer.from_pretrained(args.model)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        texts = [Path(args.file).read_text(encoding="utf-8").strip()]
    else:
        # Interactive mode
        print("Enter text to summarize (Ctrl+D to exit):")
        texts = [sys.stdin.read().strip()]

    for text in texts:
        if not text:
            continue
        result = summarize(model, tokenizer, text, device)
        print(f"\n📝 Input ({len(text.split())} words): {text[:200]}...")
        print(f"📋 Summary: {result}")

if __name__ == "__main__":
    main()