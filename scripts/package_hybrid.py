"""Package for CKEY GPU deployment."""
import zipfile, shutil
from pathlib import Path

PKG_DIR = Path("hybrid_deploy")
PKG_DIR.mkdir(exist_ok=True)

# Core files
shutil.copy("src/docproc/models/hybrid_summarizer.py", PKG_DIR / "hybrid_summarizer.py")
shutil.copy("scripts/train_hybrid_quick.py", PKG_DIR / "train.py")
shutil.copy("scripts/inference.py", PKG_DIR / "inference.py")
shutil.copy("scripts/setup_gpu.sh", PKG_DIR / "setup.sh")

# Requirements
(PKG_DIR / "requirements.txt").write_text(
    "torch>=2.0.0\n"
    "transformers>=4.36.0\n"
    "datasets\n"
    "sentencepiece\n"
    "numpy<2\n"
    "rouge-score\n"
)

# README
(PKG_DIR / "README.md").write_text(
    "# Hybrid Summarizer — CKEY GPU Deployment\n\n"
    "## Setup (1 lần)\n"
    "```bash\nchmod +x setup.sh && ./setup.sh\n```\n\n"
    "## Upload data\n"
    "```bash\nmkdir -p ~/hybrid_train/data\n"
    "scp thefirst_train.jsonl user@ip:~/hybrid_train/data/\n"
    "scp thesecond_train.jsonl user@ip:~/hybrid_train/data/\n"
    "scp thethird_train.jsonl user@ip:~/hybrid_train/data/\n```\n\n"
    "## Train\n"
    "```bash\npython train.py --data_dir ~/hybrid_train/data --out output --epochs 5 --batch 8\n"
    "# Resume if interrupted:\n"
    "python train.py --data_dir ~/hybrid_train/data --out output --resume output/checkpoint-3\n```\n\n"
    "## Inference\n"
    "```bash\npython inference.py --model output/best --text \"Văn bản...\"\n```\n\n"
    "## Download model\n"
    "```bash\nscp -r user@ip:~/hybrid_train/output/best ./\n```\n\n"
    "## Flags\n"
    "| Flag | Default | Mô tả |\n"
    "|------|---------|-------|\n"
    "| --batch | 8 | Batch size |\n"
    "| --grad_accum | 2 | Gradient accumulation |\n"
    "| --epochs | 5 | Số epochs |\n"
    "| --lr | 3e-4 | Learning rate |\n"
    "| --fp16 | ON | Mixed precision (nhanh 2x) |\n"
    "| --grad_checkpoint | OFF | Tiết kiệm VRAM |\n"
    "| --resume | None | Resume từ checkpoint |\n"
    "| --eval_steps | 2000 | Đánh giá ROUGE mỗi N steps |\n"
)

# Zip
zip_path = "hybrid_deploy.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in PKG_DIR.rglob("*"):
        if f.is_file():
            zf.write(f, f.relative_to(PKG_DIR))

size_kb = Path(zip_path).stat().st_size / 1024
print(f"✅ {zip_path} ({size_kb:.0f} KB)")
for f in sorted(PKG_DIR.rglob("*")):
    if f.is_file():
        print(f"   {f.name}")

shutil.rmtree(PKG_DIR)