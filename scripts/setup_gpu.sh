#!/bin/bash
# Setup script for CKEY GPU — RTX 5090 31GB
# Run after connecting to the rented machine

set -e

echo "=== Setting up Hybrid Summarizer training environment ==="

# 1. System deps
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-dev git build-essential 2>&1 | tail -1

# 2. PyTorch with CUDA 13.x (RTX 5090 Blackwell)
echo "[2/5] Installing PyTorch..."
pip install torch torchvision torchaudio 2>&1 | tail -3

# 3. Python deps
echo "[3/5] Installing Python dependencies..."
pip install transformers==4.36.0 datasets sentencepiece "numpy<2" 2>&1 | tail -3

# 4. Verify
echo "[4/5] Verifying GPU..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 5. Create workdir
echo "[5/5] Creating workdir..."
mkdir -p ~/hybrid_train/data

echo ""
echo "=== Setup complete! ==="
echo "Next: upload model + data, then run train"