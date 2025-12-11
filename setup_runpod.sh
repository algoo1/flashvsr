#!/bin/bash
set -e

echo "Updating system packages..."
apt-get update && apt-get install -y git git-lfs build-essential libgl1-mesa-glx libglib2.0-0 ninja-build

echo "Installing Python build tools..."
pip install packaging ninja

echo "Installing Block-Sparse-Attention (This may take a few minutes)..."
if [ ! -d "Block-Sparse-Attention" ]; then
    git clone https://github.com/mit-han-lab/Block-Sparse-Attention
fi
cd Block-Sparse-Attention
python setup.py install
cd ..

echo "Installing FlashVSR Dependencies..."
pip install -e .
pip install -r requirements.txt

echo "Setup Complete! You can now run: python app.py"
