#!/bin/bash

echo "Starting FlashVSR..."

# Check if model weights exist
MODEL_DIR="FlashVSR-v1.1"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Model weights not found. Downloading from Hugging Face..."
    git lfs install
    git clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
    
    # Check if download succeeded
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Failed to download model weights."
        exit 1
    fi
else
    echo "Model weights found."
fi

if [ "$RUNPOD_Serverless" == "true" ]; then
    echo "Starting in Serverless Mode..."
    python handler.py
else
    echo "Starting Gradio App..."
    python app.py
fi
