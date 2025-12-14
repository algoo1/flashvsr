
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/seedvr2-runpod

WORKDIR /app

# 1. System Dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ffmpeg \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Link python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py

# 2. Clone SeedVR Repo (Specific commit for stability if needed, but using HEAD of main)
# We clone into /app/seedvr_repo so it's a sibling to our code as expected by loader.py
RUN git clone https://github.com/ByteDance-Seed/SeedVR /app/seedvr_repo

# 3. Python Dependencies
RUN pip install ninja packaging

# Install generic requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Flash Attention 2 (Use pre-built wheel for Torch 2.4 + CUDA 12.x to save hours of build time)
# Using v2.6.3 which supports Torch 2.4
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 4. Copy Application Code
COPY . /app/seedvr2-runpod

# Set working directory to project root
WORKDIR /app/seedvr2-runpod

# 5. Entrypoint
CMD ["python", "-u", "-m", "src.api.handler"]
