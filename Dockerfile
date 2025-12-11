
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python build tools
RUN pip install --no-cache-dir packaging ninja

# 1. Install Block-Sparse-Attention (Critical Step)
# This usually takes time and memory.
# Limit max jobs to prevent OOM/Crash on Docker Desktop
ENV MAX_JOBS=1
WORKDIR /tmp
RUN git clone https://github.com/mit-han-lab/Block-Sparse-Attention && \
    cd Block-Sparse-Attention && \
    python setup.py install && \
    cd .. && rm -rf Block-Sparse-Attention

# 2. Setup App Directory
WORKDIR /app

# 3. Clone FlashVSR Code
# We clone the code directly here. 
RUN git clone https://github.com/OpenImagingLab/FlashVSR.git .

# 4. Install Dependencies
# Copy our cleaned requirements.txt or use the one from repo
COPY requirements.txt requirements_extra.txt
# Install repo requirements first, then ours
RUN pip install -e . && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_extra.txt

# 5. Copy App Code
COPY app.py .
COPY start.sh .
RUN chmod +x start.sh

# Expose Gradio Port
EXPOSE 7860

# Start script
# For Serverless, we should run the handler. 
# CMD ["python", "-u", "handler.py"]
# But to keep supporting Gradio (Pods), we use a script that checks env var or starts app?
# For now, let's default to start.sh which we will modify to support both.
CMD ["./start.sh"]
