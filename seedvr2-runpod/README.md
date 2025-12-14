
# SeedVR2-3B RunPod Serverless Deployment

Production-ready deployment package for SeedVR2-3B video restoration model on RunPod Serverless.

## 🚀 Quick Start

### 1. Clone & Build
```bash
git clone https://github.com/algonum1/seedvr2-runpod.git
cd seedvr2-runpod
docker build -t seedvr2-runpod .
```

### 2. Run Locally
```bash
docker run --gpus all -p 8000:8000 seedvr2-runpod
```

### 3. Test Local Inference
```bash
# In a separate terminal
python test_local.py --video your_video.mp4
```

---

## ☁️ RunPod Deployment

1. **Push to Docker Hub**
   - Fork this repo
   - Set `DOCKERHUB_USERNAME` (`algonum1`) and `DOCKERHUB_TOKEN` secrets in GitHub Actions
   - Push to main branch to trigger build

2. **Create Template on RunPod**
   - Container Image: `algonum1/seedvr2-runpod:latest`
   - Container Disk: 20GB+
   - Volume Disk: 20GB+ (for caching weights)

3. **Deploy Serverless Endpoint**
   - Select GPU: RTX 4090 (Recommended) or A40
   - Use the template created above

---

## 🔌 API Reference

### Input Format
```json
{
  "input": {
    "video": "https://example.com/video.mp4",  // OR base64 string
    "target_width": 1280,
    "target_height": 720,
    "quality": "balanced"
  }
}
```

### Output Format
```json
{
  "status": "success",
  "output_video": "data:video/mp4;base64,AAAA...",
  "processing_time": 45.2,
  "metadata": {
    "output_resolution": "1280x720",
    "quality_mode": "balanced"
  }
}
```

---

## ⚠️ Limitations

- **Degradation**: Not robust to heavy degradations or very large motions.
- **Oversharpening**: May over-generate details on lightly degraded inputs (especially 480p).
- **Resolution**: Best results on 720p to 1080p upscaling.
- **Cold Start**: First run downloads ~10GB of weights. Use a Network Volume to cache `/app/ckpts` for faster starts.

---

## 🔧 Configuration

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `MODEL_PATH` | Path to weights | `/app/ckpts/seedvr2_ema_3b.pth` |
| `PRELOAD_MODEL` | Download weights on build | `False` |

---

## 🧪 Testing

To run tests:
```bash
python test_local.py
```
This script generates a dummy video if none provided and runs the full inference pipeline.
