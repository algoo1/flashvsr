"""
RunPod Serverless Handler for FlashVSR
Uses RunPod's base PyTorch image and installs dependencies on first run
"""
import runpod
import os
import sys
import subprocess
import torch

# Global variable to cache the model
pipeline = None

def install_dependencies():
    """Install FlashVSR and dependencies on first cold start"""
    print("🔧 Installing dependencies...")
    
    # Install Block-Sparse-Attention
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/mit-han-lab/Block-Sparse-Attention.git"
    ], check=True)
    
    # Clone FlashVSR repo
    if not os.path.exists("/workspace/FlashVSR-v1.1"):
        subprocess.run([
            "git", "clone", 
            "https://huggingface.co/IVRL/FlashVSR-v1.1",
            "/workspace/FlashVSR-v1.1"
        ], check=True)
    
    # Add to path
    sys.path.insert(0, "/workspace/FlashVSR-v1.1")
    
    print("✅ Dependencies installed!")

def init_model():
    """Initialize the FlashVSR pipeline (called once per worker)"""
    global pipeline
    
    if pipeline is None:
        print("🚀 Loading FlashVSR model...")
        
        # Install dependencies if needed
        try:
            import block_sparse_attn
        except ImportError:
            install_dependencies()
        
        # Import FlashVSR modules
        from pipeline_flashvsr import FlashVSRPipeline
        
        # Load model
        pipeline = FlashVSRPipeline.from_pretrained(
            "/workspace/FlashVSR-v1.1",
            torch_dtype=torch.bfloat16
        ).to("cuda")
        
        print("✅ Model loaded!")
    
    return pipeline

def handler(job):
    """
    RunPod serverless handler
    Input: {"video_url": "https://..."}
    Output: {"output_url": "https://..."}
    """
    job_input = job['input']
    
    # Validate input
    video_url = job_input.get('video_url')
    if not video_url:
        return {"error": "Missing 'video_url' in input"}
    
    try:
        # Initialize model
        pipe = init_model()
        
        # Download input video
        import urllib.request
        import tempfile
        import imageio
        import numpy as np
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            
            print(f"📥 Downloading video from {video_url}")
            urllib.request.urlretrieve(video_url, input_path)
            
            # Process video
            print("🎬 Processing video...")
            
            # Import helper functions
            sys.path.insert(0, "/workspace/FlashVSR-v1.1")
            from utils import prepare_input_tensor, tensor2video
            
            scale = 4
            LQ, th, tw, F, fps = prepare_input_tensor(
                input_path, 
                scale=scale, 
                dtype=torch.bfloat16, 
                device='cuda'
            )
            
            # Run inference
            result_video = pipe(
                prompt="", 
                negative_prompt="", 
                cfg_scale=1.0, 
                num_inference_steps=1, 
                seed=0,
                tiled=False,
                LQ_video=LQ, 
                num_frames=F, 
                height=th, 
                width=tw,
                is_full_block=False, 
                if_buffer=True,
                topk_ratio=2.0*768*1280/(th*tw),
                kv_ratio=3.0,
                local_range=11,
                color_fix=True
            )
            
            # Save output
            output_frames = tensor2video(result_video)
            imageio.mimsave(
                output_path, 
                [np.array(f) for f in output_frames], 
                fps=fps, 
                quality=6
            )
            
            print("✅ Processing complete!")
            
            # Upload to RunPod storage (or return base64)
            # For now, return success message
            # In production, upload to S3/GCS and return URL
            
            return {
                "status": "success",
                "message": "Video processed successfully",
                "frames": F,
                "resolution": f"{tw}x{th}"
            }
            
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Start the serverless worker
runpod.serverless.start({"handler": handler})
