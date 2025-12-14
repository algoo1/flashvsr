
import runpod
import os
import uuid
import shutil
import time
import torch
import traceback
from src.model.loader import load_model
from src.model.inference import process_video
from src.utils.video_io import download_video, save_base64_video, encode_video_base64

# Initialize the model globally for warm starts
print("Initializing SeedVR2-3B model...")
try:
    MODEL_RUNNER = load_model() # This will download weights if needed
    print("Model initialized successfully.")
except Exception as e:
    print(f"FAILED to initialize model: {e}")
    traceback.print_exc()
    MODEL_RUNNER = None

def handler(job):
    """
    RunPod serverless handler.
    """
    global MODEL_RUNNER
    
    # Re-try initialization if it failed previously (optional)
    if MODEL_RUNNER is None:
        try:
            MODEL_RUNNER = load_model()
        except Exception as e:
            return {"error": f"Model initialization failed: {str(e)}"}

    job_input = job.get('input', {})
    if not job_input:
        return {"error": "No input provided"}
        
    # Extract inputs
    video_source = job_input.get("video")
    if not video_source:
        return {"error": "Missing 'video' parameter"}
        
    target_height = job_input.get("target_height", 720)
    target_width = job_input.get("target_width", 1280)
    quality_mode = job_input.get("quality", "balanced")
    
    # Setup Paths
    job_id = job.get("id", str(uuid.uuid4()))
    work_dir = os.path.join("/tmp", job_id)
    os.makedirs(work_dir, exist_ok=True)
    
    input_path = os.path.join(work_dir, "input_video") # Extension might be needed by some libs
    # We try to detect extension from URL or header, or default to .mp4
    # If base64, we might not know. ffmpeg handles extensions well usually.
    # Let's try to preserve extension if URL.
    if video_source.startswith("http"):
        ext = os.path.splitext(video_source)[1]
        if not ext: ext = ".mp4"
        input_path += ext
    else:
        input_path += ".mp4" 

    output_path = os.path.join(work_dir, "output.mp4")
    
    try:
        start_time = time.time()
        
        # 1. Get Video
        print(f"Retrieving video from input...")
        if video_source.startswith("http"):
            download_video(video_source, input_path)
        else:
            # Assume base64
            save_base64_video(video_source, input_path)
            
        # 2. Process
        print("Starting inference...")
        process_video(
            MODEL_RUNNER,
            input_path,
            output_path,
            target_resolution=(target_width, target_height), # Note: width, height logic in inference
            quality_mode=quality_mode
        )
        
        process_time = time.time() - start_time
        
        # 3. Upload/Return Output
        # Ideally upload to S3 if bucket provided in job, but user requirements say:
        # "Return processed video (URL or base64)"
        # "Expected output format: output_video: url OR base64"
        
        # For simplicity in serverless without external S3 config, we default to base64 
        # UNLESS RunPod has specific bucket intergration we can use easily. 
        # Returning large base64 strings in JSON is not ideal but standard for simple setups.
        # RunPod has an S3 upload helper? 
        # We will return base64 for now as it's self-contained.
        
        output_b64 = encode_video_base64(output_path)
        # Format as data URI
        output_data_uri = f"data:video/mp4;base64,{output_b64}"
        
        response = {
            "status": "success",
            "output_video": output_data_uri,
            "processing_time": process_time,
            "metadata": {
                "input_video": os.path.basename(input_path),
                "output_resolution": f"{target_width}x{target_height}",
                "quality_mode": quality_mode
            }
        }
        
        return response

    except Exception as e:
        print(f"Error processing job: {e}")
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}
        
    finally:
        # Cleanup
        # If running on RunPod, /tmp might not be cleared, so we must clean up.
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass
        # Clear GPU memory? handled in process_video
        torch.cuda.empty_cache()

runpod.serverless.start({"handler": handler})
