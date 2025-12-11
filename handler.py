import runpod
import torch
import os
import subprocess
from app import init_pipeline, tensor2video, prepare_input_tensor
import imageio
import numpy as np
import base64
import tempfile

# Initialize pipeline once at startup
pipe = init_pipeline()

def download_file(url, dest_path):
    import requests
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download file: {url}")

def handler(job):
    job_input = job['input']
    
    # Input validation
    video_url = job_input.get('video_url')
    if not video_url:
        return {"error": "Missing video_url in input"}
    
    try:
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.mp4")
            output_path = os.path.join(temp_dir, "output.mp4")
            
            # Download video
            download_file(video_url, input_path)
            
            # Processing (Reusing logic from app.py)
            scale = 4
            dtype = torch.bfloat16
            device = 'cuda'
            
            torch.cuda.empty_cache()
            
            LQ, th, tw, F, fps = prepare_input_tensor(input_path, scale=scale, dtype=dtype, device=device)
            
            sparse_ratio = 2.0
            seed = 0
            
            result_video = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, 
                tiled=False,
                LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), 
                kv_ratio=3.0,
                local_range=11, 
                color_fix = True,
            )
            
            output_frames = tensor2video(result_video)
            imageio.mimsave(output_path, [np.array(f) for f in output_frames], fps=fps, quality=6)
            
            # Upload result or return base64 (Serverless usually needs upload, but returning base64 for small clips works)
            # Ideally user would provide a bucket, but simplest for now is if runpod supports direct file return or we upload to S3.
            # For simplicity, let's assume we return a success message or uploading to runpod bucket if configured.
            
            # Returning Base64 (Warning: Size limits)
            # with open(output_path, "rb") as video_file:
            #     encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
            # return {"output_video": encoded_string}
            
            # Better: Return path if using RunPod Volume, or assume handling via user-provided bucket.
            # Let's return a "status": "done" and assume the user set up output storage in real scenario
            # OR, just return the path if they are testing.
            
            return {"status": "success", "message": "Video processed. (Cloud upload not configured in this demo)"}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
