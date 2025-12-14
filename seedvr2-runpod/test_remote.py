import os
import time
import requests
import base64
import json
import sys

# Configuration
API_KEY = os.environ.get("RUNPOD_API_KEY", "YOUR_API_KEY_HERE")
ENDPOINT_ID = "uiwcqbn98e87bn"
VIDEO_PATH = r"C:\Users\Doctor Laptop\Desktop\okok\03.mp4"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

def encode_video(path):
    print(f"Reading video from {path}...")
    with open(path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

def submit_job(encoded_video):
    url = f"{BASE_URL}/run"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "video": encoded_video,
            "target_width": 1280,
            "target_height": 720,
            "quality": "fast" 
        }
    }
    print(f"Submitting job to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def check_status(job_id):
    url = f"{BASE_URL}/status/{job_id}"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.get(url, headers=headers)
    return response.json()

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    # 1. Prepare Input
    b64_video = encode_video(VIDEO_PATH)
    
    # 2. Submit Job
    submit_res = submit_job(b64_video)
    job_id = submit_res.get("id")
    print(f"Job submitted! ID: {job_id}")
    
    # 3. Poll for completion
    while True:
        status_res = check_status(job_id)
        status = status_res.get("status")
        print(f"Status: {status}")
        
        if status == "COMPLETED":
            output = status_res.get("output", {})
            # Handle potential different output structures (direct dict or string)
            if isinstance(output, str):
                 print(f"Raw output: {output}")
                 # Try parsing if it's a stringified json, though handler usually returns dict
                 try:
                     output = json.loads(output)
                 except:
                     pass
            
            output_b64 = output.get("output_video")
            if output_b64:
                # Remove header if present (data:video/mp4;base64,...)
                if "," in output_b64:
                    output_b64 = output_b64.split(",")[1]
                
                output_path = "result_03_runpod.mp4"
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(output_b64))
                print(f"SUCCESS! Video saved to {os.path.abspath(output_path)}")
            else:
                print("Error: No output video found in response.")
                print(json.dumps(output, indent=2))
            break
            
        elif status == "FAILED":
            print("Job FAILED.")
            print(json.dumps(status_res, indent=2))
            break
            
        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(5)
        else:
            print(f"Unknown status: {status}")
            time.sleep(5)

if __name__ == "__main__":
    main()
