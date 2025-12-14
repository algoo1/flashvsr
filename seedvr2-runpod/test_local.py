
import os
import sys
import argparse
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model.loader import load_model
from src.model.inference import process_video

def test_local_inference(video_path):
    print(f"Testing local inference with video: {video_path}")
    
    if not os.path.exists(video_path):
        # Create a dummy video if none exists
        print("Creating dummy input video...")
        import cv2
        import numpy as np
        height, width = 720, 1280 # 720p input ( SeedVR upscales? or restores? It restores)
        # SeedVR optimized for 720p/1080p restoration
        # Let's clean degradation? No, just random noise video to test pipeline.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        for _ in range(30): # 1 second
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            video.write(frame)
        video.release()
        print(f"Created {video_path}")

    # Load Model
    print("Loading model...")
    runner = load_model()
    
    # Process
    output_path = "test_output.mp4"
    if os.path.exists(output_path):
        os.remove(output_path)
        
    print("Running process_video...")
    start = time.time()
    process_video(
        runner,
        video_path,
        output_path,
        target_resolution=(1280, 720),
        quality_mode="balanced"
    )
    print(f"Inference done in {time.time() - start:.2f}s")
    
    if os.path.exists(output_path):
        print("SUCCESS: Output video created.")
    else:
        print("FAILURE: Output video not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="test_input.mp4")
    args = parser.parse_args()
    
    test_local_inference(args.video)
