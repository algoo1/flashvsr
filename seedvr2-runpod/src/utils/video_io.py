
import os
import requests
import base64
import shutil
from urllib.parse import urlparse

def download_video(url: str, save_path: str):
    """Download video from URL to save_path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        raise ValueError(f"Failed to download video from {url}: {e}")

def save_base64_video(b64_string: str, save_path: str):
    """Save base64 string as video file."""
    try:
        # Handle data URI scheme if present
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        
        video_data = base64.b64decode(b64_string)
        with open(save_path, 'wb') as f:
            f.write(video_data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 video: {e}")

def encode_video_base64(video_path: str) -> str:
    """Read video file and return base64 string."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"{video_path} not found")
        
    with open(video_path, "rb") as video_file:
        video_data = video_file.read()
        base64_string = base64.b64encode(video_data).decode('utf-8')
        
    # Return as data URI? User might just want raw or data URI.
    # We'll return raw base64 usually, or let handler format it.
    return base64_string
