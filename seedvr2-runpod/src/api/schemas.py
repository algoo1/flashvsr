
from pydantic import BaseModel
from typing import Optional, Literal, Dict, Any

class SeedVRInput(BaseModel):
    video: str
    target_width: int = 1280
    target_height: int = 720
    quality: Literal["fast", "balanced", "best"] = "balanced"

class SeedVROutput(BaseModel):
    status: str
    output_video: str
    processing_time: float
    metadata: Dict[str, Any]
