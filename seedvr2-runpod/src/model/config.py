
import os

# Configuration Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check nested vs sibling
_repo_nested = os.path.join(PROJECT_ROOT, "seedvr_repo")
_repo_sibling = os.path.join(os.path.dirname(PROJECT_ROOT), "seedvr_repo")

if os.path.exists(_repo_nested):
    SEEDVR_REPO_PATH = _repo_nested
elif os.path.exists(_repo_sibling):
    SEEDVR_REPO_PATH = _repo_sibling
else:
    SEEDVR_REPO_PATH = _repo_sibling # Default for Docker structure
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "ckpts")
MODEL_CHECKPOINT_NAME = "seedvr2_ema_3b.pth"
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, MODEL_CHECKPOINT_NAME)

# Inference Defaults
DEFAULT_RESOLUTION = (720, 1280) # H, W
DEFAULT_BATCH_SIZE = 5
DEFAULT_SP_SIZE = 1
