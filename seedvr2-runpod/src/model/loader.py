
import os
import sys
import torch
import shutil
from huggingface_hub import snapshot_download

# Add SeedVR repo to path
# In Docker: /app/seedvr_repo (sibling to /app/seedvr2-runpod)
# Locally: might be in project root or sibling
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Check: seedvr2-runpod/seedvr_repo
REPO_ROOT_NESTED = os.path.abspath(os.path.join(CURRENT_DIR, '../../seedvr_repo'))
# Check: seedvr2-runpod/../seedvr_repo (Sibling)
REPO_ROOT_SIBLING = os.path.abspath(os.path.join(CURRENT_DIR, '../../../seedvr_repo'))

if os.path.exists(os.path.join(REPO_ROOT_NESTED, 'projects')):
    REPO_ROOT = REPO_ROOT_NESTED
elif os.path.exists(os.path.join(REPO_ROOT_SIBLING, 'projects')):
    REPO_ROOT = REPO_ROOT_SIBLING
else:
    # Fallback/Default
    REPO_ROOT = REPO_ROOT_SIBLING 

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

def ensure_seedvr_repo():
    """Ensure SeedVR repo is available."""
    if not os.path.exists(os.path.join(REPO_ROOT, 'projects')):
        print(f"SeedVR repo not found at {REPO_ROOT}. Please clone it there.")
        # In RunPod/Docker, this should be handled by Dockerfile.
        # For local test, we might want to clone it if missing?
        # subprocess.run(["git", "clone", "https://github.com/ByteDance-Seed/SeedVR", REPO_ROOT])
        raise FileNotFoundError(f"SeedVR repo not found at {REPO_ROOT}")

# --- APEX MOCK START ---
# Mock apex to avoid complex installation if it's missing.
# SeedVR uses apex.normalization.FusedRMSNorm, which can be replaced by torch.nn.RMSNorm in Torch 2.x
import sys
import types
try:
    import apex
except ImportError:
    print("DEBUG: Apex not found. Mocking apex.normalization.FusedRMSNorm with torch.nn.RMSNorm.")
    
    # Create fake module structure
    apex_module = types.ModuleType("apex")
    apex_norm_module = types.ModuleType("apex.normalization")
    
    # Mock FusedRMSNorm using torch.nn.RMSNorm (checking signature compatibility)
    class FusedRMSNorm(torch.nn.RMSNorm):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
            super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    apex_norm_module.FusedRMSNorm = FusedRMSNorm
    apex_module.normalization = apex_norm_module
    
    # Inject into sys.modules
    sys.modules["apex"] = apex_module
    sys.modules["apex.normalization"] = apex_norm_module
# --- APEX MOCK END ---

from projects.video_diffusion_sr.infer import VideoDiffusionInfer
from common.config import load_config
from common.distributed import init_torch
from common.distributed.advanced import init_sequence_parallel
from omegaconf import OmegaConf
import datetime
import torch.distributed as dist

def load_model(model_path: str = None, device: str = "cuda", use_fp8: bool = True):
    """
    Load SeedVR2-3B model with optimizations.
    
    Args:
        model_path: Path to the checkpoint directory (or None to download)
        device: 'cuda'
        use_fp8: Whether to use quantization (implicit in model loading usually)
    """
    # 0. Initialize Distributed Environment (Critical for RunPod)
    # RunPod Serverless doesn't set these, but SeedVR expects distributed group for barrier()
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        
        try:
            dist.init_process_group(
                backend="nccl", 
                init_method="env://", 
                world_size=1, 
                rank=0
            )
            print("DEBUG: Manually initialized process group (single device).")
        except Exception as e:
            print(f"DEBUG: Failed to manual init process group: {e}")

    ensure_seedvr_repo()
    
    # 1. Download/Locate Weights
    if model_path is None:
        model_path = "ckpts/seedvr2_ema_3b.pth" # Default relative to REPO_ROOT or CWD
        
    if not os.path.exists(model_path):
        print(f"Downloading SeedVR2-3B to {model_path}...")
        # We need the specific checkpoint file. 
        # The user snippet suggests: snapshot_download(repo_id="ByteDance-Seed/SeedVR2-3B")
        # But we need to place it where the config expects it or pass path.
        save_dir = os.path.dirname(model_path)
        if not save_dir: save_dir = "ckpts/"
        
        # Configure cache dir
        cache_dir = os.path.join(save_dir, "cache")
        snapshot_download(
            repo_id="ByteDance-Seed/SeedVR2-3B",
            local_dir=save_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.pth", "*.json", "*.safetensors"]
        )
        # Note: The file might be named differently in the repo. 
        # We assume 'seedvr2_ema_3b.pth' is the file we want according to README.

    # 2. Configure Runner
    # Config is inside the repo
    config_path = os.path.join(REPO_ROOT, 'configs_3b', 'main.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
        
    # CRITICAL FIX: The config files use relative paths. 
    # We must switch CWD to the repo root for them to resolve correctly.
    original_cwd = os.getcwd()
    
    print(f"DEBUG: Current CWD: {original_cwd}")
    print(f"DEBUG: REPO_ROOT (Target): {REPO_ROOT}")
    if os.path.exists(REPO_ROOT):
        print(f"DEBUG: REPO_ROOT contents: {os.listdir(REPO_ROOT)}")
    else:
        print("DEBUG: REPO_ROOT does not exist!")

    os.chdir(REPO_ROOT)
    print(f"DEBUG: New CWD: {os.getcwd()}")

    try:
        config = load_config(config_path)
        runner = VideoDiffusionInfer(config)
        OmegaConf.set_readonly(runner.config, False)
        
        # Init Torch (sets device)
        try:
            init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
        except Exception:
            pass 

        # 3. Load Checkpoint
        # Handle absolute path logic carefully
        if os.path.isabs(model_path):
             abs_model_path = width_model_path # Typo in thought, strict code below
             abs_model_path = model_path
        else:
             # If it was relative to the ORIGINAL CWD, we need to join it
             abs_model_path = os.path.join(original_cwd, model_path)
        
        print(f"DEBUG: Loading checkpoint from {abs_model_path}")

        
        runner.configure_dit_model(device=device, checkpoint=abs_model_path)
        
        # 4. Load VAE
        runner.configure_vae_model()
        
        # 5. Optimizations
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

        # Apply torch.compile if requested
        try:
            if hasattr(runner, 'dit'):
                print("Compiling DiT model...")
                runner.dit = torch.compile(runner.dit, mode="reduce-overhead")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    finally:
        os.chdir(original_cwd)

    return runner
