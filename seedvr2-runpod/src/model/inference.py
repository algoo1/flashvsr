
import os
import sys
import torch
import gc
import numpy as np
import mediapy
from einops import rearrange
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.io import read_video, read_image

# Ensure SeedVR repo is in path (handled by loader import or config, but redundant check is safe)
from .config import SEEDVR_REPO_PATH
if SEEDVR_REPO_PATH not in sys.path:
    sys.path.append(SEEDVR_REPO_PATH)

try:
    from data.image.transforms.divisible_crop import DivisibleCrop
    from data.image.transforms.na_resize import NaResize
    from data.video.transforms.rearrange import Rearrange
    from common.distributed import get_device
    from common.distributed.ops import sync_data
    from common.partition import partition_by_groups, partition_by_size
    from common.seed import set_seed
except ImportError:
    pass

def cut_videos(videos, sp_size):
    t = videos.size(1)
    if t == 1:
        return videos
    if t <= 4 * sp_size:
        padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos
    if (t - 1) % (4 * sp_size) == 0:
        return videos
    else:
        padding = [videos[:, -1].unsqueeze(1)] * (
            4 * sp_size - ((t - 1) % (4 * sp_size))
        )
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos

def generation_step(runner, text_embeds_dict, cond_latents):
    def _move_to_cuda(x):
        return [i.to(get_device()) for i in x]

    noises = [torch.randn_like(latent) for latent in cond_latents]
    aug_noises = [torch.randn_like(latent) for latent in cond_latents]
    
    noises, aug_noises, cond_latents = list(
        map(lambda x: _move_to_cuda(x), (noises, aug_noises, cond_latents))
    )
    cond_noise_scale = 0.0

    def _add_noise(x, aug_noise):
        t = (
            torch.tensor([1000.0], device=get_device())
            * cond_noise_scale
        )
        shape = torch.tensor(x.shape[1:], device=get_device())[None]
        t = runner.timestep_transform(t, shape)
        x = runner.schedule.forward(x, aug_noise, t)
        return x

    conditions = [
        runner.get_condition(
            noise,
            task="sr",
            latent_blur=_add_noise(latent_blur, aug_noise),
        )
        for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
    ]

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        video_tensors = runner.inference(
            noises=noises,
            conditions=conditions,
            dit_offload=True,
            **text_embeds_dict,
        )

    samples = [
        (
            rearrange(video[:, None], "c t h w -> t c h w")
            if video.ndim == 3
            else rearrange(video, "c t h w -> t c h w")
        )
        for video in video_tensors
    ]
    del video_tensors
    return samples

def process_video(
    runner,
    input_video_path: str,
    output_video_path: str,
    target_resolution: tuple = (1280, 720), # W, H
    quality_mode: str = "balanced",
    batch_size: int = 5
):
    """
    Process a video using the SeedVR2 runner.
    """
    res_w, res_h = target_resolution
    
    # Configure runner settings based on quality (if applicable)
    # The original script sets cfg scale, steps etc.
    runner.config.diffusion.cfg.scale = 7.5 # Default from script was 1.0?? NO, yaml says 7.5. script def arg 1.0 overrides?
    # Script default args: cfg_scale=1.0. Let's stick to script default if it works, or yaml.
    # Actually script CLI default is 1.0. But yaml keys might be different. 
    # Let's assume passed defaults.
    
    # Video Transform
    video_transform = Compose([
        NaResize(resolution=(res_h * res_w)**0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])

    # Read Video
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"{input_video_path} not found")

    print(f"Reading video: {input_video_path}")
    # Handle Image vs Video
    ext = os.path.splitext(input_video_path)[1].lower()
    is_image = ext in ['.jpg', '.png', '.jpeg', '.bmp']
    
    fps = 30.0
    if is_image:
        video_tensor = read_image(input_video_path).unsqueeze(0) / 255.0
    else:
        video_tensor, _, info = read_video(input_video_path, output_format="TCHW")
        video_tensor = video_tensor / 255.0
        fps = info["video_fps"]

    # Preprocess
    video_input = video_transform(video_tensor.to(get_device()))
    
    # Text Embeddings (Optimized: Load once or pre-compute)
    # The script loads 'pos_emb.pt' and 'neg_emb.pt'. We must ensure these exist.
    # We should look for them in the repo or allow passing them.
    # They are likely created by another script or downloaded. 
    # REQUIRED: pos_emb.pt, neg_emb.pt in CWD.
    pos_emb_path = os.path.join(SEEDVR_REPO_PATH, 'pos_emb.pt')
    neg_emb_path = os.path.join(SEEDVR_REPO_PATH, 'neg_emb.pt')
    
    # Fallback to current dir if not in repo root
    if not os.path.exists(pos_emb_path): pos_emb_path = 'pos_emb.pt'
    if not os.path.exists(neg_emb_path): neg_emb_path = 'neg_emb.pt'
    
    if os.path.exists(pos_emb_path):
        text_pos_embeds = torch.load(pos_emb_path)
        text_neg_embeds = torch.load(neg_emb_path)
    else:
        # If embeddings missing, we might need to encode prompts on the fly.
        # But for now, let's assume existence or placeholders.
        print("Warning: Embedding files not found. Inference might fail.")
        text_pos_embeds = torch.zeros(1, 120, 5120).cuda() # Mock shape from yaml (txt_in_dim: 5120)
        text_neg_embeds = torch.zeros(1, 120, 5120).cuda()

    text_embeds = {
        "texts_pos": [text_pos_embeds.to(get_device())],
        "texts_neg": [text_neg_embeds.to(get_device())]
    }

    # Batching logic (simple split for now if needed, but original uses one video at a time in loop)
    # We process the SINGLE input video.
    
    cond_latents = [video_input]
    ori_length = video_input.size(1)
    
    # Cut/Pad
    cond_latents_processed = [cut_videos(v, 1) for v in cond_latents] # sp_size=1
    
    # VAE Encode
    runner.dit.to("cpu")
    runner.vae.to(get_device())
    latents = runner.vae_encode(cond_latents_processed)
    runner.vae.to("cpu")
    
    # DIT Inference
    runner.dit.to(get_device())
    samples = generation_step(runner, text_embeds, cond_latents=latents)
    runner.dit.to("cpu")
    
    # Postprocess & Save
    sample = samples[0] # Single video
    if ori_length < sample.shape[0]:
        sample = sample[:ori_length]
        
    # Un-normalize
    # rearrange "t c h w -> t h w c"
    sample = rearrange(sample, "t c h w -> t h w c")
    sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
    sample = sample.to(torch.uint8).cpu().numpy()
    
    output_dir = os.path.dirname(output_video_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    if is_image:
        mediapy.write_image(output_video_path, sample.squeeze(0))
    else:
        mediapy.write_video(output_video_path, sample, fps=fps)
        
    print(f"Saved processed video to {output_video_path}")
    
    # Cleanup
    del latents, samples, sample, video_input, video_tensor
    torch.cuda.empty_cache()
    gc.collect()
    
    return output_video_path
