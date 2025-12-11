import sys
import os
import shutil
import tempfile
import torch
import numpy as np
import gradio as gr
from PIL import Image
import imageio
from tqdm import tqdm
from einops import rearrange
import subprocess

# Ensure the repository root is in python path
sys.path.append(os.getcwd())

# Try importing diffsynth
try:
    from diffsynth import ModelManager, FlashVSRFullPipeline
    from utils.utils import Causal_LQ4x_Proj
except ImportError:
    print("Could not import FlashVSR modules. Ensure you are running from the repository root.")
    sys.exit(1)

def tensor2video(frames: torch.Tensor):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def natural_key(name: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0                                              # CHW in [-1,1]
    return t.to(dtype)

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def upscale_then_center_crop(img: Image.Image, scale: int, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW, sH = w0 * scale, h0 * scale
    up = img.resize((sW, sH), Image.BICUBIC)
    l = max(0, (sW - tW) // 2); t = max(0, (sH - tH) // 2)
    return up.crop((l, t, l + tW, t + tH))

def prepare_input_tensor(path: str, scale: int = 4, dtype=torch.bfloat16, device='cuda'):
    # (Same implementation as original script, simplified for video only scenario mostly)
    if is_video(path):
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size
        
        meta = {}
        try:
             meta = rdr.get_meta_data()
        except Exception:
             pass
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf, int) and nf > 0:
                    return nf
            except Exception:
                pass
            try:
                return r.count_frames()
            except:
                n=0
                try:
                    while True: r.get_data(n); n+=1
                except: return n

        total = count_frames(rdr)
        
        # Limit frames for demo if too long to avoid OOM or timeout? 
        # For now keep as is.
        if total <= 0:
             rdr.close()
             raise RuntimeError(f"Cannot read frames from {path}")

        print(f"Loading {path}: {w0}x{h0}, {total} frames, {fps} fps")
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        
        idx = list(range(total)) + [total - 1] * 4
        F = largest_8n1_leq(len(idx))
        idx = idx[:F]
        
        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try: rdr.close()
            except: pass
            
        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)
        return vid, tH, tW, F, fps

    raise ValueError(f"Unsupported input: {path}")

# Pipeline Global
pipe = None

def init_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    print("Initializing FlashVSR pipeline...")
    base_path = "./FlashVSR-v1.1"
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Model path {base_path} not found.")

    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([
        os.path.join(base_path, "diffusion_pytorch_model_streaming_dmd.safetensors"),
        os.path.join(base_path, "Wan2.1_VAE.pth"),
    ])
    pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    
    LQ_proj_in_path = os.path.join(base_path, "LQ_proj_in.ckpt")
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)

    pipe.denoising_model().LQ_proj_in.to('cuda')
    pipe.vae.model.encoder = None
    pipe.vae.model.conv1 = None
    pipe.to('cuda')
    # pipe.enable_vram_management(num_persistent_param_in_dit=None) 
    # Disable vram management for speed if enough VRAM (A100), or enable if OOM.
    # The original script enabled it.
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit","vae"])
    print("Pipeline initialized.")
    return pipe

def process_video(input_video_path):
    if not input_video_path:
        return None
    
    try:
        global pipe
        if pipe is None:
            pipe = init_pipeline()

        scale = 4
        dtype = torch.bfloat16
        device = 'cuda'
        
        torch.cuda.empty_cache()
        
        LQ, th, tw, F, fps = prepare_input_tensor(input_video_path, scale=scale, dtype=dtype, device=device)
        
        # Parameters from script
        sparse_ratio = 2.0
        seed = 0
        
        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, 
            tiled=False,
            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), 
            kv_ratio=3.0,
            local_range=11, 
            color_fix = True,
        )
        
        output_frames = tensor2video(video)
        
        output_path = "output_flashvsr.mp4"
        imageio.mimsave(output_path, [np.array(f) for f in output_frames], fps=fps, quality=6)
        
        return output_path
    
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

with gr.Blocks(title="FlashVSR v1.1 Demo") as demo:
    gr.Markdown("# FlashVSR v1.1: Real-Time Diffusion-Based Streaming Video Super-Resolution")
    gr.Markdown("Upload a low-resolution video to upscale it (4x). Requires A100/A800 GPU.")
    
    with gr.Row():
        input_video = gr.Video(label="Input Low-Res Video")
        output_video = gr.Video(label="Super-Resolved Output")
    
    submit_btn = gr.Button("Upscale Video")
    
    submit_btn.click(
        fn=process_video,
        inputs=input_video,
        outputs=output_video
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
