# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import comfy.model_management as mm
from PIL import Image
import numpy as np
from comfy.utils import common_upscale

cur_path = os.path.dirname(os.path.abspath(__file__))

# SenseNova image-token grid factor (patch_size * merge); width/height must be divisible by this.
IMAGE_GRID_FACTOR = 32


def resolve_output_wh(
    width: int,
    height: int,
    target_pixels: str,
    supported_resolutions: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    """Resolve output (W, H): custom pixels when both > 0, else ``target_pixels`` preset."""
    if width > 0 or height > 0:
        if width <= 0 or height <= 0:
            raise ValueError(
                "Set both width and height for a custom resolution, or leave both at 0 to use target_pixels."
            )
        w, h = int(width), int(height)
        if w % IMAGE_GRID_FACTOR or h % IMAGE_GRID_FACTOR:
            raise ValueError(
                f"Custom width/height must be multiples of {IMAGE_GRID_FACTOR}, got {w}x{h}."
            )
        return w, h
    if target_pixels not in supported_resolutions:
        raise ValueError(
            f"Unknown target_pixels {target_pixels!r}; supported: {list(supported_resolutions)}"
        )
    return supported_resolutions[target_pixels]


def throw_if_processing_interrupted() -> None:
    """Propagate ComfyUI queue cancel into long-running SenseNova inference."""
    mm.throw_exception_if_processing_interrupted()


def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
    except: pass
    mm.soft_empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")

def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list

def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "bilinear", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "bilinear", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img


def map_0_1_to_neg1_1(t):

    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()

    try:
        vmax = float(t.max())
    except Exception:
        vmax = 1.0
    if vmax > 2.0:
        t = t / 255.0
    try:
        vmin = float(t.min())
        vmax = float(t.max())
    except Exception:
        vmin, vmax = -1.0, 1.0
    if vmin >= 0.0 and vmax <= 1.1:
        t = t * 2.0 - 1.0
    return t

def map_neg1_1_to_0_1(t):
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()
    t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    return t
