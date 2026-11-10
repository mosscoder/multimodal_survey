"""Whole-image features of the training-split drone images.

Each 308 by 308 drone image is resized to 224 by 224 by the model's image
processor and goes through the frozen satellite DINOv3 ViT-L/16. Writes
feats/train.drone_image.satellite.npy, which stores per image the CLS token and
the mean patch token side by side (2,048 values). The drone classifier uses
the first 1,024, the CLS token. Requires a CUDA GPU.
"""
import time
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

from timing_utils import log_stage, reset_vram

MODELS = {"satellite": "facebook/dinov3-vitl16-pretrain-sat493m"}    # 1024-d, nadir drone images
PLAN = [("drone_image", "satellite")]

splits = {"train": load_dataset("mpg-ranch/multimodal-survey", split="train")}
# The test split is not extracted here (see extract_test_features.py).

@torch.inference_mode()
def extract(split, column, key, batch_size=128):
    proc  = AutoImageProcessor.from_pretrained(MODELS[key])
    model = AutoModel.from_pretrained(MODELS[key]).cuda().eval()
    skip  = 1 + model.config.num_register_tokens              # CLS + register tokens precede patches
    out = []
    for i in range(0, len(split), batch_size):
        imgs = split[i:i + batch_size][column]
        x = proc(images=imgs, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**x)
        cls   = h.pooler_output.float()                        # (B, D)  CLS = last_hidden_state[:, 0] after the final norm
        patch = h.last_hidden_state[:, skip:, :].float().mean(1)   # (B, D)  mean over the 196 patch tokens at 224 px
        out.append(torch.cat([cls, patch], 1).cpu().numpy())   # (B, 2D)
    return np.concatenate(out)

import os
os.makedirs("feats", exist_ok=True)
for name, split in splits.items():
    for column, key in PLAN:
        p = f"feats/{name}.{column}.{key}.npy"
        reset_vram(); t_stage = time.time()
        a = extract(split, column, key)
        log_stage(f"cls_extract_{name}_{column}_{key}", time.time() - t_stage, note=f"{len(split)} frames")
        np.save(p, a)
        print(f"wrote {p}  {a.shape}", flush=True)
