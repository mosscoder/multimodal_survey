import numpy as np, torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

MODELS = {"general":   "facebook/dinov3-vitb16-pretrain-lvd1689m",   # 768-d,  ground photographs
          "satellite": "facebook/dinov3-vitl16-pretrain-sat493m"}    # 1024-d, nadir drone crops
PLAN = [("ground_image", "general"),
        ("drone_image",  "satellite"),
        ("drone_image",  "general")]      # general backbone on crops, for the backbone comparison

splits = {"train": load_dataset("mpg-ranch/multimodal-survey", split="train")}
# The test split is not extracted here. It is opened once, in section 6, after tuning is complete.

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
        a = extract(split, column, key)
        np.save(p, a)
        print(f"wrote {p}  {a.shape}", flush=True)
    split.remove_columns(["ground_image", "drone_image"]).to_pandas().to_parquet(f"feats/{name}.meta.parquet")
    print(f"wrote feats/{name}.meta.parquet", flush=True)
