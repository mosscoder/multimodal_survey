"""Horizontal-flip augmentation features for the best config.

Flip ground_image AND its paired drone_image left-right (drone is bearing-up, so a
horizontal mirror keeps 'forward' pointing up -- geometrically consistent). Same labels.
  ground_image  general   @768  -> feats/train.ground_image.general.r768.hflip.npy
  drone_image   satellite @320  -> feats/train.drone_image.satellite.full.hflip.npy
"""
import numpy as np, torch, os, time
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel
from PIL import ImageOps

MODELS = {"general": "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "satellite": "facebook/dinov3-vitl16-pretrain-sat493m"}
split = load_dataset("mpg-ranch/multimodal-survey", split="train")


@torch.inference_mode()
def extract(column, key, res, batch_size):
    proc  = AutoImageProcessor.from_pretrained(MODELS[key])
    model = AutoModel.from_pretrained(MODELS[key]).cuda().eval()
    skip  = 1 + model.config.num_register_tokens
    out, t0 = [], time.time()
    for i in range(0, len(split), batch_size):
        imgs = [ImageOps.mirror(im) for im in split[i:i + batch_size][column]]
        x = proc(images=imgs, return_tensors="pt", size={"height": res, "width": res}).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**x)
        cls   = h.pooler_output.float()
        patch = h.last_hidden_state[:, skip:, :].float().mean(1)
        out.append(torch.cat([cls, patch], 1).cpu().numpy())
        if i % (batch_size * 20) == 0:
            print(f"    {key} {column}: {i+len(imgs)}/{len(split)}  "
                  f"{(i+len(imgs))/(time.time()-t0):.0f} img/s", flush=True)
    del model; torch.cuda.empty_cache()
    return np.concatenate(out)


os.makedirs("feats", exist_ok=True)
jobs = [("ground_image", "general", 768, 48, "feats/train.ground_image.general.r768.hflip.npy"),
        ("drone_image", "satellite", 320, 96, "feats/train.drone_image.satellite.full.hflip.npy")]
for column, key, res, bs, path in jobs:
    if os.path.exists(path):
        print(f"skip {path}"); continue
    a = extract(column, key, res, bs)
    np.save(path, a); print(f"wrote {path}  {a.shape}", flush=True)
