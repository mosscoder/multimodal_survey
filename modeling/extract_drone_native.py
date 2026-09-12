"""Drone features at native resolution. drone_image is 312x312; ViT/16 needs a
multiple of 16, so use 320 (nearest, ~no resampling). Files tagged .r320.
"""
import numpy as np, torch, os, time
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

MODELS = {"general":   "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "satellite": "facebook/dinov3-vitl16-pretrain-sat493m"}
RES = 320
split = load_dataset("mpg-ranch/multimodal-survey", split="train")

@torch.inference_mode()
def extract(key, batch_size=64):
    proc  = AutoImageProcessor.from_pretrained(MODELS[key])
    model = AutoModel.from_pretrained(MODELS[key]).cuda().eval()
    skip  = 1 + model.config.num_register_tokens
    out, t0 = [], time.time()
    for i in range(0, len(split), batch_size):
        imgs = split[i:i + batch_size]["drone_image"]
        x = proc(images=imgs, return_tensors="pt", size={"height": RES, "width": RES}).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**x)
        cls   = h.pooler_output.float()
        patch = h.last_hidden_state[:, skip:, :].float().mean(1)
        out.append(torch.cat([cls, patch], 1).cpu().numpy())
        if i % (batch_size * 20) == 0:
            print(f"    {key}: {i+len(imgs)}/{len(split)}  {(i+len(imgs))/(time.time()-t0):.0f} img/s", flush=True)
    del model; torch.cuda.empty_cache()
    return np.concatenate(out)

os.makedirs("feats", exist_ok=True)
for key in ("satellite", "general"):
    p = f"feats/train.drone_image.{key}.r320.npy"
    if os.path.exists(p):
        print(f"skip {p}"); continue
    a = extract(key)
    np.save(p, a); print(f"wrote {p}  {a.shape}", flush=True)
