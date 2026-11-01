"""Recompute drone CLS+patch-mean features (train + test) from the re-uploaded
dataset (308x308 drone images, was 312x312). Old feats/{split}.drone_image.satellite.npy
files were computed from the stale images -- this overwrites them in place since
it's a correctness fix, not a new experiment variant."""
import numpy as np, torch, os, time
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

MODEL = "facebook/dinov3-vitl16-pretrain-sat493m"


@torch.inference_mode()
def extract(split, batch_size=128):
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).cuda().eval()
    skip = 1 + model.config.num_register_tokens
    out, t0 = [], time.time()
    for i in range(0, len(split), batch_size):
        imgs = split[i:i + batch_size]["drone_image"]
        x = proc(images=imgs, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**x)
        cls = h.pooler_output.float()
        patch = h.last_hidden_state[:, skip:, :].float().mean(1)
        out.append(torch.cat([cls, patch], 1).cpu().numpy())
        if i % (batch_size * 10) == 0:
            print(f"    {i+len(imgs)}/{len(split)}  {(i+len(imgs))/(time.time()-t0):.0f} img/s", flush=True)
    del model; torch.cuda.empty_cache()
    return np.concatenate(out)


os.makedirs("feats", exist_ok=True)
for name in ["train", "test"]:
    split = load_dataset("mpg-ranch/multimodal-survey", split=name)
    print(f"{name}: {len(split)} frames, drone_image size {split[0]['drone_image'].size}", flush=True)
    a = extract(split)
    p = f"feats/{name}.drone_image.satellite.npy"
    np.save(p, a)
    print(f"wrote {p}  {a.shape}", flush=True)

print("\ndone.")
