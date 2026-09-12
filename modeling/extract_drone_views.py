"""Drone sub-crop views, satellite backbone, training split.

drone_image is 312x312 = 78 px/m over the 4x4 m crop. Each view is a crop of that,
then resized to 320 (native-ish, matching feats/*.r320). Bottom crops are anchored
on the antenna edge (row 0 = antenna, bearing up => 0..N m ahead).
"""
import numpy as np, torch, os, time
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

MODELS = {"general": "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "satellite": "facebook/dinov3-vitl16-pretrain-sat493m"}
RES = 320
PX_PER_M = 78


def center_crop(img, size):
    w, h = img.size; l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))

def bottom_center_crop(img, size):
    w, h = img.size; l = (w - size) // 2
    return img.crop((l, h - size, l + size, h))

VIEWS = {"full":     lambda im: im,
         "center3m": lambda im: center_crop(im, 3 * PX_PER_M),
         "center2m": lambda im: center_crop(im, 2 * PX_PER_M),
         "bottom3m": lambda im: bottom_center_crop(im, 3 * PX_PER_M),
         "bottom2m": lambda im: bottom_center_crop(im, 2 * PX_PER_M)}

split = load_dataset("mpg-ranch/multimodal-survey", split="train")


@torch.inference_mode()
def extract_view(key, view, batch_size=96):
    proc  = AutoImageProcessor.from_pretrained(MODELS[key])
    model = AutoModel.from_pretrained(MODELS[key]).cuda().eval()
    skip  = 1 + model.config.num_register_tokens
    out, t0 = [], time.time()
    for i in range(0, len(split), batch_size):
        imgs = [VIEWS[view](im) for im in split[i:i + batch_size]["drone_image"]]
        x = proc(images=imgs, return_tensors="pt", size={"height": RES, "width": RES}).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**x)
        out.append(torch.cat([h.pooler_output.float(),
                              h.last_hidden_state[:, skip:, :].float().mean(1)], 1).cpu().numpy())
        if i % (batch_size * 20) == 0:
            print(f"    {view}: {i+len(imgs)}/{len(split)}  {(i+len(imgs))/(time.time()-t0):.0f} img/s", flush=True)
    del model; torch.cuda.empty_cache()
    return np.concatenate(out)


os.makedirs("feats", exist_ok=True)
for view in VIEWS:
    p = f"feats/train.drone_image.satellite.{view}.npy"
    if os.path.exists(p):
        print(f"skip {p}"); continue
    a = extract_view("satellite", view)
    np.save(p, a); print(f"wrote {p}  {a.shape}", flush=True)
