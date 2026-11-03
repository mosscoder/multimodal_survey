"""Dense per-patch feature extraction for spatial maps. Native resolution:
1280x720 (the ground_image's released size, no resize/stretch) -> 80x45 = 3600
patches. Was 1024x768 -> 64x48 = 3072 (stretched the aspect ratio); superseded
by this native-res version per the rerun spec, item 1. Old file kept on disk
(grid1024x768.npy) for comparison, this writes to a new grid1280x720.npy path."""
import time
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

from timing_utils import log_stage, reset_vram

MODELS = {"general": "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "satellite": "facebook/dinov3-vitl16-pretrain-sat493m"}
splits = {"train": load_dataset("mpg-ranch/multimodal-survey", split="train")}

RES_W, RES_H = 1280, 720                      # native ground_image size, no resize
GRID_W, GRID_H = RES_W // 16, RES_H // 16     # 80 x 45 patch grid
P = GRID_W * GRID_H                           # 3,600 patches per frame

proc  = AutoImageProcessor.from_pretrained(MODELS["general"])   # normalization stats only
model = AutoModel.from_pretrained(MODELS["general"]).cuda().eval()
skip  = 1 + model.config.num_register_tokens                    # CLS + 4 registers precede the patches
mu_px = torch.tensor(proc.image_mean).view(1, 3, 1, 1).cuda()
sd_px = torch.tensor(proc.image_std).view(1, 3, 1, 1).cuda()


@torch.inference_mode()
def extract_grid(split, out_path, batch_size=48):
    G = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16,
                                  shape=(len(split), P, 768))
    import time; t0 = time.time()
    for i in range(0, len(split), batch_size):
        imgs = split[i:i + batch_size]["ground_image"]
        # native resolution: skip resize entirely when the image already matches (true
        # no-op, avoids any resampling-filter artifact from resizing to an identical size)
        prepped = [im.convert("RGB") for im in imgs]
        prepped = [im if im.size == (RES_W, RES_H) else im.resize((RES_W, RES_H)) for im in prepped]
        x = torch.stack([torch.from_numpy(np.asarray(im)) for im in prepped]
                        ).permute(0, 3, 1, 2).cuda().float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(pixel_values=(x - mu_px) / sd_px)
        G[i:i + batch_size] = h.last_hidden_state[:, skip:, :].float().cpu().numpy()
        if i % (batch_size * 20) == 0:
            print(f"  {i+len(imgs)}/{len(split)}  {(i+len(imgs))/(time.time()-t0):.0f} img/s", flush=True)
    G.flush()
    return G


out = f"feats/train.ground_image.general.grid{RES_W}x{RES_H}.npy"
reset_vram(); t_stage = time.time()
G = extract_grid(splits["train"], out)
log_stage("grid_extract_train", time.time() - t_stage, note=f"{len(splits['train'])} frames")
print(f"wrote {out}  shape {G.shape}  {G.nbytes/1e9:.1f} GB")

# informational: pooled grid vs the 768x768 resolution-study patch features (NOT expected to
# match -- different resolution + preprocessing -- just a magnitude/sanity readout)
gmean = np.asarray(G.mean(1), dtype=np.float32)
ref = np.load("feats/train.ground_image.general.r768.npy")[:, 768:]   # patch-mean half
d = np.linalg.norm(gmean - ref, axis=1) / np.linalg.norm(ref, axis=1)
print(f"G.mean(1) vs r768 patch-mean: relL2 {d.mean():.3f} (mean) {d.max():.3f} (max)  "
      f"-- large is expected, resolutions differ")
