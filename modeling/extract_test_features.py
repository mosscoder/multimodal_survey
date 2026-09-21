"""Extract the three feature sets needed for the best config, on the TEST split
(never done before -- every prior extraction was train-only by design)."""
import numpy as np, torch, os, time
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

MODELS = {"general": "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "satellite": "facebook/dinov3-vitl16-pretrain-sat493m"}

split = load_dataset("mpg-ranch/multimodal-survey", split="test")
print(f"test split: {len(split)} frames", flush=True)
os.makedirs("feats", exist_ok=True)


@torch.inference_mode()
def extract_pooled(column, key, res, batch_size):
    proc = AutoImageProcessor.from_pretrained(MODELS[key])
    model = AutoModel.from_pretrained(MODELS[key]).cuda().eval()
    skip = 1 + model.config.num_register_tokens
    out, t0 = [], time.time()
    for i in range(0, len(split), batch_size):
        imgs = [im.convert("RGB") for im in split[i:i+batch_size][column]]
        x = proc(images=imgs, return_tensors="pt", size={"height": res, "width": res}).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**x)
        cls = h.pooler_output.float()
        patch = h.last_hidden_state[:, skip:, :].float().mean(1)
        out.append(torch.cat([cls, patch], 1).cpu().numpy())
        if i % (batch_size * 10) == 0:
            print(f"    {key} {column} @{res}: {i+len(imgs)}/{len(split)}  "
                  f"{(i+len(imgs))/(time.time()-t0):.0f} img/s", flush=True)
    del model; torch.cuda.empty_cache()
    return np.concatenate(out)


@torch.inference_mode()
def extract_grid(column, key, res_w, res_h, batch_size=48):
    proc = AutoImageProcessor.from_pretrained(MODELS[key])
    model = AutoModel.from_pretrained(MODELS[key]).cuda().eval()
    skip = 1 + model.config.num_register_tokens
    gw, gh = res_w // 16, res_h // 16
    P = gw * gh
    out_path = f"feats/test.{column}.{key}.grid{res_w}x{res_h}.npy"
    G = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(split), P, 768))
    t0 = time.time()
    for i in range(0, len(split), batch_size):
        imgs = [im.convert("RGB").resize((res_w, res_h)) for im in split[i:i+batch_size][column]]
        x = torch.stack([torch.from_numpy(np.asarray(im)) for im in imgs]).permute(0, 3, 1, 2).cuda().float() / 255
        mu = torch.tensor(proc.image_mean).view(1, 3, 1, 1).cuda()
        sd = torch.tensor(proc.image_std).view(1, 3, 1, 1).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(pixel_values=(x - mu) / sd)
        G[i:i+len(imgs)] = h.last_hidden_state[:, skip:, :].float().cpu().numpy()
        if i % (batch_size * 10) == 0:
            print(f"    grid {column}: {i+len(imgs)}/{len(split)}  "
                  f"{(i+len(imgs))/(time.time()-t0):.0f} img/s", flush=True)
    G.flush()
    del model; torch.cuda.empty_cache()
    print(f"wrote {out_path}  shape {G.shape}", flush=True)


# 1. drone alone (224px CLS + patch-mean) -- fast
p = "feats/test.drone_image.satellite.npy"
if not os.path.exists(p):
    a = extract_pooled("drone_image", "satellite", 224, 96)
    np.save(p, a); print(f"wrote {p}  {a.shape}", flush=True)
else:
    print(f"skip {p} (exists)")

# 2. ground global-pool, 768px CLS + patch-mean -- fast
p = "feats/test.ground_image.general.r768.npy"
if not os.path.exists(p):
    a = extract_pooled("ground_image", "general", 768, 48)
    np.save(p, a); print(f"wrote {p}  {a.shape}", flush=True)
else:
    print(f"skip {p} (exists)")

# 3. ground dense grid, 1024x768 -- the expensive one
p = "feats/test.ground_image.general.grid1024x768.npy"
if not os.path.exists(p):
    extract_grid("ground_image", "general", 1024, 768)
else:
    print(f"skip {p} (exists)")

print("\ndone.")
