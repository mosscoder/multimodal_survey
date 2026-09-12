"""Extract frozen DINOv3 CLS features for the multimodal-survey dataset.

Runs a one-time forward pass with a frozen backbone and caches the pooled CLS
vector (``pooler_output``, i.e. ``last_hidden_state[:, 0, :]`` after the final
norm) for every frame. Fitting a linear probe on the cached vectors is done
separately by ``fit_probe.py``.

Full matrix: 2 backbones x 2 image columns x 3 splits.

  vitb_general    facebook/dinov3-vitb16-pretrain-lvd1689m   (768-d, natural imagery)
  vitl_satellite  facebook/dinov3-vitl16-pretrain-sat493m    (1024-d, nadir aerial)

Outputs (git-ignored, *.npy):
  modeling/features/<model>__<col>__<split>.npy   float32 [N, D], row-aligned to
  modeling/features/labels__<split>.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

DATASET = "mpg-ranch/multimodal-survey"
MODELS = {
    "vitb_general": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl_satellite": "facebook/dinov3-vitl16-pretrain-sat493m",
}
IMAGE_COLS = ["ground_image", "drone_image"]
SPLITS = ["train", "test", "train_tier_2"]

# 8 target species, in dataset ClassLabel order.
SPECIES = [
    "Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
    "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
    "Bromus tectorum", "Poa bulbosa",
]

OUT_DIR = Path(__file__).parent / "features"


def write_labels(ds) -> None:
    """One parquet per split: frame_id, metadata, multi-hot species, artifact_level."""
    for split in SPLITS:
        d = ds[split]
        rows = d.remove_columns([c for c in d.column_names if c in IMAGE_COLS])
        df = rows.to_pandas()
        multihot = np.zeros((len(df), len(SPECIES)), dtype=np.int8)
        for i, labs in enumerate(df["species"]):
            for j in labs:
                multihot[i, j] = 1
        for j, name in enumerate(SPECIES):
            df[f"sp_{j}"] = multihot[:, j]
        df["split"] = split
        keep = (["frame_id", "split", "site", "strip", "run_id", "artifact_level"]
                + [f"sp_{j}" for j in range(len(SPECIES))])
        df[keep].to_parquet(OUT_DIR / f"labels__{split}.parquet", index=False)
        print(f"  labels__{split}.parquet  {len(df)} rows  "
              f"species pos/frame mean {multihot.sum(1).mean():.2f}")


@torch.inference_mode()
def extract(ds, model_key: str, col: str, batch_size: int, bf16: bool) -> None:
    name = MODELS[model_key]
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).cuda().eval()
    dim = model.config.hidden_size
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)

    for split in SPLITS:
        out_path = OUT_DIR / f"{model_key}__{col}__{split}.npy"
        if out_path.exists():
            print(f"  skip {out_path.name} (exists)")
            continue
        d = ds[split]
        n = len(d)
        feats = np.empty((n, dim), dtype=np.float32)
        t0 = time.time()
        for i0 in range(0, n, batch_size):
            i1 = min(i0 + batch_size, n)
            imgs = d[i0:i1][col]  # list of PIL images, decoded here
            imgs = [im.convert("RGB") for im in imgs]
            inputs = proc(images=imgs, return_tensors="pt").to("cuda")
            with autocast:
                out = model(**inputs).pooler_output
            feats[i0:i1] = out.float().cpu().numpy()
            if i0 % (batch_size * 20) == 0:
                done = i1
                rate = done / max(time.time() - t0, 1e-6)
                print(f"    {model_key} {col} {split}: {done}/{n}  {rate:.0f} img/s",
                      flush=True)
        np.save(out_path, feats)
        dt = time.time() - t0
        print(f"  wrote {out_path.name}  [{n}, {dim}]  {n / dt:.0f} img/s  {dt:.0f}s")

    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--cols", nargs="+", default=IMAGE_COLS, choices=IMAGE_COLS)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {DATASET} ...")
    ds = load_dataset(DATASET)
    print({k: len(v) for k, v in ds.items()})

    write_labels(ds)
    for model_key in args.models:
        for col in args.cols:
            print(f"\n== {model_key} x {col} ==")
            extract(ds, model_key, col, args.batch_size, bf16=not args.fp32)

    print("\ndone.")


if __name__ == "__main__":
    main()
