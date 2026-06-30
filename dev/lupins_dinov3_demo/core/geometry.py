"""Robot-frame grid geometry — the deployment tiling the metric is defined on.

Reflect-pad a 1280x720 frame to 1344x896 and cut it into 24 disjoint NATIVE-resolution
224x224 tiles (6x4, no downscaling), and map per-patch scores back to the 45x80 patch
grid over the original frame.
"""
from __future__ import annotations

import numpy as np
import torch

import common as C

NW, NH = 1280, 720
TILE = 224
PG = TILE // C.PATCH                              # 14 patches per tile side
PAD_L, PAD_R = 32, 32
PAD_T, PAD_B = 80, 96
PW, PH = NW + PAD_L + PAD_R, NH + PAD_T + PAD_B   # 1344 x 896
NTX, NTY = PW // TILE, PH // TILE                 # 6 x 4 tiles
R0, C0 = PAD_T // C.PATCH, PAD_L // C.PATCH       # crop offset: 5 rows, 2 cols
GH, GW = NH // C.PATCH, NW // C.PATCH             # 45 x 80 original patch grid


def tiles(pil) -> torch.Tensor:
    """PIL frame -> (24, 3, 224, 224) normalized native-resolution tiles."""
    pil = pil.convert("RGB")
    arr = np.asarray(pil if pil.size == (NW, NH) else pil.resize((NW, NH)), np.uint8)
    padded = np.pad(arr, ((PAD_T, PAD_B), (PAD_L, PAD_R), (0, 0)), mode="reflect")
    out = []
    for ty in range(NTY):
        for tx in range(NTX):
            t = padded[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]
            out.append((torch.from_numpy(np.asarray(t, np.float32) / 255.0).permute(2, 0, 1) - C.MEAN[0]) / C.STD[0])
    return torch.stack(out, 0)


def to_grid(per_patch: np.ndarray) -> np.ndarray:
    """(24, 196, ncls) tile-major per-patch values -> (45, 80, ncls) over the original frame."""
    ncls = per_patch.shape[-1]
    g = (per_patch.reshape(NTY, NTX, PG, PG, ncls).transpose(0, 2, 1, 3, 4)
         .reshape(NTY * PG, NTX * PG, ncls))
    return g[R0:R0 + GH, C0:C0 + GW]
