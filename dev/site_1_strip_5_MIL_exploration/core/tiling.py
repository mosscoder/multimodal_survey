"""Inference-time tiling strategies for robot frames (rev-4 tiling sweep + deployment).

Three any-patch tilings, every one reduced to the SAME canonical 45x80 patch grid so the overlay
and the any-patch metric are identical downstream:

  base         - 24 disjoint native 224 tiles (geometry.tiles); the rev-1..3 deployment.
  macro        - re-tile at stride 112 (half-tile overlap): every location is seen near a tile
                 CENTRE (full natural context) rather than only at a tile edge.
  macro+patch  - macro UNION an 8px-shifted patch lattice (PAD8): also samples BETWEEN the 16px
                 patches (a feature straddling a patch boundary is no longer split).

Denser tilings can only RAISE the any-patch max, so each tiling is paired with its OWN per-species
thresholds (re-derived on val) — never the base thresholds. Single source of truth for both
sweeps/sweep_tiling.py (selection) and inference_viz/infer_mil.py (deployment).
"""
from __future__ import annotations

import numpy as np
import torch

import common as C
import geometry as G

TILINGS = ["base", "macro", "macro+patch"]
PAD = (G.PAD_T, G.PAD_B, G.PAD_L, G.PAD_R)   # (80,96,32,32) -> 1344x896, the canonical 16px lattice
PAD8 = (72, 104, 24, 40)                      # 8px-shifted lattice (corner<->centre patch offset)
STRIDE = G.TILE // 2                          # 112 (half-tile macro overlap)


def _frame_arr(pil) -> np.ndarray:
    pil = pil.convert("RGB")
    return np.asarray(pil if pil.size == (G.NW, G.NH) else pil.resize((G.NW, G.NH)), np.uint8)


def _accumulate(model, head, device, farr, pad, n_out, smap) -> None:
    """Max-accumulate per-patch sigmoids from a stride-112 tiling at padding `pad` onto the
    per-pixel score map smap (NH,NW,n_out). Each 16px patch paints its score onto its pixel block;
    overlapping tiles combine by max (the any-patch rule)."""
    pt, pb, pl, pr = pad
    canvas = np.pad(farr, ((pt, pb), (pl, pr), (0, 0)), "reflect")
    Hc, Wc = canvas.shape[:2]
    tiles, origins = [], []
    for ty in range(0, Hc - G.TILE + 1, STRIDE):
        for tx in range(0, Wc - G.TILE + 1, STRIDE):
            t = canvas[ty:ty + G.TILE, tx:tx + G.TILE]
            tiles.append((torch.from_numpy(t.astype(np.float32) / 255.).permute(2, 0, 1) - C.MEAN[0]) / C.STD[0])
            origins.append((ty, tx))
    tiles = torch.stack(tiles)
    with torch.no_grad():
        for k in range(0, len(tiles), 16):
            feats, _, _ = C.patch_features(model, tiles[k:k + 16], device)
            s = torch.sigmoid(head(feats.float().cpu())[1])[:, :, :n_out].numpy()
            for bi in range(s.shape[0]):
                ty, tx = origins[k + bi]
                g = s[bi].reshape(G.PG, G.PG, n_out)
                for py in range(G.PG):
                    fy = ty + py * C.PATCH - pt
                    if fy >= G.NH or fy + C.PATCH <= 0:
                        continue
                    y0, y1 = max(fy, 0), min(fy + C.PATCH, G.NH)
                    for px in range(G.PG):
                        fx = tx + px * C.PATCH - pl
                        if fx >= G.NW or fx + C.PATCH <= 0:
                            continue
                        x0, x1 = max(fx, 0), min(fx + C.PATCH, G.NW)
                        smap[y0:y1, x0:x1] = np.maximum(smap[y0:y1, x0:x1], g[py, px])


def _smap_to_grid(smap, n_out) -> np.ndarray:
    return smap.reshape(G.GH, C.PATCH, G.GW, C.PATCH, n_out).max((1, 3))


def base_grid(model, head, pil, n_out, device=None) -> np.ndarray:
    """(GH,GW,n_out) any-patch grid from the canonical 24 disjoint tiles (the deployed metric)."""
    device = device or C.get_device()
    t = G.tiles(pil)
    with torch.no_grad():
        feats, _, _ = C.patch_features(model, t, device)
        ps = torch.sigmoid(head(feats.float().cpu())[1])[:, :, :n_out].numpy()   # (24,196,n_out)
    return G.to_grid(ps)


def all_grids(model, head, pil, n_out, device=None) -> dict:
    """All three tilings' (GH,GW,n_out) grids for ONE frame, sharing passes (macro+patch reuses
    macro's score map). Used by the tiling sweep to score every method from one encode."""
    device = device or C.get_device()
    farr = _frame_arr(pil)
    macro_smap = np.full((G.NH, G.NW, n_out), -1.0, np.float32)
    _accumulate(model, head, device, farr, PAD, n_out, macro_smap)
    mp_smap = macro_smap.copy()
    _accumulate(model, head, device, farr, PAD8, n_out, mp_smap)
    return {"base": base_grid(model, head, pil, n_out, device),
            "macro": _smap_to_grid(macro_smap, n_out),
            "macro+patch": _smap_to_grid(mp_smap, n_out)}


def frame_grid(model, head, pil, tiling, n_out, device=None) -> np.ndarray:
    """(GH,GW,n_out) any-patch grid for ONE frame under a single tiling (used by deployment)."""
    device = device or C.get_device()
    if tiling == "base":
        return base_grid(model, head, pil, n_out, device)
    if tiling not in ("macro", "macro+patch"):
        raise ValueError(f"unknown tiling {tiling!r}")
    farr = _frame_arr(pil)
    smap = np.full((G.NH, G.NW, n_out), -1.0, np.float32)
    _accumulate(model, head, device, farr, PAD, n_out, smap)
    if tiling == "macro+patch":
        _accumulate(model, head, device, farr, PAD8, n_out, smap)
    return _smap_to_grid(smap, n_out)
