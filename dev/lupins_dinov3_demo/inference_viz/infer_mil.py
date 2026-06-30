"""Run the trained MIL head over the full robot dataset and paint per-patch firing.

Loads checkpoints/mil.pt — the FROZEN per-species thresholds AND the deployed `tiling` (rev-4:
base / macro / macro+patch, chosen by sweeps/sweep_tiling.py) — and applies the exact any-patch rule
from CLAUDE.md: a patch fires iff its score >= that species' frozen threshold; a species is present
iff >=1 patch fires. The painted patches ARE the patches that count toward the metric. No Otsu, no
re-calibration. Draws the three forbs. The tiling + thresholds are recorded in summary.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
import common as C
import data as D
import render as R
import tiling as TL

CKPT = f"{D.ROOT}/checkpoints/mil.pt"
VIZ = f"{D.ROOT}/inference_viz/overlays"
SUMMARY = f"{D.ROOT}/inference_viz/summary.json"


def main():
    ck = torch.load(CKPT)
    classes = ck["classes"]; ncls = len(classes)                     # the 9 scored TARGET species
    n_out = ck.get("n_out", ncls)                                    # head may output extra (auxiliary) classes
    thr = np.array(ck["thresholds"], np.float32)                     # FROZEN per-species thresholds
    tiling = ck.get("tiling", "base")                                # deployed tiling (rev-4)
    head = C.MILHead(n_classes=n_out, hidden=ck["hidden"], r=ck["r"]); head.load_state_dict(ck["state"]); head.eval()

    fids, _ = D.load_robot_gt()
    device = C.get_device(); model = C.load_backbone(device)
    os.makedirs(VIZ, exist_ok=True)
    print(f"inference tiling = {tiling}  (frozen thresholds, any-patch rule)", flush=True)
    summary, present_counter = [], {n: 0 for n in classes}
    for i, fid in enumerate(fids):
        pil = Image.open(os.path.join(C.CAPTURES, fid + ".jpg"))
        grid = TL.frame_grid(model, head, pil, tiling, n_out, device)[..., :ncls]    # (45,80,ncls)
        ov, regs = R.overlay(pil, grid, thr, classes)
        ov.save(f"{VIZ}/{fid}.jpg", quality=88)
        present = [classes[c] for c in range(ncls) if (grid[..., c] >= thr[c]).any()]   # any-patch metric
        for s in present:
            present_counter[s] += 1
        summary.append({"image": fid, "present": present,                                # metric (all 9)
                        "regions": {sp: [{"centroid": r["centroid"], "area": r["area"]} for r in rs]
                                    for sp, rs in regs.items()}})                          # alpha-shape centroids (forbs)
        if i % 25 == 0:
            print(f"  {i}/{len(fids)}", end="\r", flush=True)
    print()
    json.dump({"classes": classes, "tiling": tiling,
               "thresholds": dict(zip(classes, [round(float(t), 4) for t in thr])),
               "frames_with_species": present_counter, "results": summary}, open(SUMMARY, "w"), indent=2)
    print("frames with each species:", present_counter)
    print("viz ->", VIZ)


if __name__ == "__main__":
    main()
