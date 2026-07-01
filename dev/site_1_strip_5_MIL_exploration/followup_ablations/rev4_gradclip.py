"""rev-4 follow-up ablation D — gradient (grad-norm) clipping. Head-only on CACHED features, scored at
BASE any-patch (deployment adds macro+patch separately). Two questions:
  1. Does grad-norm clipping help the deployed gentle config (r=2, lr 1e-5)?
  2. Does clipping RESCUE the hot high-LR regime? rev-3/4 found lr 5e-4 overfits at epoch ~0-1 (best
     epoch collapses); bounding the step might let it train stably and converge fast.
Config = deployed: species all(36)+Sky, hidden 64, crop square+native(0.5), n_per 512, r 2; vary LR + clip.
Thresholds re-derived on val (F1-argmax), applied to held-out test.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import common as C
import data as D
import train_mil as T

FORBS = ["Lupinus sericeus", "Gaillardia aristata", "Balsamorhiza sagittata"]
VIEWS = [("square",), ("native", 0.5)]


def run(tag, species, trXL, r, lr, clip=None):
    trX, trL = trXL
    b = T.train_head(trX, trL, valP, valY, lr=lr, hidden=64, r=r, n_classes=len(species), clip=clip)
    head = C.MILHead(n_classes=len(species), hidden=64, r=r); head.load_state_dict(b["state"]); head.eval()
    vs = T.frame_scores(head, valP)[:, :D.N]; ts = T.frame_scores(head, testP)[:, :D.N]
    _, vf1, ths = D.macro_f1(vs, valY, None)
    tall9, tf1, _ = D.macro_f1(ts, testY, np.array(ths))
    vsel = float(np.mean([vf1[c] for c in D.SWEEP_IDX])); tsel = float(np.mean([tf1[c] for c in D.SWEEP_IDX]))
    forb = []
    for f in FORBS:
        c = D.NAMES.index(f); pred = ts[:, c] >= ths[c]; y = testY[:, c]
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); Fv = 2 * P * R / max(P + R, 1e-9)
        forb.append(f"{f.split()[0][:4]} F{Fv:.2f}(P{P:.2f}/R{R:.2f})")
    print(f"RESULT | {tag:26} | val sel7 {vsel:.4f} | test sel7 {tsel:.4f} all9 {tall9:.4f} | ep {b['ep']:3d} | "
          + "  ".join(forb), flush=True)


device = C.get_device(); model = C.load_backbone(device)
valP, valY, testP, testY = T.load_splits(model, device)
ALL = D.species_set("all")
allXL = T.extract_inat_bags(model, device, ALL, n_per=512, views=VIEWS, n_extract=512)

print("\n##### Ablation D — gradient clipping (base tiling) #####")
run("lr1e-5 no-clip (deployed)", ALL, allXL, r=2, lr=1e-5, clip=None)
run("lr1e-5 clip 1.0", ALL, allXL, r=2, lr=1e-5, clip=1.0)
run("lr1e-5 clip 0.5", ALL, allXL, r=2, lr=1e-5, clip=0.5)
run("lr5e-4 no-clip (hot ref)", ALL, allXL, r=2, lr=5e-4, clip=None)
run("lr5e-4 clip 1.0", ALL, allXL, r=2, lr=5e-4, clip=1.0)
run("lr5e-4 clip 0.5", ALL, allXL, r=2, lr=5e-4, clip=0.5)
print("\ndone")
