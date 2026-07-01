"""rev-4 follow-up ablations — head-only on CACHED features, scored at BASE any-patch (the protocol the
staged sweep selected on; deployment adds macro+patch on top). No DINOv3 re-encode. Three questions:

  A. Sky class A/B          — does the Sky negative drive the headline-forb precision gain, or did the
                              other rev-4 changes (license data / r=2 / lr=1e-5)? Train the deployed
                              config WITH vs WITHOUT the Sky class, everything else identical.
  B. Raised-cap convergence — is lr 1e-5 under-converged at ep 71/80? does 5e-6 train given 200 epochs?
                              (cap 200, early-stop disabled so it runs the full budget.)
  C. r < 2                  — where does the LSE bag temperature turn over (toward mean / GAP pooling)?

Base config = deployed: species all(36)+Sky, hidden 64, crop square+native(0.5), n_per 512, r 2, lr 1e-5.
Thresholds re-derived on val (F1-argmax), applied to held-out test. Reuses the sweep's cached features.
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

FORBS = ["Lupinus sericeus", "Gaillardia aristata", "Balsamorhiza sagittata"]   # headline forbs
VIEWS = [("square",), ("native", 0.5)]


def run(tag, species, trXL, r, lr, cap=80, patience=12):
    oe, op = D.EPOCHS, D.PATIENCE
    D.EPOCHS, D.PATIENCE = cap, patience
    trX, trL = trXL
    b = T.train_head(trX, trL, valP, valY, lr=lr, hidden=64, r=r, n_classes=len(species))
    D.EPOCHS, D.PATIENCE = oe, op
    head = C.MILHead(n_classes=len(species), hidden=64, r=r); head.load_state_dict(b["state"]); head.eval()
    vs = T.frame_scores(head, valP)[:, :D.N]; ts = T.frame_scores(head, testP)[:, :D.N]
    _, vf1, ths = D.macro_f1(vs, valY, None)                    # re-derive on val
    tall9, tf1, _ = D.macro_f1(ts, testY, np.array(ths))        # apply to test
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
ALL = D.species_set("all"); NOSKY = ALL[:-1]                    # 36 plants + Sky ; 36 plants only
allXL = T.extract_inat_bags(model, device, ALL, n_per=512, views=VIEWS, n_extract=512)
noskyXL = T.extract_inat_bags(model, device, NOSKY, n_per=512, views=VIEWS, n_extract=512)

print("\n##### Ablation A — Sky class A/B (base tiling) #####")
run("all+Sky (deployed ref)", ALL, allXL, r=2, lr=1e-5)
run("all, NO Sky", NOSKY, noskyXL, r=2, lr=1e-5)

print("\n##### Ablation B — raised-cap convergence (base tiling) #####")
run("1e-5 @ cap80 (deployed)", ALL, allXL, r=2, lr=1e-5, cap=80, patience=12)
run("1e-5 @ cap200 (no stop)", ALL, allXL, r=2, lr=1e-5, cap=200, patience=200)
run("5e-6 @ cap200 (no stop)", ALL, allXL, r=2, lr=5e-6, cap=200, patience=200)

print("\n##### Ablation C — r < 2 turnover (base tiling) #####")
run("r=2 (deployed)", ALL, allXL, r=2.0, lr=1e-5)
run("r=1", ALL, allXL, r=1.0, lr=1e-5)
run("r=0.5", ALL, allXL, r=0.5, lr=1e-5)
print("\ndone")
