"""Follow-up ablation — low LR at 2048 imgs/species.

Size sweep showed 2048 < 512 with LR 5e-4 (best epoch 0 → the head overfit big data in <1 epoch). If
that was an LR×data artifact, a much lower LR should let the full 2048 data help. Retrains the deployed
config (species all, hidden 64, crop square+native(0.5), r 8, BCE) at LR in {1e-6,1e-5,1e-4,5e-4} on
2048 imgs/species (cached @2048 features). 5e-4 is the control (reproduces the size-sweep 2048 number).
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import common as C, data as D, train_mil as T

SPECIES = D.species_set("all"); VIEWS = [("square",), ("native", 0.5)]
NPER, HID, RL, NCLS = 2048, 64, 8.0, len(D.species_set("all"))
LRS = [1e-6]                  # only the new point; 5e-4@2048 (val sel7 0.753) is already in the size sweep

valP, valY, testP, testY = T.load_splits(None, None)
trX, trL = T.extract_inat_bags(None, None, SPECIES, NPER, VIEWS, n_extract=2048)
print(f"trained on {len(trX)} bags (2048/species x {NCLS} species x 2 views)\n")
print(f"{'LR':>8}{'val sel7':>10}{'test sel7':>11}{'test all9':>11}{'ep':>5}")
results = []
for lr in LRS:
    b = T.train_head(trX, trL, valP, valY, lr=lr, hidden=HID, r=RL, n_classes=NCLS)
    head = C.MILHead(n_classes=NCLS, hidden=HID, r=RL); head.load_state_dict(b["state"]); head.eval()
    ts = T.frame_scores(head, testP)[:, :D.N]
    ta9, tf1, _ = D.macro_f1(ts, testY, b["thr"])
    tsel = float(np.mean([tf1[c] for c in D.SWEEP_IDX]))
    print(f"{lr:>8.0e}{b['sel']:>10.4f}{tsel:>11.4f}{ta9:>11.4f}{b['ep']:>5}", flush=True)
    results.append((lr, b, head, tsel))

best = max(results, key=lambda x: x[1]["sel"])
T.report(T.frame_scores(best[2], testP)[:, :D.N], testY, best[1]["thr"], f"best (LR={best[0]:.0e}) @2048 — TEST")
print("\nreference (rev-3 size sweep, LR 5e-4): 512 -> val sel7 0.7785 | 2048 -> val sel7 0.7528")
