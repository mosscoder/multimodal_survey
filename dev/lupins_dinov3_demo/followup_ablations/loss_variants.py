"""Follow-up ablation — training LOSS variants for the MIL head: focal loss + label smoothing.

Retrains the head on the deployed config (species all, hidden 64, LR 5e-4, crop square+native(0.5),
512 imgs, r 8) swapping only the loss; features are cached so each run is a quick head-train. Loss is
applied to the LSE-pooled bag logits (one-vs-all), keeping the class-balanced pos_weight.
  - BCE (baseline)        : F.binary_cross_entropy_with_logits + pos_weight
  - focal(γ)              : (1-p_t)^γ * BCE   — down-weights easy bags (γ=0.5,1,2)
  - label-smooth(ε)       : targets 1->1-ε, 0->ε  (ε=0.05,0.1)
Selection = val sel7 (same as the chain); reports val/test sel7 + test all9, then per-species for the
baseline and the best variant.
"""
from __future__ import annotations
import os, sys
import numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import common as C, data as D, train_mil as T

SPECIES = D.species_set("all"); VIEWS = [("square",), ("native", 0.5)]
NPER, LR, HID, RL = 512, 5e-4, 64, 8.0
NCLS = len(SPECIES)


def bce(logits, t, posw):
    return F.binary_cross_entropy_with_logits(logits, t, pos_weight=posw)


def focal(gamma):
    def f(logits, t, posw):
        ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none", pos_weight=posw)
        p = torch.sigmoid(logits); pt = p * t + (1 - p) * (1 - t)
        return (((1 - pt) ** gamma) * ce).mean()
    return f


def lsmooth(eps):
    def f(logits, t, posw):
        return F.binary_cross_entropy_with_logits(logits, t * (1 - eps) + (1 - t) * eps, pos_weight=posw)
    return f


def train_variant(trX, trL, valP, valY, lossfn):
    counts = np.bincount(trL.numpy(), minlength=NCLS); ntot = int(counts.sum())
    posw = torch.tensor([(ntot - c) / max(c, 1) for c in counts], dtype=torch.float32)
    torch.manual_seed(D.SEED)
    head = C.MILHead(n_classes=NCLS, hidden=HID, r=RL)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=D.WD)
    trY = F.one_hot(trL, NCLS).float()
    idx = np.arange(len(trX)); rng = np.random.default_rng(D.SEED); best = {"sel": -1.0}; bad = 0
    for ep in range(D.EPOCHS):
        head.train(); rng.shuffle(idx)
        for i in range(0, len(idx), D.BATCH):
            b = idx[i:i + D.BATCH]
            bl, _ = head(trX[b].float())
            loss = lossfn(bl, trY[b], posw); opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        m = T.val_metrics(head, valP, valY)
        if m["sel"] > best["sel"] + 1e-4:
            best = {**m, "ep": ep, "state": {k: v.clone() for k, v in head.state_dict().items()}}; bad = 0
        else:
            bad += 1
            if bad >= D.PATIENCE:
                break
    return best


def main():
    valP, valY, testP, testY = T.load_splits(None, None)
    trX, trL = T.extract_inat_bags(None, None, SPECIES, NPER, VIEWS)
    variants = [("BCE (baseline)", bce), ("focal g=0.5", focal(0.5)), ("focal g=1", focal(1.0)),
                ("focal g=2", focal(2.0)), ("label-smooth e=0.05", lsmooth(0.05)), ("label-smooth e=0.10", lsmooth(0.10))]
    rows = []
    print(f"{'variant':22}{'val sel7':>10}{'test sel7':>11}{'test all9':>11}{'ep':>4}")
    for tag, fn in variants:
        b = train_variant(trX, trL, valP, valY, fn)
        head = C.MILHead(n_classes=NCLS, hidden=HID, r=RL); head.load_state_dict(b["state"]); head.eval()
        ts = T.frame_scores(head, testP)[:, :D.N]
        ta9, tf1, tths = D.macro_f1(ts, testY, b["thr"])
        tsel = float(np.mean([tf1[c] for c in D.SWEEP_IDX]))
        rows.append((tag, b, head, round(tsel, 4)))
        print(f"{tag:22}{b['sel']:>10.4f}{tsel:>11.4f}{ta9:>11.4f}{b['ep']:>4}")
    # per-species: baseline + best-by-val-sel7
    best_i = max(range(1, len(rows)), key=lambda i: rows[i][1]["sel"])
    for i in (0, best_i):
        tag, b, head, _ = rows[i]
        T.report(T.frame_scores(head, testP)[:, :D.N], testY, b["thr"], f"{tag} — TEST")


if __name__ == "__main__":
    main()
