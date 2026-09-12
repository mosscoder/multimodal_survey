"""Horizontal-flip augmentation (2x training data) on the best config.
ground@768/both + drone@320-full/both. Flipped copies added to TRAIN folds only;
validation is always the original (unflipped) frames. Augmented copies inherit the
frame's leg-fold, so no leakage.

Heads: linear, MLP(h256, dropout 0.4), and their 0.5/0.5 ensemble; 3 seeds each.
Baseline (no aug) numbers from results/mlp_tune.csv for comparison.
"""
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool, standardize
from head_cv import CLASSES

folds = pool["fold"].values
IDX = pool.index


def cat(gfile, dfile):
    g = np.load(gfile); d = np.load(dfile)
    return np.concatenate([g, d], 1)[IDX]

X_orig = cat("feats/train.ground_image.general.r768.npy",
             "feats/train.drone_image.satellite.full.npy")
X_flip = cat("feats/train.ground_image.general.r768.hflip.npy",
             "feats/train.drone_image.satellite.full.hflip.npy")


def make_head(kind, d_in, hidden=256, p_drop=0.4):
    if kind == "linear":
        return nn.Linear(d_in, 8)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p_drop),
                         nn.Linear(hidden, 8))


def fit_predict(X_tr, Y_tr, X_va, Y_va, kind, lr, wd=1e-2, epochs=60, patience=10, seed=0):
    torch.manual_seed(seed)
    head = make_head(kind, X_tr.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    p = Y_tr.mean(0)
    pw = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, Yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(Y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32)
    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        head.train()
        for idx in torch.randperm(len(Xt)).split(256):
            opt.zero_grad(); loss_fn(head(Xt[idx]), Yt[idx]).backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            P = torch.sigmoid(head(Xv)).numpy()
        f1 = f1_score(Y_va, P > 0.5, average="macro", zero_division=0)
        if f1 > best:
            best, state, bad = f1, {k: v.clone() for k, v in head.state_dict().items()}, 0
        elif (bad := bad + 1) >= patience:
            break
    head.load_state_dict(state)
    with torch.no_grad():
        return torch.sigmoid(head(Xv)).numpy()


def cv_probs(kind, lr, aug, seed=0):
    oof = np.zeros_like(Y_pool, dtype=np.float32)
    for k in np.unique(folds):
        va = folds == k
        if aug:
            Xtr = np.concatenate([X_orig[~va], X_flip[~va]], 0)
            Ytr = np.concatenate([Y_pool[~va], Y_pool[~va]], 0)
        else:
            Xtr, Ytr = X_orig[~va], Y_pool[~va]
        Xtr_s, Xva_s = standardize(Xtr, X_orig[va])
        oof[va] = fit_predict(Xtr_s, Ytr, Xva_s, Y_pool[va], kind, lr, seed=seed)
    return oof


def metrics(P):
    per_ap = average_precision_score(Y_pool, P, average=None)
    per_f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
    fold = [f1_score(Y_pool[folds == k], P[folds == k] > 0.5, average="macro", zero_division=0)
            for k in np.unique(folds)]
    return per_f1.mean(), per_ap.mean(), np.mean(fold), np.std(fold), per_ap


rows, apr = [], []
def log(tag, P):
    f1, ap, fm, fsd, per_ap = metrics(P)
    rows.append(dict(config=tag, macro_F1=f1, macro_AP=ap, perfold_F1=fm, perfold_sd=fsd))
    apr.append(dict(config=tag, **{c.split()[1][:6]: v for c, v in zip(CLASSES, per_ap)}))
    print(f"{tag:28s} macroF1 {f1:.3f}  macroAP {ap:.3f}  perfold {fm:.3f}+/-{fsd:.3f}", flush=True)


store = {}
for aug in (False, True):
    key = "aug" if aug else "noaug"
    lin = np.mean([cv_probs("linear", 1e-2, aug, s) for s in (0, 1, 2)], 0)
    mlp = np.mean([cv_probs("mlp", 1e-3, aug, s) for s in (0, 1, 2)], 0)
    store[key] = (lin, mlp)
    log(f"{key}: linear (3-seed)", lin)
    log(f"{key}: mlp h256/p0.4 (3-seed)", mlp)
    log(f"{key}: linear + mlp ENS", 0.5 * lin + 0.5 * mlp)

df = pd.DataFrame(rows)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP ===")
print(pd.DataFrame(apr).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/hflip.csv", index=False)
pd.DataFrame(apr).to_csv("results/hflip_perclass_ap.csv", index=False)
print("\nwrote results/hflip.csv")
