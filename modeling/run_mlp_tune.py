"""Best config (concat: ground@768/both + drone@320-full/both). Try to make the MLP
beat the linear head: smaller hidden, more dropout / weight decay, seed-ensembling,
and a linear+MLP ensemble.
"""
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool, standardize
from head_cv import CLASSES

folds = pool["fold"].values
IDX = pool.index


def X_best():
    g = np.load("feats/train.ground_image.general.r768.npy")            # [N,1536] cls|patch
    d = np.load("feats/train.drone_image.satellite.full.npy")           # [N,2048] cls|patch
    return np.concatenate([g, d], 1)[IDX]                               # [Npool, 3584]


def make_head(kind, d_in, hidden=512, p_drop=0.2):
    if kind == "linear":
        return nn.Linear(d_in, 8)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p_drop),
                         nn.Linear(hidden, 8))


def fit_predict(X_tr, Y_tr, X_va, Y_va, kind, lr, wd, hidden, p_drop,
                epochs=60, patience=10, seed=0):
    torch.manual_seed(seed)
    head = make_head(kind, X_tr.shape[1], hidden, p_drop)
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


def cv_probs(X, kind, lr=1e-3, wd=1e-2, hidden=512, p_drop=0.2, seed=0):
    oof = np.zeros_like(Y_pool, dtype=np.float32)
    for k in np.unique(folds):
        va = folds == k
        Xtr, Xva = standardize(X[~va], X[va])
        oof[va] = fit_predict(Xtr, Y_pool[~va], Xva, Y_pool[va],
                              kind, lr, wd, hidden, p_drop, seed=seed)
    return oof


def metrics(P):
    per_ap = average_precision_score(Y_pool, P, average=None)
    per_f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
    fold = [f1_score(Y_pool[folds == k], P[folds == k] > 0.5, average="macro", zero_division=0)
            for k in np.unique(folds)]
    return per_f1.mean(), per_ap.mean(), np.mean(fold), np.std(fold), per_ap


X = X_best()
rows, apr = [], []


def log(tag, P):
    f1, ap, fm, fsd, per_ap = metrics(P)
    rows.append(dict(config=tag, macro_F1=f1, macro_AP=ap, perfold_F1=fm, perfold_sd=fsd))
    apr.append(dict(config=tag, **{c.split()[1][:6]: v for c, v in zip(CLASSES, per_ap)}))
    print(f"{tag:34s} macroF1 {f1:.3f}  macroAP {ap:.3f}  perfold {fm:.3f}+/-{fsd:.3f}", flush=True)
    return ap


lin = cv_probs(X, "linear", lr=1e-2)
log("linear/both (ref)", lin)

GRID = [
    ("h512 p0.2 wd1e-2", dict(hidden=512, p_drop=0.2, wd=1e-2)),   # current baseline
    ("h256 p0.2 wd1e-2", dict(hidden=256, p_drop=0.2, wd=1e-2)),   # the ask
    ("h256 p0.4 wd1e-2", dict(hidden=256, p_drop=0.4, wd=1e-2)),
    ("h256 p0.2 wd5e-2", dict(hidden=256, p_drop=0.2, wd=5e-2)),
    ("h128 p0.3 wd3e-2", dict(hidden=128, p_drop=0.3, wd=3e-2)),
    ("h64  p0.2 wd1e-2", dict(hidden=64,  p_drop=0.2, wd=1e-2)),
]
best_ens, best_ap = None, -1
for tag, kw in GRID:
    seed_probs = [cv_probs(X, "mlp", lr=1e-3, seed=s, **kw) for s in (0, 1, 2)]
    for s, P in zip((0, 1, 2), seed_probs):
        log(f"mlp {tag} s{s}", P)
    ens = np.mean(seed_probs, 0)
    ap = log(f"mlp {tag} ENS(3 seeds)", ens)
    if ap > best_ap:
        best_ap, best_ens = ap, (tag, ens)

tag, mlp_ens = best_ens
log(f"linear + mlp[{tag}] ENS", 0.5 * lin + 0.5 * mlp_ens)

df = pd.DataFrame(rows)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP ===")
print(pd.DataFrame(apr).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/mlp_tune.csv", index=False)
pd.DataFrame(apr).to_csv("results/mlp_tune_perclass_ap.csv", index=False)
print("\nwrote results/mlp_tune.csv")
