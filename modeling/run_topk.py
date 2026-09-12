"""Top-k patch pooling on the 1024x768 ground grid + aerial-CLS logit offset.

TopKHead: per-patch linear logits on the ground grid, mean of the top-k per species,
plus a per-frame offset from the aerial (drone satellite) CLS. k=None (all patches) is
the mean-pool control == pooled linear cell. 8 k-values x 3 seeds, 5-fold leg CV.
"""
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool
from head_cv import CLASSES

folds = pool["fold"].values
IDX = pool.index.to_numpy()

KS = [1, 2, 4, 8, 16, 64, 256, None]
SEEDS = [0, 1, 2]

print("loading grid subset into RAM ...", flush=True)
_G = np.load("feats/train.ground_image.general.grid1024x768.npy", mmap_mode="r")  # (6480,3072,768) f16
GRID = np.asarray(_G[IDX])                                                        # (6089,3072,768) f16 in RAM
AER = np.load("feats/train.drone_image.satellite.full.npy")[IDX, :1024].astype(np.float32)  # aerial CLS
PMEAN = GRID.astype(np.float32).mean(1) if GRID.nbytes < 8e9 else \
        np.stack([GRID[i].astype(np.float32).mean(0) for i in range(len(GRID))])  # per-frame patch mean
print(f"GRID {GRID.shape} {GRID.nbytes/1e9:.1f}GB   AER {AER.shape}", flush=True)


class TopKHead(nn.Module):
    def __init__(self, k, d_g=768, d_a=1024, n_out=8):
        super().__init__()
        self.k = k
        self.ground = nn.Linear(d_g, n_out, bias=False)
        self.aerial = nn.Linear(d_a, n_out)

    def forward(self, G, a):
        s = self.ground(G)                                   # (B, P, 8)
        k = s.shape[1] if self.k is None else min(self.k, s.shape[1])
        return s.topk(k, dim=1).values.mean(1) + self.aerial(a)


def predict(head, rows, mu_t, sd_t, Astd, bs=128):
    head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            g = torch.from_numpy(GRID[rows[i:i+bs]]).cuda().float()
            g = (g - mu_t) / sd_t
            a = torch.from_numpy(Astd[i:i+bs]).cuda()
            out.append(torch.sigmoid(head(g, a)).cpu().numpy())
    return np.concatenate(out)


def fit_fold(tr, va, k, seed, epochs=50, patience=10, bs=128):
    mu = PMEAN[tr].mean(0); sd = PMEAN[tr].std(0) + 1e-6
    mu_t = torch.tensor(mu, dtype=torch.float32).cuda()
    sd_t = torch.tensor(sd, dtype=torch.float32).cuda()
    a_mu, a_sd = AER[tr].mean(0), AER[tr].std(0) + 1e-6
    Atr = ((AER[tr] - a_mu) / a_sd).astype(np.float32)
    Ava = ((AER[va] - a_mu) / a_sd).astype(np.float32)
    Ytr, Yva = Y_pool[tr], Y_pool[va]
    p = Ytr.mean(0)
    pw = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20).cuda()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    torch.manual_seed(seed)
    head = TopKHead(k).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        head.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(tr), bs):
            b = order[i:i+bs]
            g = torch.from_numpy(GRID[tr[b]]).cuda().float()
            g = (g - mu_t) / sd_t
            a = torch.from_numpy(Atr[b]).cuda()
            y = Ytr_t[b].cuda()
            opt.zero_grad(); loss_fn(head(g, a), y).backward(); opt.step()
        sched.step()
        P = predict(head, va, mu_t, sd_t, Ava)
        f1 = f1_score(Yva, P > 0.5, average="macro", zero_division=0)
        if f1 > best:
            best, state, bad = f1, {kk: v.clone() for kk, v in head.state_dict().items()}, 0
        elif (bad := bad + 1) >= patience:
            break
    head.load_state_dict(state)
    return predict(head, va, mu_t, sd_t, Ava)


rows_out, ap_out = [], []
for k in KS:
    seed_oof = []
    for seed in SEEDS:
        oof = np.zeros_like(Y_pool, dtype=np.float32)
        for fk in np.unique(folds):
            va = np.where(folds == fk)[0]
            tr = np.where(folds != fk)[0]
            oof[va] = fit_fold(tr, va, k, seed)
        seed_oof.append(oof)
    ens = np.mean(seed_oof, 0)
    for label, P in [(f"k={k} s{s}", seed_oof[i]) for i, s in enumerate(SEEDS)] + [(f"k={k} ENS", ens)]:
        per_ap = average_precision_score(Y_pool, P, average=None)
        per_f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
        fold_f1 = [f1_score(Y_pool[folds == fk], P[folds == fk] > 0.5, average="macro", zero_division=0)
                   for fk in np.unique(folds)]
        rows_out.append(dict(config=label, macro_F1=per_f1.mean(), macro_AP=per_ap.mean(),
                             perfold_F1=np.mean(fold_f1), perfold_sd=np.std(fold_f1)))
        if label.endswith("ENS"):
            ap_out.append(dict(config=label, **{c.split()[1][:6]: v for c, v in zip(CLASSES, per_ap)}))
        print(f"{label:14s} macroF1 {per_f1.mean():.3f}  macroAP {per_ap.mean():.3f}  "
              f"perfold {np.mean(fold_f1):.3f}+/-{np.std(fold_f1):.3f}", flush=True)

df = pd.DataFrame(rows_out)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP (seed-ensemble) ===")
print(pd.DataFrame(ap_out).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/topk.csv", index=False)
pd.DataFrame(ap_out).to_csv("results/topk_perclass_ap.csv", index=False)
print("\nwrote results/topk.csv")
