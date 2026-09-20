"""Residual/boosting fusion: ground's logits are FROZEN (converted from its
already-trained OOF probabilities), and only a drone-side correction is
trained on top via the same combined BCE loss. Unlike every joint-fusion
attempt, ground's weights never move here -- there's no model for drone's
gradients to destabilize, and no learnable-scale sign-symmetry degeneracy
since the frozen term isn't parameterized at all.

  boost_fixed  : logit = z_ground (frozen, coef=1) + drone_linear(a)
  boost_scaled : logit = scale*z_ground (scale learnable, init 1) + drone_linear(a)

drone input = 'both' pooling (CLS + mean-patch, 2048-d).
"""
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
SEEDS = [0, 1, 2]
K = 5
folds = pool["fold"].values
rows_pool = pool.index.to_numpy()

d = np.load("results/armhead_ablation_oof.npz")
ground_p = np.mean([d[f"ground_linear_s{s}"] for s in SEEDS], axis=0)
eps = 1e-6
pc = np.clip(ground_p, eps, 1 - eps)
z_ground = np.log(pc / (1 - pc)).astype(np.float32)                      # frozen logits, (N,8)

A = np.load("feats/train.drone_image.satellite.npy").astype(np.float32)[rows_pool]  # (N,2048) both


class BoostHead(nn.Module):
    def __init__(self, d_a, scaled, n_out=8):
        super().__init__()
        self.drone = nn.Linear(d_a, n_out)
        nn.init.zeros_(self.drone.weight); nn.init.zeros_(self.drone.bias)  # start = ground alone
        self.scaled = scaled
        if scaled:
            self.scale = nn.Parameter(torch.ones(n_out))

    def forward(self, zg, a):
        base = self.scale * zg if self.scaled else zg
        return base + self.drone(a)


def fit_predict(zg_tr, A_tr, Ytr, zg_va, A_va, Yva, scaled,
                lr=1e-2, wd=1e-2, epochs=60, patience=10, batch=64, seed=0):
    torch.manual_seed(seed)
    head = BoostHead(A_tr.shape[1], scaled)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    p = Ytr.mean(0)
    pw = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Zt = torch.tensor(zg_tr, dtype=torch.float32)
    At = torch.tensor(A_tr, dtype=torch.float32)
    Yt = torch.tensor(Ytr, dtype=torch.float32)
    Zv = torch.tensor(zg_va, dtype=torch.float32)
    Av = torch.tensor(A_va, dtype=torch.float32)
    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        head.train()
        order = np.random.permutation(len(At))
        for j in range(0, len(order), batch):
            idx = order[j:j+batch]
            opt.zero_grad(); loss_fn(head(Zt[idx], At[idx]), Yt[idx]).backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            P = torch.sigmoid(head(Zv, Av)).numpy()
        f1 = f1_score(Yva, P > 0.5, average="macro", zero_division=0)
        if f1 > best:
            best, state, bad = f1, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            break
    head.load_state_dict(state); head.eval()
    with torch.no_grad():
        return torch.sigmoid(head(Zv, Av)).numpy(), head


def cv_oof(scaled, tag):
    seed_oof, scales = [], []
    for s in SEEDS:
        oof = np.zeros(Y_pool.shape, dtype=np.float32)
        for f in range(K):
            tr, va = folds != f, folds == f
            mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
            Atr, Ava = (A[tr] - mu) / sd, (A[va] - mu) / sd
            oof[va], head = fit_predict(z_ground[tr], Atr, Y_pool[tr],
                                        z_ground[va], Ava, Y_pool[va], scaled, seed=s)
            if scaled:
                scales.append(dict(seed=s, fold=f, **{c.split()[0]: v for c, v in
                                                       zip(CLASSES, head.scale.detach().numpy())}))
        seed_oof.append(oof)
        f1 = f1_score(Y_pool, oof > 0.5, average=None, zero_division=0)
        ap = average_precision_score(Y_pool, oof, average=None)
        print(f"  {tag} seed{s}  macroF1 {f1.mean():.3f}  macroAP {ap.mean():.3f}", flush=True)
    return np.mean(seed_oof, 0), scales


def score(P):
    f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
    ap = average_precision_score(Y_pool, P, average=None)
    ff = [f1_score(Y_pool[folds == k], P[folds == k] > 0.5, average="macro", zero_division=0) for k in range(K)]
    return f1, ap, np.mean(ff), np.std(ff)


def report(tag, P, rows, apr):
    f1, ap, fm, fsd = score(P)
    print(f"{tag:24s} macroF1 {f1.mean():.3f}  macroAP {ap.mean():.3f}  perfold {fm:.3f}+/-{fsd:.3f}")
    rows.append(dict(config=tag, macro_F1=f1.mean(), macro_AP=ap.mean(), perfold_F1=fm, perfold_sd=fsd))
    apr.append(dict(config=tag, **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap)}))


rows, apr = [], []
report("ground alone (ref)", ground_p, rows, apr)

print("training boost_fixed (drone corrects frozen ground logits, coef=1) ...")
p_fixed, _ = cv_oof(False, "boost_fixed")
report("boost_fixed", p_fixed, rows, apr)

print("training boost_scaled (+ learnable scale on frozen ground logits) ...")
p_scaled, scales = cv_oof(True, "boost_scaled")
report("boost_scaled", p_scaled, rows, apr)

print("\nlearned scale on frozen ground logits (1.0 = no change), by species (mean over seeds/folds):")
sc_df = pd.DataFrame(scales)
print(sc_df[[c.split()[0] for c in CLASSES]].mean().to_string(float_format=lambda x: f"{x:.3f}"))

# reference: previous best (late fusion, weighted avg) for comparison
prev = pd.read_csv("results/late_fusion.csv")
best_row = prev[prev.config.str.contains("weighted avg")].iloc[0]
rows.append(dict(config="weighted avg late fusion (prev best)", macro_F1=best_row.macro_F1,
                 macro_AP=best_row.macro_AP, perfold_F1=best_row.perfold_F1, perfold_sd=best_row.perfold_sd))

df = pd.DataFrame(rows)
print("\n=== summary (incl. previous best for comparison) ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP ===")
print(pd.DataFrame(apr).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/boosting_fusion.csv", index=False)
pd.DataFrame(apr).to_csv("results/boosting_fusion_perclass_ap.csv", index=False)
sc_df.to_csv("results/boosting_fusion_scales.csv", index=False)
print("\nwrote results/boosting_fusion.csv")
