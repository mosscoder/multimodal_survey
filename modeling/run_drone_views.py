"""Ablation: does a plant-scale drone sub-crop help? Ground fixed at 768/both.

A) swap each drone view in as the drone block, rerun CV.
B) augmentation: stack (full, center3m, bottom3m) as extra train rows, score full only.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool, standardize
from head_cv import fit_predict, CLASSES

folds = pool["fold"].values
IDX = pool.index
VIEWS = ["full", "center3m", "center2m", "bottom3m", "bottom2m"]


def gblock(pooling="both"):
    F = np.load("feats/train.ground_image.general.r768.npy")
    return {"cls": F[:, :768], "patch": F[:, 768:], "both": F}[pooling]

def dblock(view, pooling="both"):
    F = np.load(f"feats/train.drone_image.satellite.{view}.npy")
    return {"cls": F[:, :1024], "patch": F[:, 1024:], "both": F}[pooling]

G = gblock("both")                          # full array, index later


def cv_oof(Xfull, kind, lr, aug_views=None, seed=0):
    """Xfull: [Npool, d] concat design matrix for the 'full' drone view.
    aug_views: list of extra [Npool, d] matrices to stack as train rows (labels copied)."""
    oof = np.zeros_like(Y_pool, dtype=np.float32)
    for k in np.unique(folds):
        va = folds == k
        Xtr, Ytr = Xfull[~va], Y_pool[~va]
        if aug_views:
            Xtr = np.concatenate([Xtr] + [A[~va] for A in aug_views], 0)
            Ytr = np.concatenate([Ytr] * (1 + len(aug_views)), 0)
        Xtr_s, Xva_s = standardize(Xtr, Xfull[va])
        P, _ = fit_predict(Xtr_s, Ytr, Xva_s, Y_pool[va], kind, lr, seed=seed)
        oof[va] = P
    return oof


def score(P):
    per_f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
    per_ap = average_precision_score(Y_pool, P, average=None)
    fold_f1 = [f1_score(Y_pool[folds == k], P[folds == k] > 0.5, average="macro", zero_division=0)
               for k in np.unique(folds)]
    return per_f1, per_ap, np.mean(fold_f1), np.std(fold_f1)


rows, ap_rows = [], []


def run(tag, Xfull, kind, lr, aug=None):
    P = cv_oof(Xfull, kind, lr, aug_views=aug)
    per_f1, per_ap, fm, fsd = score(P)
    rows.append(dict(config=tag, d=Xfull.shape[1], macro_F1=per_f1.mean(), macro_AP=per_ap.mean(),
                     perfold_F1=fm, perfold_sd=fsd))
    ap_rows.append(dict(config=tag, **{c.split()[1][:6]: v for c, v in zip(CLASSES, per_ap)}))
    print(f"{tag:32s} d={Xfull.shape[1]:4d}  macroF1 {per_f1.mean():.3f}  macroAP {per_ap.mean():.3f}  "
          f"perfold {fm:.3f}+/-{fsd:.3f}", flush=True)


def concat(view, pooling="both"):
    return np.concatenate([gblock("both"), dblock(view, pooling)], 1)[IDX]


# A) view ablation, linear/both (recommended) + mlp/cls
for view in VIEWS:
    run(f"linear/both/{view}", concat(view, "both"), "linear", 1e-2)
for view in VIEWS:
    run(f"mlp/cls/{view}", concat(view, "cls" if False else "both"), "mlp", 1e-3)

# B) augmentation: full scored, (center3m, bottom3m) stacked as train rows
Xfull = concat("full", "both")
aug = [concat("center3m", "both"), concat("bottom3m", "both")]
run("linear/both/full+aug(c3,b3)", Xfull, "linear", 1e-2, aug=aug)
run("mlp/both/full+aug(c3,b3)",   Xfull, "mlp", 1e-3, aug=aug)

df = pd.DataFrame(rows)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP ===")
print(pd.DataFrame(ap_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/drone_views.csv", index=False)
print("\nwrote results/drone_views.csv")
