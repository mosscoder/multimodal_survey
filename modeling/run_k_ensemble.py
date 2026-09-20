"""Ensemble adjacent k values from the top-k sweep (results/topk_oof.npz).
No retraining -- just averaging already-saved OOF predictions. Note: this npz
is from the OLD two-branch architecture (ground top-k + drone CLS fixed sum),
so its own k=1 baseline (~0.766 AP) is below the current best (ground-alone
0.811, late-fusion 0.823) -- reported for full context, not as the headline.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
folds = pool["fold"].values
K = 5
SEEDS = [0, 1, 2]

d = np.load("results/topk_oof.npz")
def ens_k(k):
    return np.mean([d[f"k{k}_s{s}"] for s in SEEDS], axis=0)

by_k = {k: ens_k(k) for k in [1, 2, 4, 8, 16]}


def score(P):
    f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
    ap = average_precision_score(Y_pool, P, average=None)
    ff = [f1_score(Y_pool[folds == j], P[folds == j] > 0.5, average="macro", zero_division=0) for j in range(K)]
    return f1, ap, np.mean(ff), np.std(ff)


rows, apr = [], []
def report(tag, P):
    f1, ap, fm, fsd = score(P)
    print(f"{tag:24s} macroF1 {f1.mean():.3f}  macroAP {ap.mean():.3f}  perfold {fm:.3f}+/-{fsd:.3f}")
    rows.append(dict(config=tag, macro_F1=f1.mean(), macro_AP=ap.mean(), perfold_F1=fm, perfold_sd=fsd))
    apr.append(dict(config=tag, **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap)}))


for k in [1, 2, 4, 8]:
    report(f"k={k} alone (this arch)", by_k[k])

report("avg(k=1,2)", np.mean([by_k[1], by_k[2]], axis=0))
report("avg(k=1,2,4)", np.mean([by_k[1], by_k[2], by_k[4]], axis=0))
report("avg(k=1,2,4,8)", np.mean([by_k[1], by_k[2], by_k[4], by_k[8]], axis=0))

# per-class weighted variant: nested 5-fold, weight sweep over a 2-simplex for (k1,k2,k4)
Ws = np.linspace(0, 1, 11)
P_w = np.zeros(Y_pool.shape, dtype=np.float32)
w_chosen = np.zeros((K, 8, 3))
stack = np.stack([by_k[1], by_k[2], by_k[4]], axis=0)  # (3, N, 8)
for f in range(K):
    tr, va = folds != f, folds == f
    for j in range(8):
        y = Y_pool[tr, j]
        best_ap, best_w = -1, (1, 0, 0)
        for w1 in Ws:
            for w2 in Ws:
                if w1 + w2 > 1:
                    continue
                w4 = 1 - w1 - w2
                p = w1 * stack[0, tr, j] + w2 * stack[1, tr, j] + w4 * stack[2, tr, j]
                apw = average_precision_score(y, p)
                if apw > best_ap:
                    best_ap, best_w = apw, (w1, w2, w4)
        w_chosen[f, j] = best_w
        P_w[va, j] = best_w[0] * stack[0, va, j] + best_w[1] * stack[1, va, j] + best_w[2] * stack[2, va, j]

report("weighted(k=1,2,4) nested", P_w)

print("\nchosen (w1,w2,w4) mean over folds, per species:")
print(pd.DataFrame(w_chosen.mean(0), columns=["w_k1", "w_k2", "w_k4"],
                   index=[c.split()[0] for c in CLASSES]).to_string(float_format=lambda x: f"{x:.2f}"))

df = pd.DataFrame(rows)
print("\n=== summary (this arch's own k values, for internal comparison) ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== for context: other project milestones ===")
print("ground alone (separate arch):        macro AP 0.811")
print("weighted-avg late fusion (current best): macro AP 0.823")
print("\n=== per-class AP ===")
print(pd.DataFrame(apr).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/k_ensemble.csv", index=False)
pd.DataFrame(apr).to_csv("results/k_ensemble_perclass_ap.csv", index=False)
print("\nwrote results/k_ensemble.csv")
