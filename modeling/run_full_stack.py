"""Combine every independent signal: dense-grid top-1 MIL ground, global-pool
768px ground, drone-alone CLS. Nested 5-fold 3-way weighted average per class
(stacking LR included for comparison, expected to underperform per the
established pattern). Then layer per-class threshold tuning on top of the
winning combination for the final number.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression

from assemble import pool, Y_pool

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
folds = pool["fold"].values
K = 5
SEEDS = [0, 1, 2]

d = np.load("results/armhead_ablation_oof.npz")
ground_grid = np.mean([d[f"ground_linear_s{s}"] for s in SEEDS], axis=0)
drone = np.mean([d[f"drone_linear_s{s}"] for s in SEEDS], axis=0)
r = np.load("results/res_ensemble_oof.npz")
ground_global = r["ground_global"]

sigs = {"grid": ground_grid, "global": ground_global, "drone": drone}


def score(P):
    f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
    ap = average_precision_score(Y_pool, P, average=None)
    ff = [f1_score(Y_pool[folds == j], P[folds == j] > 0.5, average="macro", zero_division=0) for j in range(K)]
    return f1, ap, np.mean(ff), np.std(ff)


rows, apr = [], []
def report(tag, P):
    f1, ap, fm, fsd = score(P)
    print(f"{tag:32s} macroF1 {f1.mean():.3f}  macroAP {ap.mean():.3f}  perfold {fm:.3f}+/-{fsd:.3f}")
    rows.append(dict(config=tag, macro_F1=f1.mean(), macro_AP=ap.mean(), perfold_F1=fm, perfold_sd=fsd))
    apr.append(dict(config=tag, **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap)}))
    return f1, ap


for name, P in sigs.items():
    report(f"{name} alone", P)

# ---- 3-way nested weighted average (w_grid, w_global, w_drone), simplex grid search ----
Ws = np.linspace(0, 1, 11)
P_3way = np.zeros(Y_pool.shape, dtype=np.float32)
w_chosen = np.zeros((K, 8, 3))
for f in range(K):
    tr, va = folds != f, folds == f
    for j in range(8):
        y = Y_pool[tr, j]
        best_ap, best_w = -1, (1, 0, 0)
        for wg in Ws:
            for wl in Ws:
                if wg + wl > 1:
                    continue
                wd_ = 1 - wg - wl
                p = wg * ground_grid[tr, j] + wl * ground_global[tr, j] + wd_ * drone[tr, j]
                apw = average_precision_score(y, p)
                if apw > best_ap:
                    best_ap, best_w = apw, (wg, wl, wd_)
        w_chosen[f, j] = best_w
        wg, wl, wd_ = best_w
        P_3way[va, j] = wg * ground_grid[va, j] + wl * ground_global[va, j] + wd_ * drone[va, j]

report("3-way weighted avg (grid+global+drone)", P_3way)

# stacking on all 3 (24-d input), for comparison
X = np.concatenate([ground_grid, ground_global, drone], axis=1)
P_stack = np.zeros(Y_pool.shape, dtype=np.float32)
for f in range(K):
    tr, va = folds != f, folds == f
    for j in range(8):
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        clf.fit(X[tr], Y_pool[tr, j])
        P_stack[va, j] = clf.predict_proba(X[va])[:, 1]
report("stacking LR (all 3, 24-d)", P_stack)

print("\nchosen weights (w_grid, w_global, w_drone), mean over folds:")
print(pd.DataFrame(w_chosen.mean(0), columns=["w_grid", "w_global", "w_drone"],
                   index=[c.split()[0] for c in CLASSES]).to_string(float_format=lambda x: f"{x:.2f}"))

# ---- best combination so far: 3-way weighted avg vs the earlier 2-way (grid+drone) ----
best_ap_so_far = max(rows, key=lambda r: r["macro_AP"])
print(f"\nbest pre-threshold combination: {best_ap_so_far['config']}  AP {best_ap_so_far['macro_AP']:.3f}")
P_best = P_3way  # use the 3-way weighted avg as the carrier for threshold tuning

# ---- layer per-class threshold tuning on top ----
Ts = np.linspace(0.05, 0.95, 37)
pred_tuned = np.zeros(Y_pool.shape, dtype=int)
t_chosen = np.zeros((K, 8))
for f in range(K):
    tr, va = folds != f, folds == f
    for j in range(8):
        y, p = Y_pool[tr, j], P_best[tr, j]
        best_f1, best_t = -1, 0.5
        for t in Ts:
            f1t = f1_score(y, p > t, zero_division=0)
            if f1t > best_f1:
                best_f1, best_t = f1t, t
        t_chosen[f, j] = best_t
        pred_tuned[va, j] = (P_best[va, j] > best_t).astype(int)

f1_tuned = f1_score(Y_pool, pred_tuned, average=None, zero_division=0)
ff_tuned = [f1_score(Y_pool[folds == k], pred_tuned[folds == k], average="macro", zero_division=0) for k in range(K)]
ap_best = average_precision_score(Y_pool, P_best, average=None)
print(f"\n=== FINAL: 3-way weighted avg + per-class tuned threshold ===")
print(f"macro AP {ap_best.mean():.3f} (unchanged by thresholding)   "
      f"macro F1@tuned {f1_tuned.mean():.3f}   perfold F1 {np.mean(ff_tuned):.3f}+/-{np.std(ff_tuned):.3f}")
rows.append(dict(config="3-way weighted avg + tuned threshold", macro_F1=f1_tuned.mean(),
                 macro_AP=ap_best.mean(), perfold_F1=np.mean(ff_tuned), perfold_sd=np.std(ff_tuned)))
apr.append(dict(config="3-way + tuned threshold",
                **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap_best)}))

df = pd.DataFrame(rows)
print("\n=== summary (all combinations tried) ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP ===")
print(pd.DataFrame(apr).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== reference: every major milestone ===")
print("ground alone (dense-grid top-1):              AP 0.811")
print("ground+drone weighted-avg late fusion:         AP 0.823")
print("grid+global768 weighted-avg (2-way):           AP 0.815")
df.to_csv("results/full_stack.csv", index=False)
pd.DataFrame(apr).to_csv("results/full_stack_perclass_ap.csv", index=False)
print("\nwrote results/full_stack.csv")
