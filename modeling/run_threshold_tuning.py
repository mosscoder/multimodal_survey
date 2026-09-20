"""Per-class threshold tuning on the current best fused predictions (ground+drone
weighted-avg late fusion). No retraining -- AP is threshold-free so it can't
change; this only tests whether F1 improves at a tuned cutoff vs the blind 0.5.
Nested 5-fold: tune each class's threshold on 4 folds, apply to the 5th.
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

# rebuild the current-best fused OOF (ground+drone weighted avg), same recipe as run_late_fusion.py
d = np.load("results/armhead_ablation_oof.npz")
ground = np.mean([d[f"ground_linear_s{s}"] for s in SEEDS], axis=0)
drone = np.mean([d[f"drone_linear_s{s}"] for s in SEEDS], axis=0)

Ws = np.linspace(0, 1, 21)
P = np.zeros(Y_pool.shape, dtype=np.float32)
for f in range(K):
    tr, va = folds != f, folds == f
    for j in range(8):
        y = Y_pool[tr, j]
        best_ap, best_w = -1, 0.5
        for w in Ws:
            apw = average_precision_score(y, w * ground[tr, j] + (1 - w) * drone[tr, j])
            if apw > best_ap:
                best_ap, best_w = apw, w
        P[va, j] = best_w * ground[va, j] + (1 - best_w) * drone[va, j]

ap = average_precision_score(Y_pool, P, average=None)
f1_05 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
print(f"reconstructed fused OOF: macro AP {ap.mean():.3f} (should match ~0.823)  "
      f"macro F1@0.5 {f1_05.mean():.3f} (should match ~0.763)")

# ---- nested per-class threshold tuning ----
Ts = np.linspace(0.05, 0.95, 37)
pred_tuned = np.zeros(Y_pool.shape, dtype=int)
t_chosen = np.zeros((K, 8))
for f in range(K):
    tr, va = folds != f, folds == f
    for j in range(8):
        y = Y_pool[tr, j]
        p = P[tr, j]
        best_f1, best_t = -1, 0.5
        for t in Ts:
            f1t = f1_score(y, p > t, zero_division=0)
            if f1t > best_f1:
                best_f1, best_t = f1t, t
        t_chosen[f, j] = best_t
        pred_tuned[va, j] = (P[va, j] > best_t).astype(int)

f1_tuned = f1_score(Y_pool, pred_tuned, average=None, zero_division=0)
ff_05 = [f1_score(Y_pool[folds == k], P[folds == k] > 0.5, average="macro", zero_division=0) for k in range(K)]
ff_tuned = [f1_score(Y_pool[folds == k], pred_tuned[folds == k], average="macro", zero_division=0) for k in range(K)]

print(f"\n{'species':24s} {'F1@0.5':>8s} {'F1@tuned':>9s} {'delta':>7s}   mean_t")
for j, c in enumerate(CLASSES):
    print(f"{c:24s} {f1_05[j]:8.3f} {f1_tuned[j]:9.3f} {f1_tuned[j]-f1_05[j]:+7.3f}   {t_chosen[:, j].mean():.2f}")
print(f"{'MACRO':24s} {f1_05.mean():8.3f} {f1_tuned.mean():9.3f} {f1_tuned.mean()-f1_05.mean():+7.3f}")
print(f"\nper-fold macro F1: @0.5 = {np.mean(ff_05):.3f}+/-{np.std(ff_05):.3f}   "
      f"@tuned = {np.mean(ff_tuned):.3f}+/-{np.std(ff_tuned):.3f}")

pd.DataFrame({"species": CLASSES, "F1_at_0.5": f1_05, "F1_at_tuned": f1_tuned,
             "delta": f1_tuned - f1_05, "mean_threshold": t_chosen.mean(0)}).to_csv(
    "results/threshold_tuning.csv", index=False)
pd.DataFrame(t_chosen, columns=[c.split()[0] for c in CLASSES]).to_csv(
    "results/threshold_tuning_per_fold.csv", index=False)
print("\nwrote results/threshold_tuning.csv (+ per_fold thresholds)")
