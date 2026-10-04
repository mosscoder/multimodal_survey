"""2-way late fusion (ground-alone + drone-alone): per-class CV results, then
applied to the held-out test set using already-trained final models (no
retraining -- reuses results/test_component_preds.npz from the 3-way test run).
"""
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
folds = pool["fold"].values
K = 5
SEEDS = [0, 1, 2]

# ---- CV: nested 5-fold weighted average, ground + drone only ----
d = np.load("results/armhead_ablation_oof.npz")
ground_oof = np.mean([d[f"ground_linear_s{s}"] for s in SEEDS], axis=0)
drone_oof = np.mean([d[f"drone_linear_s{s}"] for s in SEEDS], axis=0)

Ws = np.linspace(0, 1, 21)
P_cv = np.zeros(Y_pool.shape, dtype=np.float32)
w_per_fold = np.zeros((K, 8))
for f in range(K):
    tr, va = folds != f, folds == f
    for j in range(8):
        y = Y_pool[tr, j]
        best_ap, best_w = -1, 0.5
        for w in Ws:
            apw = average_precision_score(y, w * ground_oof[tr, j] + (1 - w) * drone_oof[tr, j])
            if apw > best_ap:
                best_ap, best_w = apw, w
        w_per_fold[f, j] = best_w
        P_cv[va, j] = best_w * ground_oof[va, j] + (1 - best_w) * drone_oof[va, j]

cv_f1 = f1_score(Y_pool, P_cv > 0.5, average=None, zero_division=0)
cv_ap = average_precision_score(Y_pool, P_cv, average=None)
print("=== CV (2-way late fusion, nested 5-fold, per-class) ===")
print(f"{'species':24s} {'AP':>7s} {'F1@0.5':>8s}")
for j, c in enumerate(CLASSES):
    print(f"{c:24s} {cv_ap[j]:7.3f} {cv_f1[j]:8.3f}")
print(f"{'MACRO':24s} {cv_ap.mean():7.3f} {cv_f1.mean():8.3f}")

# ---- final weights for test: refit on the FULL pool (no leakage into test) ----
w_final = np.zeros(8)
for j in range(8):
    y = Y_pool[:, j]
    best_ap, best_w = -1, 0.5
    for w in Ws:
        apw = average_precision_score(y, w * ground_oof[:, j] + (1 - w) * drone_oof[:, j])
        if apw > best_ap:
            best_ap, best_w = apw, w
    w_final[j] = best_w

# ---- apply to TEST using the already-trained final models ----
t = np.load("results/test_component_preds.npz")
ground_test, drone_test = t["ground_grid"], t["drone"]
P_test = w_final[None, :] * ground_test + (1 - w_final[None, :]) * drone_test

test_ds = load_dataset("mpg-ranch/multimodal-survey", split="test")
Y_test = np.zeros((len(test_ds), 8), dtype=np.int8)
for i, ids in enumerate(test_ds["species"]):
    Y_test[i, list(ids)] = 1

test_ap = average_precision_score(Y_test, P_test, average=None)
test_f1 = f1_score(Y_test, P_test > 0.5, average=None, zero_division=0)

print(f"\nfinal weights (full pool, ground share): " +
     ", ".join(f"{c.split()[0]}={w:.2f}" for c, w in zip(CLASSES, w_final)))

print(f"\n=== TEST (2-way late fusion, n={len(test_ds)}) ===")
print(f"{'species':24s} {'AP':>7s} {'F1@0.5':>8s}")
for j, c in enumerate(CLASSES):
    print(f"{c:24s} {test_ap[j]:7.3f} {test_f1[j]:8.3f}")
print(f"{'MACRO':24s} {test_ap.mean():7.3f} {test_f1.mean():8.3f}")

print(f"\n=== CV vs TEST, per class ===")
print(f"{'species':24s} {'CV AP':>7s} {'Test AP':>8s} {'delta':>7s}   {'CV F1':>7s} {'Test F1':>8s} {'delta':>7s}")
for j, c in enumerate(CLASSES):
    print(f"{c:24s} {cv_ap[j]:7.3f} {test_ap[j]:8.3f} {test_ap[j]-cv_ap[j]:+7.3f}   "
          f"{cv_f1[j]:7.3f} {test_f1[j]:8.3f} {test_f1[j]-cv_f1[j]:+7.3f}")
print(f"{'MACRO':24s} {cv_ap.mean():7.3f} {test_ap.mean():8.3f} {test_ap.mean()-cv_ap.mean():+7.3f}   "
      f"{cv_f1.mean():7.3f} {test_f1.mean():8.3f} {test_f1.mean()-cv_f1.mean():+7.3f}")

pd.DataFrame({"species": CLASSES, "cv_AP": cv_ap, "cv_F1": cv_f1,
             "test_AP": test_ap, "test_F1": test_f1}).to_csv("results/2way_cv_vs_test.csv", index=False)
print("\nwrote results/2way_cv_vs_test.csv")
