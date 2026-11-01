"""2-way late fusion (ground-alone + drone-alone): per-class CV results, then
applied to the held-out test set using already-trained final models (no
retraining -- reuses results/preds/test_component_preds.npz from the 3-way test run).
"""
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

from assemble import pool, Y_pool

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
folds = pool["fold"].values
K = 5
SEEDS = [0, 1, 2]

# ---- CV: nested 5-fold weighted average, ground + drone only ----
d = np.load("results/preds/armhead_ablation_oof.npz")
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

def per_class(Y, P, thresh=0.5):
    ap = average_precision_score(Y, P, average=None)
    f1 = f1_score(Y, P > thresh, average=None, zero_division=0)
    prec = precision_score(Y, P > thresh, average=None, zero_division=0)
    rec = recall_score(Y, P > thresh, average=None, zero_division=0)
    return ap, f1, prec, rec


cv_ap, cv_f1, cv_P, cv_R = per_class(Y_pool, P_cv)
print("=== CV (2-way late fusion, nested 5-fold, per-class) ===")
print(f"{'species':24s} {'AP':>7s} {'F1@0.5':>8s} {'P':>7s} {'R':>7s}")
for j, c in enumerate(CLASSES):
    print(f"{c:24s} {cv_ap[j]:7.3f} {cv_f1[j]:8.3f} {cv_P[j]:7.3f} {cv_R[j]:7.3f}")
print(f"{'MACRO':24s} {cv_ap.mean():7.3f} {cv_f1.mean():8.3f} {cv_P.mean():7.3f} {cv_R.mean():7.3f}")

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
t = np.load("results/preds/test_component_preds.npz")
ground_test, drone_test = t["ground_grid"], t["drone"]
P_test = w_final[None, :] * ground_test + (1 - w_final[None, :]) * drone_test

test_ds = load_dataset("mpg-ranch/multimodal-survey", split="test")
Y_test = np.zeros((len(test_ds), 8), dtype=np.int8)
for i, ids in enumerate(test_ds["species"]):
    Y_test[i, list(ids)] = 1

test_ap, test_f1, test_P, test_R = per_class(Y_test, P_test)

print(f"\nfinal weights (full pool, ground share): " +
     ", ".join(f"{c.split()[0]}={w:.2f}" for c, w in zip(CLASSES, w_final)))

print(f"\n=== TEST (2-way late fusion, n={len(test_ds)}) ===")
print(f"{'species':24s} {'AP':>7s} {'F1@0.5':>8s} {'P':>7s} {'R':>7s}")
for j, c in enumerate(CLASSES):
    print(f"{c:24s} {test_ap[j]:7.3f} {test_f1[j]:8.3f} {test_P[j]:7.3f} {test_R[j]:7.3f}")
print(f"{'MACRO':24s} {test_ap.mean():7.3f} {test_f1.mean():8.3f} {test_P.mean():7.3f} {test_R.mean():7.3f}")

print(f"\n=== CV vs TEST, per class ===")
print(f"{'species':24s} {'CV AP':>7s} {'Test AP':>8s} {'delta':>7s}   {'CV F1':>7s} {'Test F1':>8s} {'delta':>7s}")
for j, c in enumerate(CLASSES):
    print(f"{c:24s} {cv_ap[j]:7.3f} {test_ap[j]:8.3f} {test_ap[j]-cv_ap[j]:+7.3f}   "
          f"{cv_f1[j]:7.3f} {test_f1[j]:8.3f} {test_f1[j]-cv_f1[j]:+7.3f}")
print(f"{'MACRO':24s} {cv_ap.mean():7.3f} {test_ap.mean():8.3f} {test_ap.mean()-cv_ap.mean():+7.3f}   "
      f"{cv_f1.mean():7.3f} {test_f1.mean():8.3f} {test_f1.mean()-cv_f1.mean():+7.3f}")

# ---- item 5: score ground-alone and drone-alone too, not just the fusion ----
rows = []
for name, P_c, P_t in [("ground", ground_oof, ground_test), ("drone", drone_oof, drone_test),
                        ("fused", P_cv, P_test)]:
    ap_c, f1_c, p_c, r_c = per_class(Y_pool, P_c)
    ap_t, f1_t, p_t, r_t = per_class(Y_test, P_t)
    for j, c in enumerate(CLASSES):
        rows.append(dict(config=name, species=c, cv_AP=ap_c[j], cv_F1=f1_c[j], cv_P=p_c[j], cv_R=r_c[j],
                          test_AP=ap_t[j], test_F1=f1_t[j], test_P=p_t[j], test_R=r_t[j]))
    rows.append(dict(config=name, species="MACRO", cv_AP=ap_c.mean(), cv_F1=f1_c.mean(),
                      cv_P=p_c.mean(), cv_R=r_c.mean(), test_AP=ap_t.mean(), test_F1=f1_t.mean(),
                      test_P=p_t.mean(), test_R=r_t.mean()))

print(f"\n=== ground-alone / drone-alone / fused, CV + TEST (AP, F1, P, R) ===")
for name in ["ground", "drone", "fused"]:
    m = [r for r in rows if r["config"] == name and r["species"] == "MACRO"][0]
    print(f"{name:8s} CV: AP {m['cv_AP']:.3f} F1 {m['cv_F1']:.3f} P {m['cv_P']:.3f} R {m['cv_R']:.3f}   "
          f"Test: AP {m['test_AP']:.3f} F1 {m['test_F1']:.3f} P {m['test_P']:.3f} R {m['test_R']:.3f}")

# NEW file only -- results/2way_cv_vs_test.csv (fused-only, AP/F1) is left untouched
# per the baseline-rerun ground rule (old vs new must stay comparable).
pd.DataFrame(rows).to_csv("results/2way_cv_vs_test_per_modality.csv", index=False)
print("\nwrote results/2way_cv_vs_test_per_modality.csv (ground/drone/fused, CV+test, AP/F1/P/R)")
print("(results/2way_cv_vs_test.csv left untouched, per the no-overwrite ground rule)")
