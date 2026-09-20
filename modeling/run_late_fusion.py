"""Late fusion of independently-trained ground-alone and drone-alone models.
Uses already-saved OOF probabilities (results/armhead_ablation_oof.npz) --
no retraining. Both methods use nested 5-fold validation: tune/fit on the
other 4 folds, apply to the held-out fold, rotate -- honest, full-pool OOF.

  1. Simple: P = w*P_ground + (1-w)*P_drone, w swept per class in [0,1].
  2. Stacking: per-class LogisticRegression(C=0.1, heavily L2-regularized)
     on the 16-d [P_ground(8), P_drone(8)] input, trained on OOF predictions.
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

d = np.load("results/armhead_ablation_oof.npz")
ground = np.mean([d[f"ground_linear_s{s}"] for s in (0, 1, 2)], axis=0)   # (N,8) ENS probs
drone = np.mean([d[f"drone_linear_s{s}"] for s in (0, 1, 2)], axis=0)


def score(P):
    per_f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
    per_ap = average_precision_score(Y_pool, P, average=None)
    ff = [f1_score(Y_pool[folds == k], P[folds == k] > 0.5, average="macro", zero_division=0)
          for k in range(K)]
    return per_f1, per_ap, np.mean(ff), np.std(ff)


def report(tag, P):
    f1, ap, fm, fsd = score(P)
    print(f"{tag:28s} macroF1 {f1.mean():.3f}  macroAP {ap.mean():.3f}  "
          f"perfold {fm:.3f}+/-{fsd:.3f}")
    return dict(config=tag, macro_F1=f1.mean(), macro_AP=ap.mean(),
                perfold_F1=fm, perfold_sd=fsd), ap


rows, apr = [], []
r, ap = report("ground alone (ref)", ground); rows.append(r)
apr.append(dict(config="ground alone", **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap)}))
r, ap = report("drone alone (ref)", drone); rows.append(r)
apr.append(dict(config="drone alone", **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap)}))

# ---- 1. simple per-class weighted average, nested 5-fold tuning ----
Ws = np.linspace(0, 1, 21)
P_wavg = np.zeros(Y_pool.shape, dtype=np.float32)
w_chosen = np.zeros((K, 8))
for f in range(K):
    tr, va = folds != f, folds == f
    for j in range(8):
        y = Y_pool[tr, j]
        best_ap, best_w = -1, 0.5
        for w in Ws:
            p = w * ground[tr, j] + (1 - w) * drone[tr, j]
            apw = average_precision_score(y, p)
            if apw > best_ap:
                best_ap, best_w = apw, w
        w_chosen[f, j] = best_w
        P_wavg[va, j] = best_w * ground[va, j] + (1 - best_w) * drone[va, j]

r, ap = report("weighted avg (nested-tuned w)", P_wavg); rows.append(r)
apr.append(dict(config="weighted avg", **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap)}))

print("\nchosen w per fold per class (w=1 -> pure ground, w=0 -> pure drone):")
print(pd.DataFrame(w_chosen, columns=[c.split()[0] for c in CLASSES],
                   index=[f"fold{f}" for f in range(K)]).to_string(float_format=lambda x: f"{x:.2f}"))

# ---- 2. stacking: per-class logistic regression on [P_ground, P_drone], nested 5-fold ----
X = np.concatenate([ground, drone], axis=1)  # (N, 16)
P_stack = np.zeros(Y_pool.shape, dtype=np.float32)
coefs = []
for f in range(K):
    tr, va = folds != f, folds == f
    for j in range(8):
        y = Y_pool[tr, j]
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        clf.fit(X[tr], y)
        P_stack[va, j] = clf.predict_proba(X[va])[:, 1]
        coefs.append(dict(fold=f, species=CLASSES[j], **{f"g_{k}": v for k, v in enumerate(clf.coef_[0, :8])},
                          **{f"d_{k}": v for k, v in enumerate(clf.coef_[0, 8:])}))

r, ap = report("stacking LR (C=0.1, nested)", P_stack); rows.append(r)
apr.append(dict(config="stacking LR", **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap)}))

df = pd.DataFrame(rows)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP ===")
print(pd.DataFrame(apr).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

df.to_csv("results/late_fusion.csv", index=False)
pd.DataFrame(apr).to_csv("results/late_fusion_perclass_ap.csv", index=False)
pd.DataFrame(w_chosen, columns=[c.split()[0] for c in CLASSES]).to_csv("results/late_fusion_weights.csv", index=False)
pd.DataFrame(coefs).to_csv("results/late_fusion_stacking_coefs.csv", index=False)
print("\nwrote results/late_fusion.csv (+ perclass_ap, weights, stacking_coefs)")
