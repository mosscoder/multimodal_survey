"""Learning-rate sweep for linear + mlp heads on concat / 5-fold leg CV."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool, features, standardize
from head_cv import fit_predict, CLASSES

folds = pool["fold"].values


def report(Y, P, verbose=True):
    per = f1_score(Y, P > 0.5, average=None, zero_division=0)
    macro = per.mean()
    if verbose:
        ap = average_precision_score(Y, P, average=None)
        for name, f, a in zip(CLASSES, per, ap):
            print(f"{name:24s} F1 {f:.3f}   AP {a:.3f}")
        print(f"{'macro average':24s} F1 {macro:.3f}   AP {ap.mean():.3f}")
    return per, macro


def per_fold(Y, P, folds, thresholds=0.5):
    scores = [report(Y[folds == k], P[folds == k], verbose=False)[1] for k in np.unique(folds)]
    return np.mean(scores), np.std(scores)


def cv_predict(X, kind, lr, seed=0):
    oof = np.zeros_like(Y_pool, dtype=np.float32)
    eps = []
    for k in np.unique(folds):
        va = folds == k
        Xtr, Xva = standardize(X[~va], X[va])
        P, ep = fit_predict(Xtr, Y_pool[~va], Xva, Y_pool[va], kind, lr, seed=seed)
        oof[va] = P
        eps.append(ep)
    return oof, int(np.median(eps))


GRID = {"linear": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        "mlp":    [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]}
X = features("train", "concat", rows=pool.index)

rows = []
for kind, lrs in GRID.items():
    for lr in lrs:
        seeds = [0] if kind == "linear" else [0, 1, 2]
        for seed in seeds:
            oof, best_ep = cv_predict(X, kind, lr, seed=seed)
            per, macro = report(Y_pool, oof, verbose=False)
            mean_k, sd_k = per_fold(Y_pool, oof, folds)
            rows.append({"head": kind, "lr": lr, "seed": seed, "macro_f1": macro,
                         "fold_mean": mean_k, "fold_sd": sd_k, "median_best_epoch": best_ep,
                         **{c.split()[0]: f for c, f in zip(CLASSES, per)}})
            print(f"  {kind:6s} lr {lr:.0e} seed {seed}  macroF1 {macro:.3f}  "
                  f"fold {mean_k:.3f}+/-{sd_k:.3f}  ep {best_ep}", flush=True)

table = pd.DataFrame(rows)
agg = table.groupby(["head", "lr"]).agg(macro_f1=("macro_f1", "mean"),
                                        fold_sd=("fold_sd", "mean"),
                                        epochs=("median_best_epoch", "median")).round(3)
print("\n=== lr sweep (mean over seeds) ===")
print(agg)

# per-class F1 at each (head, lr), averaged over seeds
sp = [c.split()[0] for c in CLASSES]
print("\n=== per-class F1 ===")
print(table.groupby(["head", "lr"])[sp].mean().round(3))

table.to_csv("results/lr_grid.csv", index=False)
print("\nwrote results/lr_grid.csv")
