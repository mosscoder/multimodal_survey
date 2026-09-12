"""Run the leg-fold CV with the user's canonical report() / per_fold()."""
import numpy as np
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool, features, standardize
from head_cv import fit_predict, CLASSES

folds = pool["fold"].values


def report(Y, P, thresholds=0.5, verbose=True):
    pred = P > thresholds                                      # scalar or per-species array
    per = f1_score(Y, pred, average=None, zero_division=0)     # one-vs-all F1 per species
    macro = per.mean()                                          # identical to average="macro"
    ap = average_precision_score(Y, P, average=None)            # threshold-free companion
    if verbose:
        for name, f, a in zip(CLASSES, per, ap):
            print(f"{name:24s} F1 {f:.3f}   AP {a:.3f}")
        print(f"{'macro average':24s} F1 {macro:.3f}   AP {ap.mean():.3f}")
    return per, macro


def per_fold(Y, P, folds, thresholds=0.5):
    scores = [report(Y[folds == k], P[folds == k], thresholds, verbose=False)[1] for k in np.unique(folds)]
    return np.mean(scores), np.std(scores)                     # macro F1 mean and sd across folds


def cv_oof(modality, kind, lr, seed=0):
    X = features("train", modality, rows=pool.index)
    oof = np.zeros_like(Y_pool, dtype=np.float32)
    for k in np.unique(folds):
        va = folds == k
        Xtr, Xva = standardize(X[~va], X[va])
        P, _ = fit_predict(Xtr, Y_pool[~va], Xva, Y_pool[va], kind, lr, seed=seed)
        oof[va] = P
    return oof


CONFIGS = [
    ("concat", "linear", 1e-2),
    ("ground", "linear", 1e-2),
    ("drone",  "linear", 1e-2),
    ("concat", "mlp",    1e-3),
    ("ground", "mlp",    1e-3),
    ("drone",  "mlp",    1e-3),
]

for mod, kind, lr in CONFIGS:
    print(f"\n===== {mod} x {kind}  (lr {lr:.0e}) =====")
    oof = cv_oof(mod, kind, lr)
    per, macro = report(Y_pool, oof, verbose=True)
    mean_k, sd_k = per_fold(Y_pool, oof, folds)
    print(f"{'per-fold macro F1':24s} {mean_k:.3f} +/- {sd_k:.3f}")
