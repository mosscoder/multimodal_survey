"""Compare CLS-only vs CLS+mean-patch ('both') features on the 5-fold leg CV.

Same heads (linear / mlp), same folds, pooled OOF. Prints macro F1 / macro AP
and per-class AP for every (modality, head, pooling).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool, standardize
import probe_lib
from head_cv import fit_predict, CLASSES

folds = pool["fold"].values
D = {"general": 768, "satellite": 1024}
BLOCKS = {"ground": [("ground_image", "general")],
          "drone":  [("drone_image",  "satellite")],
          "concat": [("ground_image", "general"), ("drone_image", "satellite")]}


def load_block(column, key, pooling):
    F = np.load(f"feats/train.{column}.{key}.npy")
    d = D[key]
    return {"cls": F[:, :d], "patch": F[:, d:], "both": F}[pooling]


def features(modality, pooling, rows):
    X = np.concatenate([load_block(c, k, pooling) for c, k in BLOCKS[modality]], axis=1)
    return X[rows]


def report(Y, P):
    per = f1_score(Y, P > 0.5, average=None, zero_division=0)
    ap = average_precision_score(Y, P, average=None)
    return per, per.mean(), ap, ap.mean()


def per_fold(Y, P):
    s = [f1_score(Y[folds == k], P[folds == k] > 0.5, average="macro", zero_division=0)
         for k in np.unique(folds)]
    return np.mean(s), np.std(s)


def cv_oof(modality, pooling, kind, lr, seed=0):
    X = features(modality, pooling, pool.index)
    oof = np.zeros_like(Y_pool, dtype=np.float32)
    for k in np.unique(folds):
        va = folds == k
        Xtr, Xva = standardize(X[~va], X[va])
        P, _ = fit_predict(Xtr, Y_pool[~va], Xva, Y_pool[va], kind, lr, seed=seed)
        oof[va] = P
    return oof


CONFIGS = [(m, h, lr) for m in ("concat", "ground", "drone")
           for h, lr in (("linear", 1e-2), ("mlp", 1e-3))]

summary, per_ap_rows = [], []
for mod, kind, lr in CONFIGS:
    for pooling in ("cls", "both"):
        X = features(mod, pooling, pool.index)
        oof = cv_oof(mod, pooling, kind, lr)
        per, macro_f1, ap, macro_ap = report(Y_pool, oof)
        mean_k, sd_k = per_fold(Y_pool, oof)
        summary.append(dict(modality=mod, head=kind, pooling=pooling, d=X.shape[1],
                            macro_F1=macro_f1, macro_AP=macro_ap,
                            perfold_F1=mean_k, perfold_sd=sd_k))
        per_ap_rows.append(dict(tag=f"{mod}/{kind}/{pooling}",
                                **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap)}))
        print(f"{mod:7s} {kind:6s} {pooling:5s} d={X.shape[1]:4d}  "
              f"macroF1 {macro_f1:.3f}  macroAP {macro_ap:.3f}  perfold {mean_k:.3f}+/-{sd_k:.3f}",
              flush=True)

df = pd.DataFrame(summary)
print("\n=== summary (cls vs both) ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

piv = df.pivot_table(index=["modality", "head"], columns="pooling",
                     values=["macro_F1", "macro_AP"])
print("\n=== delta (both - cls) ===")
for (mod, head), r in piv.iterrows():
    print(f"{mod:7s} {head:6s}  dF1 {r[('macro_F1','both')]-r[('macro_F1','cls')]:+.3f}   "
          f"dAP {r[('macro_AP','both')]-r[('macro_AP','cls')]:+.3f}")

print("\n=== per-class AP ===")
print(pd.DataFrame(per_ap_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/pooling_compare.csv", index=False)
print("\nwrote results/pooling_compare.csv")
