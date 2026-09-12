"""Effect of GROUND-image extraction resolution on the 5-fold leg CV.

Ground block: swept over 224 / 512 / 768   (feats/train.ground_image.general[.r512|.r768].npy)
Drone  block: fixed at native 320           (feats/train.drone_image.<key>.r320.npy)
concat = ground(res) + drone(320).  Files are [N, 2D] = [CLS | mean-patch].
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool, standardize
from head_cv import fit_predict, CLASSES

folds = pool["fold"].values
D = {"general": 768, "satellite": 1024}
GROUND_RES = (224, 512, 768)
DRONE_RES = 320


def load_block(column, key, res, pooling):
    tag = "" if res == 224 else f".r{res}"
    F = np.load(f"feats/train.{column}.{key}{tag}.npy")
    d = D[key]
    return {"cls": F[:, :d], "patch": F[:, d:], "both": F}[pooling]


def ground(res, pooling):
    return load_block("ground_image", "general", res, pooling)[pool.index]

def drone(pooling):
    return load_block("drone_image", "satellite", DRONE_RES, pooling)[pool.index]

def concat(res, pooling):
    return np.concatenate([load_block("ground_image", "general", res, pooling),
                           load_block("drone_image", "satellite", DRONE_RES, pooling)], 1)[pool.index]


def cv_oof(X, kind, lr, seed=0):
    oof = np.zeros_like(Y_pool, dtype=np.float32)
    for k in np.unique(folds):
        va = folds == k
        Xtr, Xva = standardize(X[~va], X[va])
        P, _ = fit_predict(Xtr, Y_pool[~va], Xva, Y_pool[va], kind, lr, seed=seed)
        oof[va] = P
    return oof


def score(Y, P):
    per_f1 = f1_score(Y, P > 0.5, average=None, zero_division=0)
    per_ap = average_precision_score(Y, P, average=None)
    fold_f1 = [f1_score(Y[folds == k], P[folds == k] > 0.5, average="macro", zero_division=0)
               for k in np.unique(folds)]
    return per_f1, per_ap, np.mean(fold_f1), np.std(fold_f1)


rows, ap_rows = [], []


def run(tag, X, kind, lr):
    oof = cv_oof(X, kind, lr)
    per_f1, per_ap, fm, fsd = score(Y_pool, oof)
    rows.append(dict(config=tag, d=X.shape[1], macro_F1=per_f1.mean(), macro_AP=per_ap.mean(),
                     perfold_F1=fm, perfold_sd=fsd))
    ap_rows.append(dict(config=tag, **{c.split()[1][:6]: v for c, v in zip(CLASSES, per_ap)}))
    print(f"{tag:34s} d={X.shape[1]:4d}  macroF1 {per_f1.mean():.3f}  macroAP {per_ap.mean():.3f}  "
          f"perfold {fm:.3f}+/-{fsd:.3f}", flush=True)


# drone-only (native 320) — one row per head/pooling
for kind, lr in (("linear", 1e-2), ("mlp", 1e-3)):
    for pooling in ("cls", "both"):
        run(f"drone/{kind}/{pooling}/r320", drone(pooling), kind, lr)

# ground-only and concat, swept over ground resolution
for res in GROUND_RES:
    for kind, lr in (("linear", 1e-2), ("mlp", 1e-3)):
        for pooling in ("cls", "both"):
            run(f"ground/{kind}/{pooling}/g{res}", ground(res, pooling), kind, lr)
            run(f"concat/{kind}/{pooling}/g{res}", concat(res, pooling), kind, lr)

df = pd.DataFrame(rows)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP ===")
print(pd.DataFrame(ap_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/res_compare.csv", index=False)
pd.DataFrame(ap_rows).to_csv("results/res_compare_perclass_ap.csv", index=False)
print("\nwrote results/res_compare.csv + results/res_compare_perclass_ap.csv")
