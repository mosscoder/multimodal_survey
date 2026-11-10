"""Assemble + standardize the CV data from the feats/ blocks.

- pool  = odd (survey) legs of train, with the leg-fold id from folds_leg5.json
- Y_pool = 8-way multi-hot species labels, row-aligned to pool
- features(split, modality, rows=pool.index) gives the design matrix
- standardize() is applied per training fold (train stats only) inside CV
"""
import json
import numpy as np
import pandas as pd
from datasets import load_dataset

D = {"general": 768, "satellite": 1024}

def load_block(split_name, column, key, pooling="cls"):
    F = np.load(f"feats/{split_name}.{column}.{key}.npy")     # columns [:D] are CLS, [D:] are the patch mean
    d = D[key]
    return {"cls": F[:, :d], "patch": F[:, d:], "both": F}[pooling]

BLOCKS = {"ground": [("ground_image", "general")],
          "drone":  [("drone_image",  "satellite")],
          "concat": [("ground_image", "general"), ("drone_image", "satellite")]}   # 768 + 1024 = 1,792

def features(split_name, modality, rows=None):
    X = np.concatenate([load_block(split_name, c, k) for c, k in BLOCKS[modality]], axis=1)
    return X if rows is None else X[rows]                      # rows = pool.index for the CV pool

def standardize(X_tr, X_va):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6                  # training-fold statistics only
    return (X_tr - mu) / sd, (X_va - mu) / sd


# ---- assemble -------------------------------------------------------------
train = load_dataset("mpg-ranch/multimodal-survey", split="train")
CLASSES = train.features["species"].feature.names
meta = pd.read_parquet("feats/train.meta.parquet")            # 6480 rows, same order as the .npy blocks

Y = np.zeros((len(meta), len(CLASSES)), dtype=np.int8)
for i, ids in enumerate(meta["species"]):
    Y[i, list(ids)] = 1

leg_no = meta["leg"].str[3:].astype(int)
meta["group"] = meta["site"] + "/" + meta["strip"] + "/" + meta["leg"]
pool = meta[leg_no % 2 == 1].copy()

folds = json.load(open("folds_leg5.json"))
pool["fold"] = pool["group"].map(folds["fold_of_group"])
assert pool["fold"].notna().all(), "some survey legs missing from folds_leg5.json"
pool["fold"] = pool["fold"].astype(int)

Y_pool = Y[pool.index]
K = folds["k"]

if __name__ == "__main__":
    print(f"pool: {len(pool)} frames on {pool['group'].nunique()} survey legs, "
          f"folds {dict(sorted(pool['fold'].value_counts().items()))}")
    print(f"Y_pool: {Y_pool.shape}  positives/class {Y_pool.sum(0).tolist()}")
    for mod in BLOCKS:
        X = features("train", mod, rows=pool.index)
        print(f"  X[{mod:6s}] {X.shape}")
    # demo: standardize fold 0 as the held-out fold
    X = features("train", "concat", rows=pool.index)
    va = pool["fold"].values == 0
    Xtr, Xva = standardize(X[~va], X[va])
    print(f"fold-0 split  train {Xtr.shape} (mean {Xtr.mean():+.2e} std {Xtr.std():.3f})  "
          f"val {Xva.shape} (mean {Xva.mean():+.2e} std {Xva.std():.3f})")
