"""Assemble the cross-validation pool, its labels and its folds.

- pool   = the training-split ground images on the legs (odd leg numbers, the
           even ones are the turns), 6,089 rows, each with the fold of its leg
           from folds_leg5.json (written by make_folds.py)
- Y_pool = eight-way multi-hot species labels, row-aligned to pool

The table comes straight from the dataset, in the dataset's row order, which is
also the row order of the feature files in feats/.
"""
import json
import numpy as np
import pandas as pd
from datasets import load_dataset

train = load_dataset("mpg-ranch/multimodal-survey", split="train")
CLASSES = train.features["species"].feature.names
meta = train.remove_columns(["ground_image", "drone_image"]).to_pandas()   # 6,480 rows

Y = np.zeros((len(meta), len(CLASSES)), dtype=np.int8)
for i, ids in enumerate(meta["species"]):
    Y[i, list(ids)] = 1

leg_no = meta["leg"].str[3:].astype(int)
meta["group"] = meta["site"] + "/" + meta["strip"] + "/" + meta["leg"]
pool = meta[leg_no % 2 == 1].copy()

folds = json.load(open("folds_leg5.json"))
pool["fold"] = pool["group"].map(folds["fold_of_group"])
assert pool["fold"].notna().all(), "some legs are missing from folds_leg5.json"
pool["fold"] = pool["fold"].astype(int)

Y_pool = Y[pool.index]
K = folds["k"]

if __name__ == "__main__":
    print(f"pool: {len(pool)} ground images on {pool['group'].nunique()} legs, "
          f"folds {dict(sorted(pool['fold'].value_counts().items()))}")
    print(f"Y_pool: {Y_pool.shape}  positives per species {Y_pool.sum(0).tolist()}")
