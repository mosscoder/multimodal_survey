import json, numpy as np
from datasets import load_dataset

train = load_dataset("mpg-ranch/multimodal-survey", split="train")
CLASSES = train.features["species"].feature.names          # canonical 8-class order
meta = train.remove_columns(["ground_image", "drone_image"]).to_pandas()

Y = np.zeros((len(meta), len(CLASSES)), dtype=np.int8)      # multi-hot labels
for i, ids in enumerate(meta["species"]):
    Y[i, list(ids)] = 1

leg_no = meta["leg"].str[3:].astype(int)
meta["group"] = meta["site"] + "/" + meta["strip"] + "/" + meta["leg"]
pool = meta[leg_no % 2 == 1]             # survey legs only (odd numbers); even legs are the turns

groups = pool["group"].unique()          # one group per leg; both capture events share it
gpos   = {g: Y[pool.index[pool["group"] == g]].sum(0) for g in groups}
gsize  = pool["group"].value_counts().to_dict()

K = 5
target   = Y[pool.index].sum(0) / K      # positives per species in a balanced fold
target_n = len(pool) / K                 # frames in a balanced fold
order = sorted(groups, key=lambda g: -(gpos[g] / target).sum())   # legs carrying rare species first
load, n_in, fold_of = np.zeros((K, len(CLASSES))), np.zeros(K), {}
for g in order:
    cost = (((load + gpos[g]) / target) ** 2).sum(1) + ((n_in + gsize[g]) / target_n) ** 2
    k = int(np.argmin(cost))
    fold_of[g] = k; load[k] += gpos[g]; n_in[k] += gsize[g]

pool = pool.assign(fold=pool["group"].map(fold_of))
json.dump({"k": K, "classes": CLASSES, "fold_of_group": fold_of},
          open("folds_leg5.json", "w"), indent=1)

print(pool.groupby("fold").size().to_dict())                # frames per fold
print(load.astype(int))                                     # positives per species per fold
