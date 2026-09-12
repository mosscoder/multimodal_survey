import json
import numpy as np
from datasets import load_dataset

train = load_dataset("mpg-ranch/multimodal-survey", split="train")
meta = train.remove_columns(["ground_image", "drone_image"]).to_pandas()
leg_no = meta["leg"].str[3:].astype(int)
meta["group"] = meta["site"] + "/" + meta["strip"] + "/" + meta["leg"]
pool = meta[leg_no % 2 == 1].copy()

folds = json.load(open("folds_leg5.json"))
pool["fold"] = pool["group"].map(folds["fold_of_group"])
K = folds["k"]


# Training mask for leg fold k. buffered=True also withholds the survey legs 5 m either side of a held-out leg.
def neighbours(group):
    site, strip, leg = group.split("/"); n = int(leg[3:])
    return {f"{site}/{strip}/leg{n-2:02d}", f"{site}/{strip}/leg{n+2:02d}"}

def train_mask(pool, k, col="fold", buffered=False):
    held = set(pool.loc[pool[col] == k, "group"])
    drop = set(held)
    if buffered:
        for g in held:
            drop |= neighbours(g)
    return ~pool["group"].isin(drop)


print(f"{'fold':>4}  {'val':>5}  {'train':>6}  {'train_buf':>9}  {'buffered_out':>12}")
for k in range(K):
    m = train_mask(pool, k)
    mb = train_mask(pool, k, buffered=True)
    val = int((pool["fold"] == k).sum())
    print(f"{k:>4}  {val:>5}  {int(m.sum()):>6}  {int(mb.sum()):>9}  {int(m.sum() - mb.sum()):>12}")

# sanity: how many real neighbour legs exist in the pool for each held-out fold
for k in range(K):
    held = set(pool.loc[pool["fold"] == k, "group"])
    nbrs = set().union(*(neighbours(g) for g in held)) - held
    present = nbrs & set(pool["group"])
    print(f"fold {k}: {len(held)} held legs, {len(present)} neighbour legs also in pool")
