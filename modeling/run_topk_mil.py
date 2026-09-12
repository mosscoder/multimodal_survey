"""Top-k patch pooling, batched: all 24 heads (8 k-values x 3 seeds) trained on ONE
shared batch stream per fold. Grid stays a memmap (no 28.7GB resident array);
per fold the val frames are pulled into RAM fp16 so per-epoch validation doesn't
re-hit the memmap 24x. 5-fold leg CV.
"""
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score, average_precision_score

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
KS = [1, 2, 4, 8, 16, 64, 256, None]
SEEDS = [0, 1, 2]

# ---- data (absolute frame index space, 0..6479) --------------------------
GRID = np.load("feats/train.ground_image.general.grid1024x768.npy", mmap_mode="r")  # (6480,3072,768) f16
A = np.load("feats/train.drone_image.satellite.full.npy")[:, :1024].astype(np.float32)  # aerial CLS

ds = load_dataset("mpg-ranch/multimodal-survey", split="train")
meta = ds.remove_columns(["ground_image", "drone_image"]).to_pandas()
Y = np.zeros((len(meta), 8), dtype=np.float32)
for i, ids in enumerate(meta["species"]):
    Y[i, list(ids)] = 1
leg_no = meta["leg"].str[3:].astype(int)
group = meta["site"] + "/" + meta["strip"] + "/" + meta["leg"]
import json
folds_map = json.load(open("folds_leg5.json"))["fold_of_group"]
fold_abs = np.array([folds_map.get(g, -1) if ln % 2 == 1 else -1
                     for g, ln in zip(group, leg_no)], dtype=int)
K = 5
print(f"pool frames {int((fold_abs >= 0).sum())}  per fold {[int((fold_abs==k).sum()) for k in range(K)]}",
      flush=True)

print("precomputing per-frame patch means (one memmap pass) ...", flush=True)
PMEAN = np.empty((len(GRID), 768), np.float32)
for i in range(0, len(GRID), 256):
    PMEAN[i:i+256] = np.asarray(GRID[i:i+256], dtype=np.float32).mean(1)


def grid_stats(rows):
    m = PMEAN[rows]
    return m.mean(0), m.std(0) + 1e-6


class TopKHead(nn.Module):
    def __init__(self, k, d_g=768, d_a=1024, n_out=8):
        super().__init__()
        self.k = k
        self.ground = nn.Linear(d_g, n_out, bias=False)
        self.aerial = nn.Linear(d_a, n_out)

    def forward(self, G, a):
        s = self.ground(G)
        k = s.shape[1] if self.k is None else min(self.k, s.shape[1])
        return s.topk(k, dim=1).values.mean(1) + self.aerial(a)


def fit_predict_mil(rows_tr, rows_va, lr=1e-2, wd=1e-2, epochs=60, patience=10, batch=64):
    mu_g, sd_g = grid_stats(rows_tr)
    mu_gt = torch.tensor(mu_g).cuda(); sd_gt = torch.tensor(sd_g).cuda()
    mu_a, sd_a = A[rows_tr].mean(0), A[rows_tr].std(0) + 1e-6
    a_tr = torch.tensor((A[rows_tr] - mu_a) / sd_a, dtype=torch.float32).cuda()
    a_va = torch.tensor((A[rows_va] - mu_a) / sd_a, dtype=torch.float32).cuda()
    Yt = torch.tensor(Y[rows_tr], dtype=torch.float32).cuda()
    Yva = Y[rows_va]
    p = Y[rows_tr].mean(0)
    pw = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20).cuda()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    Gva = np.asarray(GRID[rows_va])                       # (n_va, 3072, 768) f16 in RAM (~5-7 GB)

    def grid_batch(rows):
        g = torch.tensor(np.asarray(GRID[rows], dtype=np.float32)).cuda()
        return (g - mu_gt) / sd_gt

    def predict(m):
        m.eval()
        out = []
        with torch.no_grad():
            for j in range(0, len(rows_va), batch):
                g = torch.from_numpy(Gva[j:j+batch]).cuda().float()
                g = (g - mu_gt) / sd_gt
                out.append(torch.sigmoid(m(g, a_va[j:j+batch])).cpu().numpy())
        return np.concatenate(out)

    heads, opts, scheds, best = {}, {}, {}, {}
    for h in [(k, s) for k in KS for s in SEEDS]:
        torch.manual_seed(h[1]); heads[h] = TopKHead(h[0]).cuda()
        opts[h] = torch.optim.AdamW(heads[h].parameters(), lr=lr, weight_decay=wd)
        scheds[h] = torch.optim.lr_scheduler.CosineAnnealingLR(opts[h], epochs)
        best[h] = dict(f1=-1.0, state=None, bad=0)

    for ep in range(epochs):
        order = np.random.permutation(len(rows_tr))
        for j in range(0, len(order), batch):
            sel = order[j:j+batch]
            Gb, ab, yb = grid_batch(rows_tr[sel]), a_tr[sel], Yt[sel]
            for h, m in heads.items():
                if best[h]["bad"] >= patience:
                    continue
                m.train(); opts[h].zero_grad()
                loss_fn(m(Gb, ab), yb).backward(); opts[h].step()
        for h, m in heads.items():
            if best[h]["bad"] >= patience:
                continue
            scheds[h].step()
            f1 = f1_score(Yva, predict(m) > 0.5, average="macro", zero_division=0)
            if f1 > best[h]["f1"]:
                best[h].update(f1=f1, bad=0, state={n: v.clone() for n, v in m.state_dict().items()})
            else:
                best[h]["bad"] += 1
        live = sum(b["bad"] < patience for b in best.values())
        print(f"    ep {ep:2d}  live {live}/24", flush=True)
        if all(b["bad"] >= patience for b in best.values()):
            break

    out = {}
    for h, m in heads.items():
        m.load_state_dict(best[h]["state"]); m.eval()
        out[h] = predict(m)
    return out


# ---- CV ------------------------------------------------------------------
oof = {(k, s): np.zeros((len(Y), 8), np.float32) for k in KS for s in SEEDS}
for fk in range(K):
    va = np.where(fold_abs == fk)[0]
    tr = np.where((fold_abs >= 0) & (fold_abs != fk))[0]
    print(f"\n== fold {fk}  tr {len(tr)}  va {len(va)} ==", flush=True)
    res = fit_predict_mil(tr, va)
    for h, P in res.items():
        oof[h][va] = P

pool_mask = fold_abs >= 0
Yp = Y[pool_mask]
fp = fold_abs[pool_mask]
rows_out, ap_out = [], []
for k in KS:
    seed_P = [oof[(k, s)][pool_mask] for s in SEEDS]
    for label, P in list(zip([f"k={k} s{s}" for s in SEEDS], seed_P)) + [(f"k={k} ENS", np.mean(seed_P, 0))]:
        per_ap = average_precision_score(Yp, P, average=None)
        per_f1 = f1_score(Yp, P > 0.5, average=None, zero_division=0)
        ff = [f1_score(Yp[fp == j], P[fp == j] > 0.5, average="macro", zero_division=0) for j in range(K)]
        rows_out.append(dict(config=label, macro_F1=per_f1.mean(), macro_AP=per_ap.mean(),
                             perfold_F1=np.mean(ff), perfold_sd=np.std(ff)))
        if label.endswith("ENS"):
            ap_out.append(dict(config=label, **{c.split()[1][:6]: v for c, v in zip(CLASSES, per_ap)}))
        print(f"{label:12s} macroF1 {per_f1.mean():.3f}  macroAP {per_ap.mean():.3f}  "
              f"perfold {np.mean(ff):.3f}+/-{np.std(ff):.3f}", flush=True)

df = pd.DataFrame(rows_out)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP (seed ensemble) ===")
print(pd.DataFrame(ap_out).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/topk.csv", index=False)
pd.DataFrame(ap_out).to_csv("results/topk_perclass_ap.csv", index=False)
print("\nwrote results/topk.csv")
