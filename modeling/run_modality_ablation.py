"""Modality ablation on the best config (TopKHead, k=1):

  1. ground   -- ground-image MIL (top-1 patch pooling) ALONE, no aerial term
  2. drone    -- drone/satellite CLS token ALONE, plain linear head
  3. concat   -- ground MIL (top-1) + drone CLS offset  [[the winning config]]

Same data, folds, seeds, optimizer as the top-k sweep, so all 3 numbers are
directly comparable (trained together in one script, one shared batch stream
per fold across the 3 modes x 3 seeds = 9 heads).
"""
import json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score, average_precision_score

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
MODES = ["ground", "drone", "concat"]
SEEDS = [0, 1, 2]
P_PATCHES = 3072

GRID = np.load("feats/train.ground_image.general.grid1024x768.npy", mmap_mode="r")   # (6480,3072,768) f16
A = np.load("feats/train.drone_image.satellite.npy")[:, :1024].astype(np.float32)     # aerial CLS, 224px

ds = load_dataset("mpg-ranch/multimodal-survey", split="train")
meta = ds.remove_columns(["ground_image", "drone_image"]).to_pandas()
Y = np.zeros((len(meta), 8), dtype=np.float32)
for i, ids in enumerate(meta["species"]):
    Y[i, list(ids)] = 1
leg_no = meta["leg"].str[3:].astype(int)
grp = meta["site"] + "/" + meta["strip"] + "/" + meta["leg"]
fmap = json.load(open("folds_leg5.json"))["fold_of_group"]
fold_abs = np.array([fmap.get(g, -1) if ln % 2 == 1 else -1 for g, ln in zip(grp, leg_no)], dtype=int)
K = 5
pool_mask = fold_abs >= 0
Y_pool = Y[pool_mask]
fp = fold_abs[pool_mask]
print(f"pool {int(pool_mask.sum())}  per fold {[int((fold_abs==k).sum()) for k in range(K)]}", flush=True)

print("per-frame patch means (one memmap pass) ...", flush=True)
PMEAN = np.empty((len(GRID), 768), np.float32)
for i in range(0, len(GRID), 256):
    PMEAN[i:i+256] = np.asarray(GRID[i:i+256], dtype=np.float32).mean(1)


class AblationHead(nn.Module):
    """mode='ground': top-1 ground patch only. 'drone': aerial CLS only.
    'concat': both, as in the winning TopKHead(k=1)."""
    def __init__(self, mode, d_g=768, d_a=1024, n_out=8):
        super().__init__()
        self.mode = mode
        if mode in ("ground", "concat"):
            self.ground = nn.Linear(d_g, n_out, bias=(mode == "ground"))
        if mode in ("drone", "concat"):
            self.aerial = nn.Linear(d_a, n_out)

    def forward(self, G, a):
        if self.mode == "ground":
            return self.ground(G).topk(1, dim=1).values.mean(1)
        if self.mode == "drone":
            return self.aerial(a)
        return self.ground(G).topk(1, dim=1).values.mean(1) + self.aerial(a)


def fit_predict_mil(rows_tr, rows_va, lr=1e-2, wd=1e-2, epochs=60, patience=10, batch=64):
    mu = torch.tensor(PMEAN[rows_tr].mean(0)).cuda()
    sd = torch.tensor(PMEAN[rows_tr].std(0) + 1e-6).cuda()
    a_mu, a_sd = A[rows_tr].mean(0), A[rows_tr].std(0) + 1e-6
    a_tr = torch.tensor((A[rows_tr] - a_mu) / a_sd, dtype=torch.float32).cuda()
    a_va = torch.tensor((A[rows_va] - a_mu) / a_sd, dtype=torch.float32).cuda()
    Yt = torch.tensor(Y[rows_tr], dtype=torch.float32).cuda()
    Yva = Y[rows_va]
    p = Y[rows_tr].mean(0)
    pw = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20).cuda()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    n_va = len(rows_va)
    Gva = torch.empty((n_va, P_PATCHES, 768), dtype=torch.float16, device="cuda")
    for j in range(0, n_va, 64):
        idx = rows_va[j:j+64]
        g = torch.from_numpy(np.asarray(GRID[idx], dtype=np.float32)).cuda()
        Gva[j:j+len(idx)] = ((g - mu) / sd).half()

    def predict(m, bs=64):
        m.eval(); out = []
        with torch.no_grad():
            for j in range(0, n_va, bs):
                out.append(torch.sigmoid(m(Gva[j:j+bs].float(), a_va[j:j+bs])).cpu().numpy())
        return np.concatenate(out)

    heads, opts, scheds, best = {}, {}, {}, {}
    for h in [(mode, s) for mode in MODES for s in SEEDS]:
        torch.manual_seed(h[1]); heads[h] = AblationHead(h[0]).cuda()
        opts[h] = torch.optim.AdamW(heads[h].parameters(), lr=lr, weight_decay=wd)
        scheds[h] = torch.optim.lr_scheduler.CosineAnnealingLR(opts[h], epochs)
        best[h] = dict(f1=-1.0, state=None, bad=0)

    for ep in range(epochs):
        order = np.random.permutation(len(rows_tr))
        for j in range(0, len(order), batch):
            sel = rows_tr[order[j:j+batch]]
            g = torch.from_numpy(np.asarray(GRID[sel], dtype=np.float32)).cuda()
            Gb = (g - mu) / sd
            ab = a_tr[order[j:j+batch]]; yb = Yt[order[j:j+batch]]
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
        print(f"    ep {ep:2d}  live {live}/{len(heads)}", flush=True)
        if live == 0:
            break

    out = {}
    for h, m in heads.items():
        m.load_state_dict(best[h]["state"]); m.eval()
        out[h] = predict(m)
    del Gva; torch.cuda.empty_cache()
    return out


# ---- CV ----------------------------------------------------------------
oof = {(mode, s): np.zeros(Y_pool.shape, np.float32) for mode in MODES for s in SEEDS}
abs_idx = np.where(pool_mask)[0]
for f in range(K):
    va = fp == f
    tr = ~va
    print(f"\n== fold {f}  tr {int(tr.sum())}  va {int(va.sum())} ==", flush=True)
    preds = fit_predict_mil(abs_idx[tr], abs_idx[va])
    for h, Pv in preds.items():
        oof[h][va] = Pv

np.savez("results/modality_ablation_oof.npz", **{f"{m}_s{s}": oof[(m, s)] for m in MODES for s in SEEDS})

rows, apr = [], []
for mode in MODES:
    seed_P = [oof[(mode, s)] for s in SEEDS]
    for label, P in list(zip([f"{mode} s{s}" for s in SEEDS], seed_P)) + [(f"{mode} ENS", np.mean(seed_P, 0))]:
        per_f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
        per_ap = average_precision_score(Y_pool, P, average=None)
        ff = [f1_score(Y_pool[fp == j], P[fp == j] > 0.5, average="macro", zero_division=0) for j in range(K)]
        rows.append(dict(config=label, macro_F1=per_f1.mean(), macro_AP=per_ap.mean(),
                         perfold_F1=np.mean(ff), perfold_sd=np.std(ff)))
        if label.endswith("ENS"):
            apr.append(dict(config=label, **{c.split()[1][:6]: v for c, v in zip(CLASSES, per_ap)}))
        print(f"{label:12s} macroF1 {per_f1.mean():.3f}  macroAP {per_ap.mean():.3f}  "
              f"perfold {np.mean(ff):.3f}+/-{np.std(ff):.3f}", flush=True)

df = pd.DataFrame(rows)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP (seed ensemble) ===")
print(pd.DataFrame(apr).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/modality_ablation.csv", index=False)
pd.DataFrame(apr).to_csv("results/modality_ablation_perclass_ap.csv", index=False)
print("\nwrote results/modality_ablation.csv")
