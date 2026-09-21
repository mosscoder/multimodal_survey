"""Long-tail loss on the winning architecture (ground-alone top-1 MIL, k=1).
Same dense-grid infra as every top-k experiment; only the loss changes.

  bce_pos_weight -- current baseline: BCEWithLogits + pos_weight=(1-p)/p, clamp 20
  focal          -- Focal Loss (gamma=2) on top of the same pos_weight: down-weights
                    easy (well-classified) examples so gradient isn't dominated by
                    the ~98% easy negatives for rare classes like Euphorbia.
  asl            -- Asymmetric Loss (Ridnik et al. 2021), purpose-built for
                    multi-label long-tail: separate focusing gamma for positives
                    (gamma_pos=1) vs negatives (gamma_neg=4), plus a probability
                    margin (clip=0.05) that zeroes loss from very-easy negatives.

3 loss types x 3 seeds = 9 heads/fold, shared batch stream, 5-fold leg CV.
"""
import json
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import f1_score, average_precision_score

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
LOSSES = ["bce_pos_weight", "focal", "asl"]
SEEDS = [0, 1, 2]
P_PATCHES = 3072

GRID = np.load("feats/train.ground_image.general.grid1024x768.npy", mmap_mode="r")   # (6480,3072,768) f16

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


class GroundOnlyHead(nn.Module):
    def __init__(self, d_g=768, n_out=8):
        super().__init__()
        self.ground = nn.Linear(d_g, n_out, bias=True)

    def forward(self, G):
        return self.ground(G).topk(1, dim=1).values.mean(1)


class FocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma=2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=self.pos_weight)
        pt = p * y + (1 - p) * (1 - y)
        return (ce * (1 - pt).clamp(min=1e-6).pow(self.gamma)).mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        p_pos, p_neg = p, 1 - p
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)
        loss = y * torch.log(p_pos.clamp(min=self.eps)) + (1 - y) * torch.log(p_neg.clamp(min=self.eps))
        pt = p_pos * y + p_neg * (1 - y)
        gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        loss = loss * (1 - pt).clamp(min=0).pow(gamma)
        return -loss.mean()


def fit_predict_mil(rows_tr, rows_va, lr=1e-2, wd=1e-2, epochs=60, patience=10, batch=64):
    mu = torch.tensor(PMEAN[rows_tr].mean(0)).cuda()
    sd = torch.tensor(PMEAN[rows_tr].std(0) + 1e-6).cuda()
    Yt = torch.tensor(Y[rows_tr], dtype=torch.float32).cuda()
    Yva = Y[rows_va]
    p = Y[rows_tr].mean(0)
    pw = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20).cuda()

    loss_fns = {
        "bce_pos_weight": nn.BCEWithLogitsLoss(pos_weight=pw),
        "focal": FocalLoss(pos_weight=pw, gamma=2.0),
        "asl": AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05),
    }

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
                out.append(torch.sigmoid(m(Gva[j:j+bs].float())).cpu().numpy())
        return np.concatenate(out)

    heads, opts, scheds, best = {}, {}, {}, {}
    for h in [(lt, s) for lt in LOSSES for s in SEEDS]:
        torch.manual_seed(h[1]); heads[h] = GroundOnlyHead().cuda()
        opts[h] = torch.optim.AdamW(heads[h].parameters(), lr=lr, weight_decay=wd)
        scheds[h] = torch.optim.lr_scheduler.CosineAnnealingLR(opts[h], epochs)
        best[h] = dict(f1=-1.0, state=None, bad=0)

    for ep in range(epochs):
        order = np.random.permutation(len(rows_tr))
        for j in range(0, len(order), batch):
            sel = rows_tr[order[j:j+batch]]
            g = torch.from_numpy(np.asarray(GRID[sel], dtype=np.float32)).cuda()
            Gb = (g - mu) / sd
            yb = Yt[order[j:j+batch]]
            for h, m in heads.items():
                if best[h]["bad"] >= patience:
                    continue
                m.train(); opts[h].zero_grad()
                loss_fns[h[0]](m(Gb), yb).backward(); opts[h].step()
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
oof = {(lt, s): np.zeros(Y_pool.shape, np.float32) for lt in LOSSES for s in SEEDS}
abs_idx = np.where(pool_mask)[0]
for f in range(K):
    va = fp == f
    tr = ~va
    print(f"\n== fold {f}  tr {int(tr.sum())}  va {int(va.sum())} ==", flush=True)
    preds = fit_predict_mil(abs_idx[tr], abs_idx[va])
    for h, Pv in preds.items():
        oof[h][va] = Pv

np.savez("results/longtail_loss_oof.npz", **{f"{lt}_s{s}": oof[(lt, s)] for lt in LOSSES for s in SEEDS})

rows, apr = [], []
for lt in LOSSES:
    seed_P = [oof[(lt, s)] for s in SEEDS]
    for label, P in list(zip([f"{lt} s{s}" for s in SEEDS], seed_P)) + [(f"{lt} ENS", np.mean(seed_P, 0))]:
        per_f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
        per_ap = average_precision_score(Y_pool, P, average=None)
        ff = [f1_score(Y_pool[fp == j], P[fp == j] > 0.5, average="macro", zero_division=0) for j in range(K)]
        rows.append(dict(config=label, macro_F1=per_f1.mean(), macro_AP=per_ap.mean(),
                         perfold_F1=np.mean(ff), perfold_sd=np.std(ff)))
        if label.endswith("ENS"):
            apr.append(dict(config=label, **{c.split()[1][:6]: v for c, v in zip(CLASSES, per_ap)}))
        print(f"{label:20s} macroF1 {per_f1.mean():.3f}  macroAP {per_ap.mean():.3f}  "
              f"perfold {np.mean(ff):.3f}+/-{np.std(ff):.3f}", flush=True)

df = pd.DataFrame(rows)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP (seed ensemble) ===")
print(pd.DataFrame(apr).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/longtail_loss.csv", index=False)
pd.DataFrame(apr).to_csv("results/longtail_loss_perclass_ap.csv", index=False)
print("\nwrote results/longtail_loss.csv")
