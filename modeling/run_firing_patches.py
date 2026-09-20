"""Retrain TopKHead(k=8, seed=0) for fold 0 only (checkpoints were never saved),
then use firing_patches() to build a QA list: for each species, the most confident
predicted-present and predicted-absent frames in fold 0's held-out set, with the
top-3 firing patch boxes (16px, in 1024x768 coords) for a human to eyeball.
"""
import json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
RES_W, RES_H = 1024, 768
GRID_W = RES_W // 16  # 64

GRID = np.load(f"feats/train.ground_image.general.grid{RES_W}x{RES_H}.npy", mmap_mode="r")
A = np.load("feats/train.drone_image.satellite.npy")[:, :1024].astype(np.float32)  # aerial CLS, unchanged

ds = load_dataset("mpg-ranch/multimodal-survey", split="train")
meta = ds.remove_columns(["ground_image", "drone_image"]).to_pandas()
Y = np.zeros((len(meta), 8), dtype=np.float32)
for i, ids in enumerate(meta["species"]):
    Y[i, list(ids)] = 1
leg_no = meta["leg"].str[3:].astype(int)
grp = meta["site"] + "/" + meta["strip"] + "/" + meta["leg"]
fmap = json.load(open("folds_leg5.json"))["fold_of_group"]
fold_abs = np.array([fmap.get(g, -1) if ln % 2 == 1 else -1 for g, ln in zip(grp, leg_no)], dtype=int)

FOLD = 0
va = np.where(fold_abs == FOLD)[0]
tr = np.where((fold_abs >= 0) & (fold_abs != FOLD))[0]
print(f"fold {FOLD}: train {len(tr)}  val {len(va)}", flush=True)


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


def grid_stats(rows):
    m = np.stack([np.asarray(GRID[r], np.float32).mean(0) for r in rows])
    return m.mean(0), m.std(0) + 1e-6


mu_g, sd_g = grid_stats(tr)
mu_gt = torch.tensor(mu_g).cuda(); sd_gt = torch.tensor(sd_g).cuda()
a_mu, a_sd = A[tr].mean(0), A[tr].std(0) + 1e-6
a_tr = torch.tensor((A[tr] - a_mu) / a_sd, dtype=torch.float32).cuda()
a_va = torch.tensor((A[va] - a_mu) / a_sd, dtype=torch.float32).cuda()
Yt = torch.tensor(Y[tr], dtype=torch.float32).cuda()
Yva = Y[va]
p = Y[tr].mean(0)
pw = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20).cuda()
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

torch.manual_seed(0)
head = TopKHead(8).cuda()
opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)


def grid_batch(rows):
    g = torch.tensor(np.asarray(GRID[rows], dtype=np.float32)).cuda()
    return (g - mu_gt) / sd_gt


def predict(m, bs=64):
    m.eval(); out = []
    with torch.no_grad():
        for j in range(0, len(va), bs):
            out.append(torch.sigmoid(m(grid_batch(va[j:j+bs]), a_va[j:j+bs])).cpu().numpy())
    return np.concatenate(out)


best, state, bad = -1.0, None, 0
for ep in range(60):
    head.train()
    order = np.random.permutation(len(tr))
    for j in range(0, len(order), 64):
        sel = tr[order[j:j+64]]
        opt.zero_grad()
        loss_fn(head(grid_batch(sel), a_tr[order[j:j+64]]), Yt[order[j:j+64]]).backward()
        opt.step()
    sched.step()
    f1 = f1_score(Yva, predict(head) > 0.5, average="macro", zero_division=0)
    if f1 > best:
        best, state, bad = f1, {n: v.clone() for n, v in head.state_dict().items()}, 0
    else:
        bad += 1
    print(f"  ep {ep:2d}  f1 {f1:.3f}  best {best:.3f}  bad {bad}", flush=True)
    if bad >= 10:
        break

head.load_state_dict(state); head.eval()
P_va = predict(head)
print(f"\nfold {FOLD} k=8 seed=0 macro F1 {best:.3f}")


@torch.no_grad()
def firing_patches(row, head, mu_gt, sd_gt, top=3):
    """Top-scoring patches per species for one frame, with the fold's standardization
    constants. Ground branch only: the aerial offset is flat across patches and
    cannot move the ranking."""
    g = (torch.tensor(np.asarray(GRID[row], np.float32)).cuda() - mu_gt) / sd_gt
    s = head.ground(g)  # (P, 8)
    v, p = s.topk(top, dim=0)  # per-species top patches
    yx = [[(int(i) // GRID_W * 16, int(i) % GRID_W * 16) for i in p[:, j]]
          for j in range(len(CLASSES))]  # 16 px boxes at RES_W x RES_H
    return v.cpu().numpy(), yx


# ---- QA lists: per species, most-confident present / absent frames in fold 0's val set ----
frame_id = meta["frame_id"].to_numpy()
rows_qa = []
for j, name in enumerate(CLASSES):
    present = np.where(Yva[:, j] == 1)[0]
    absent = np.where(Yva[:, j] == 0)[0]
    top_present = present[np.argsort(-P_va[present, j])[:2]] if len(present) else []
    top_absent = absent[np.argsort(-P_va[absent, j])[:2]] if len(absent) else []
    for bucket, idxs in [("present", top_present), ("absent (check for slip)", top_absent)]:
        for i in idxs:
            row = va[i]
            v, yx = firing_patches(row, head, mu_gt, sd_gt, top=3)
            boxes = "; ".join(f"({y},{x})" for y, x in yx[j])
            rows_qa.append(dict(species=name, bucket=bucket, frame_id=frame_id[row],
                                p_hat=P_va[i, j], patch_logits=", ".join(f"{x:.2f}" for x in v[:, j]),
                                top3_boxes_yx_px=boxes))

qa = pd.DataFrame(rows_qa)
qa.to_csv("results/firing_patches_qa.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 40)
print("\n=== QA list (fold 0, k=8, seed=0) ===")
print(qa.to_string(index=False))
print("\nwrote results/firing_patches_qa.csv")
