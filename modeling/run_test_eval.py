"""Final held-out test evaluation of the best config (3-way late fusion +
per-class tuned thresholds): dense-grid ground top-1 MIL + global-pool 768px
ground + drone-alone CLS.

No CV fold model was ever checkpointed, so each component gets a proper final
training run here: train on 4 of the 5 leg-folds, hold out the 5th purely for
early stopping (same method as every CV run, just not scored on that fold).
3 seeds each. Fusion weights and thresholds are refit on the FULL train pool's
already-computed OOF predictions (no test leakage) and applied as-is to test.
"""
import json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
SEEDS = [0, 1, 2]
K = 5
P_PATCHES = 3072
VAL_FOLD = 4                                # held out purely for early-stopping the final model

folds = pool["fold"].values
rows_pool = pool.index.to_numpy()
tr_mask = folds != VAL_FOLD
va_mask = folds == VAL_FOLD

GRID = np.load("feats/train.ground_image.general.grid1024x768.npy", mmap_mode="r")
GRID_TEST = np.load("feats/test.ground_image.general.grid1024x768.npy", mmap_mode="r")

print("per-frame patch means (train pool, for standardization) ...", flush=True)
tr_abs = rows_pool[tr_mask]
PMEAN_TR = np.stack([np.asarray(GRID[r], np.float32).mean(0) for r in tr_abs])
mu_g = torch.tensor(PMEAN_TR.mean(0)).cuda()
sd_g = torch.tensor(PMEAN_TR.std(0) + 1e-6).cuda()


class GroundOnlyHead(nn.Module):
    def __init__(self, d_g=768, n_out=8):
        super().__init__()
        self.ground = nn.Linear(d_g, n_out, bias=True)

    def forward(self, G):
        return self.ground(G).topk(1, dim=1).values.mean(1)


def train_ground_grid(seed):
    va_abs = rows_pool[va_mask]
    Yt = torch.tensor(Y_pool[tr_mask], dtype=torch.float32).cuda()
    Yva = Y_pool[va_mask]
    p = Y_pool[tr_mask].mean(0)
    pw = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20).cuda()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    n_va = len(va_abs)
    Gva = torch.empty((n_va, P_PATCHES, 768), dtype=torch.float16, device="cuda")
    for j in range(0, n_va, 64):
        idx = va_abs[j:j+64]
        g = torch.from_numpy(np.asarray(GRID[idx], dtype=np.float32)).cuda()
        Gva[j:j+len(idx)] = ((g - mu_g) / sd_g).half()

    def predict_on(Gcache, bs=64):
        m.eval(); out = []
        with torch.no_grad():
            for j in range(0, Gcache.shape[0], bs):
                out.append(torch.sigmoid(m(Gcache[j:j+bs].float())).cpu().numpy())
        return np.concatenate(out)

    torch.manual_seed(seed)
    m = GroundOnlyHead().cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
    best, state, bad = -1.0, None, 0
    for ep in range(60):
        m.train()
        order = np.random.permutation(len(tr_abs))
        for j in range(0, len(order), 64):
            sel = tr_abs[order[j:j+64]]
            g = torch.from_numpy(np.asarray(GRID[sel], dtype=np.float32)).cuda()
            Gb = (g - mu_g) / sd_g
            opt.zero_grad(); loss_fn(m(Gb), Yt[order[j:j+64]]).backward(); opt.step()
        sched.step()
        P = predict_on(Gva)
        f1 = f1_score(Yva, P > 0.5, average="macro", zero_division=0)
        if f1 > best:
            best, state, bad = f1, {n: v.clone() for n, v in m.state_dict().items()}, 0
        else:
            bad += 1
        print(f"    ground_grid seed{seed} ep{ep:2d} valF1 {f1:.3f} best {best:.3f} bad {bad}", flush=True)
        if bad >= 10:
            break
    m.load_state_dict(state); m.eval()
    del Gva; torch.cuda.empty_cache()

    # apply to TEST
    n_test = GRID_TEST.shape[0]
    P_test = []
    with torch.no_grad():
        for j in range(0, n_test, 64):
            g = torch.from_numpy(np.asarray(GRID_TEST[j:j+64], dtype=np.float32)).cuda()
            Gb = (g - mu_g) / sd_g
            P_test.append(torch.sigmoid(m(Gb)).cpu().numpy())
    return np.concatenate(P_test)


def fit_small(Xtr, Ytr, Xva, Yva, seed, lr=1e-2, wd=1e-2, epochs=60, patience=10, batch=64):
    torch.manual_seed(seed)
    head = nn.Linear(Xtr.shape[1], 8)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    p = Ytr.mean(0)
    pw = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, Yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        head.train()
        order = np.random.permutation(len(Xt))
        for j in range(0, len(order), batch):
            idx = order[j:j+batch]
            opt.zero_grad(); loss_fn(head(Xt[idx]), Yt[idx]).backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            P = torch.sigmoid(head(Xv)).numpy()
        f1 = f1_score(Yva, P > 0.5, average="macro", zero_division=0)
        if f1 > best:
            best, state, bad = f1, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= patience:
            break
    head.load_state_dict(state); head.eval()
    return head


def train_small(Xtr_full, seed, X_test):
    X_all = Xtr_full[rows_pool]
    Xtr, Xva = X_all[tr_mask], X_all[va_mask]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_s, Xva_s = (Xtr - mu) / sd, (Xva - mu) / sd
    head = fit_small(Xtr_s, Y_pool[tr_mask], Xva_s, Y_pool[va_mask], seed)
    with torch.no_grad():
        return torch.sigmoid(head(torch.tensor((X_test - mu) / sd, dtype=torch.float32))).numpy()


# ---- train final models, 3 seeds each ----
print("\ntraining final ground_grid (dense-grid top-1 MIL) ...")
ground_grid_test = np.mean([train_ground_grid(s) for s in SEEDS], axis=0)

print("training final ground_global (768px CLS+patch-mean) ...")
X768_train = np.load("feats/train.ground_image.general.r768.npy").astype(np.float32)
X768_test = np.load("feats/test.ground_image.general.r768.npy").astype(np.float32)
ground_global_test = np.mean([train_small(X768_train, s, X768_test) for s in SEEDS], axis=0)

print("training final drone (224px CLS) ...")
Adrone_train = np.load("feats/train.drone_image.satellite.npy").astype(np.float32)[:, :1024]
Adrone_test = np.load("feats/test.drone_image.satellite.npy").astype(np.float32)[:, :1024]
drone_test = np.mean([train_small(Adrone_train, s, Adrone_test) for s in SEEDS], axis=0)

np.savez("results/test_component_preds.npz", ground_grid=ground_grid_test,
        ground_global=ground_global_test, drone=drone_test)

# ---- fusion weights + thresholds refit on the FULL train pool's existing OOF (no test leakage) ----
d = np.load("results/armhead_ablation_oof.npz")
ground_grid_oof = np.mean([d[f"ground_linear_s{s}"] for s in SEEDS], axis=0)
drone_oof = np.mean([d[f"drone_linear_s{s}"] for s in SEEDS], axis=0)
ground_global_oof = np.load("results/res_ensemble_oof.npz")["ground_global"]

Ws = np.linspace(0, 1, 11)
w_final = np.zeros((8, 3))
for j in range(8):
    y = Y_pool[:, j]
    best_ap, best_w = -1, (1, 0, 0)
    for wg in Ws:
        for wl in Ws:
            if wg + wl > 1:
                continue
            wd_ = 1 - wg - wl
            p = wg * ground_grid_oof[:, j] + wl * ground_global_oof[:, j] + wd_ * drone_oof[:, j]
            apw = average_precision_score(y, p)
            if apw > best_ap:
                best_ap, best_w = apw, (wg, wl, wd_)
    w_final[j] = best_w

P_train_fused = np.stack([w_final[j, 0] * ground_grid_oof[:, j] + w_final[j, 1] * ground_global_oof[:, j]
                          + w_final[j, 2] * drone_oof[:, j] for j in range(8)], axis=1)

Ts = np.linspace(0.05, 0.95, 37)
t_final = np.zeros(8)
for j in range(8):
    y, p = Y_pool[:, j], P_train_fused[:, j]
    best_f1, best_t = -1, 0.5
    for t in Ts:
        f1t = f1_score(y, p > t, zero_division=0)
        if f1t > best_f1:
            best_f1, best_t = f1t, t
    t_final[j] = best_t

print("\nfinal per-class fusion weights (w_grid, w_global, w_drone) and thresholds:")
print(pd.DataFrame(np.hstack([w_final, t_final[:, None]]),
                   columns=["w_grid", "w_global", "w_drone", "threshold"],
                   index=[c.split()[0] for c in CLASSES]).to_string(float_format=lambda x: f"{x:.2f}"))

# ---- apply to TEST ----
P_test_fused = np.stack([w_final[j, 0] * ground_grid_test[:, j] + w_final[j, 1] * ground_global_test[:, j]
                         + w_final[j, 2] * drone_test[:, j] for j in range(8)], axis=1)

test_ds = load_dataset("mpg-ranch/multimodal-survey", split="test")
Y_test = np.zeros((len(test_ds), 8), dtype=np.int8)
for i, ids in enumerate(test_ds["species"]):
    Y_test[i, list(ids)] = 1

ap_test = average_precision_score(Y_test, P_test_fused, average=None)
f1_test_05 = f1_score(Y_test, P_test_fused > 0.5, average=None, zero_division=0)
pred_tuned = (P_test_fused > t_final[None, :]).astype(int)
f1_test_tuned = f1_score(Y_test, pred_tuned, average=None, zero_division=0)

print(f"\n=== HELD-OUT TEST RESULTS (n={len(test_ds)}) ===")
print(f"{'species':24s} {'AP':>7s} {'F1@0.5':>8s} {'F1@tuned':>9s}")
for j, c in enumerate(CLASSES):
    print(f"{c:24s} {ap_test[j]:7.3f} {f1_test_05[j]:8.3f} {f1_test_tuned[j]:9.3f}")
print(f"{'MACRO':24s} {ap_test.mean():7.3f} {f1_test_05.mean():8.3f} {f1_test_tuned.mean():9.3f}")

print(f"\n=== reference: CV results ===")
print("CV macro AP (3-way + tuned threshold): 0.824")
print("CV macro F1 @0.5: 0.767   @tuned: 0.771")

pd.DataFrame({"species": CLASSES, "test_AP": ap_test, "test_F1_at_0.5": f1_test_05,
             "test_F1_at_tuned": f1_test_tuned}).to_csv("results/test_eval.csv", index=False)
print("\nwrote results/test_eval.csv")
