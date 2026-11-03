"""Item 2: epoch selection by best MEAN macro-AP across folds (no per-fold
early stopping). Full fixed 60-epoch cosine schedule, every fold, every seed.
Native-resolution ground grid (item 1) + freshly recomputed drone CLS.

For each branch (ground, drone):
  - 5-fold CV, 3 seeds, full 60 epochs, per-epoch val predictions recorded.
  - E* = argmax of the mean-over-seeds-then-mean-over-folds val macro-AP curve.
  - Per-fold metrics reported AT E*; pooled OOF at E* feeds fusion.
  - Final model: trained on the FULL 6089-pool, 3 seeds, full 60 epochs, with
    a checkpoint saved at every epoch (cheap -- a single linear layer) so the
    epoch-E* checkpoint can be selected after CV determines E*, with no need
    to retrain. That checkpoint predicts the test split.

Item 3 (reproducibility): all 3 seed-heads in a fold still share one batch
stream (one memmap read per batch, consumed by every head) -- option (b) from
the spec, chosen to avoid tripling the ground branch's memmap streaming cost.
The shared stream is now seeded per fold (np.random.default_rng(fold) for CV,
a fixed constant for the full-pool final run), so a rerun reproduces the same
batch order. Because the order is shared, the 3 "seeds" differ only in weight
initialization (torch.manual_seed(seed), already set per head), not in data
order -- the manuscript must describe them as initialization seeds, not fully
independent resamples.
"""
import json, os, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score, average_precision_score

from assemble import pool, Y_pool
from timing_utils import log_stage, reset_vram

# item 7: set to skip the (hours-long) 5-fold CV loop and go straight to final-model
# training + test prediction, reusing the already-committed E* -- for getting a clean,
# isolated wall-time measurement of just those two stages without redoing the CV.
TIMING_ONLY = os.environ.get("ITEM7_TIMING_FINAL_ONLY") == "1"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)   # warn (not crash) on any op lacking a deterministic kernel
FINAL_RNG_SEED = 1000   # full-pool final run has no fold index; distinct from fold ids 0-4

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
SEEDS = [0, 1, 2]
K = 5
EPOCHS = 60
LR, WD, BATCH = 1e-2, 1e-2, 64
folds = pool["fold"].values
rows_pool = pool.index.to_numpy()
test_ds_frame_id = np.array(load_dataset("mpg-ranch/multimodal-survey", split="test")["frame_id"])

GRID = np.load("feats/train.ground_image.general.grid1280x720.npy", mmap_mode="r")
P_PATCHES = GRID.shape[1]
assert P_PATCHES == 3600, P_PATCHES
A_drone = np.load("feats/train.drone_image.satellite.npy").astype(np.float32)[:, :1024][rows_pool]

print("per-frame patch means (one memmap pass, pool rows only) ...", flush=True)
reset_vram()
t0 = time.time()
PMEAN = np.stack([np.asarray(GRID[r], np.float32).mean(0) for r in rows_pool])
print(f"  done in {time.time()-t0:.0f}s", flush=True)
log_stage("ground_patch_mean_precompute", time.time() - t0)


class GroundHead(nn.Module):
    def __init__(self, d_g=768, n_out=8):
        super().__init__()
        self.ground = nn.Linear(d_g, n_out, bias=True)

    def forward(self, G):
        return self.ground(G).topk(1, dim=1).values.mean(1)


class DroneHead(nn.Module):
    def __init__(self, d_a=1024, n_out=8):
        super().__init__()
        self.drone = nn.Linear(d_a, n_out, bias=True)

    def forward(self, a):
        return self.drone(a)


def pos_weight_for(y):
    p = y.mean(0)
    return torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20)


# ---------------------------------------------------------------- GROUND ---
def ground_fold_run(rows_tr, rows_va, fold):
    """3 seeds, shared batch stream (seeded per fold -- item 3), full EPOCHS,
    no early stop. Returns: per-epoch val-prob array per seed (EPOCHS, n_va, 8)."""
    rng = np.random.default_rng(fold)
    mu = torch.tensor(PMEAN[rows_tr].mean(0)).cuda()
    sd = torch.tensor(PMEAN[rows_tr].std(0) + 1e-6).cuda()
    Yt = torch.tensor(Y_pool[rows_tr], dtype=torch.float32).cuda()
    pw = pos_weight_for(Y_pool[rows_tr]).cuda()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    abs_tr, abs_va = rows_pool[rows_tr], rows_pool[rows_va]
    n_va = len(abs_va)
    Gva = torch.empty((n_va, P_PATCHES, 768), dtype=torch.float16, device="cuda")
    for j in range(0, n_va, 64):
        idx = abs_va[j:j+64]
        g = torch.from_numpy(np.asarray(GRID[idx], dtype=np.float32)).cuda()
        Gva[j:j+len(idx)] = ((g - mu) / sd).half()

    def predict(m, bs=64):
        m.eval(); out = []
        with torch.no_grad():
            for j in range(0, n_va, bs):
                out.append(torch.sigmoid(m(Gva[j:j+bs].float())).cpu().numpy())
        return np.concatenate(out)

    heads, opts, scheds = {}, {}, {}
    for s in SEEDS:
        torch.manual_seed(s); heads[s] = GroundHead().cuda()
        opts[s] = torch.optim.AdamW(heads[s].parameters(), lr=LR, weight_decay=WD)
        scheds[s] = torch.optim.lr_scheduler.CosineAnnealingLR(opts[s], EPOCHS)

    epoch_preds = {s: np.empty((EPOCHS, n_va, 8), dtype=np.float32) for s in SEEDS}
    for ep in range(EPOCHS):
        order = rng.permutation(len(rows_tr))
        for j in range(0, len(order), BATCH):
            sel = abs_tr[order[j:j+BATCH]]
            g = torch.from_numpy(np.asarray(GRID[sel], dtype=np.float32)).cuda()
            Gb = (g - mu) / sd
            yb = Yt[order[j:j+BATCH]]
            for s, m in heads.items():
                m.train(); opts[s].zero_grad()
                loss_fn(m(Gb), yb).backward(); opts[s].step()
        for s, m in heads.items():
            scheds[s].step()
            epoch_preds[s][ep] = predict(m)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"    ground ep {ep:2d}  (t={time.time()-t0:.0f}s)", flush=True)
    del Gva; torch.cuda.empty_cache()
    return epoch_preds


def ground_final_run():
    """Train on the FULL pool, 3 seeds, full EPOCHS, checkpoint every epoch."""
    rng = np.random.default_rng(FINAL_RNG_SEED)
    mu = torch.tensor(PMEAN.mean(0)).cuda()
    sd = torch.tensor(PMEAN.std(0) + 1e-6).cuda()
    Yt = torch.tensor(Y_pool, dtype=torch.float32).cuda()
    pw = pos_weight_for(Y_pool).cuda()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    heads, opts, scheds, ckpts = {}, {}, {}, {s: [] for s in SEEDS}
    for s in SEEDS:
        torch.manual_seed(s); heads[s] = GroundHead().cuda()
        opts[s] = torch.optim.AdamW(heads[s].parameters(), lr=LR, weight_decay=WD)
        scheds[s] = torch.optim.lr_scheduler.CosineAnnealingLR(opts[s], EPOCHS)

    n = len(rows_pool)
    for ep in range(EPOCHS):
        order = rng.permutation(n)
        for j in range(0, n, BATCH):
            sel = rows_pool[order[j:j+BATCH]]
            g = torch.from_numpy(np.asarray(GRID[sel], dtype=np.float32)).cuda()
            Gb = (g - mu) / sd
            yb = Yt[order[j:j+BATCH]]
            for s, m in heads.items():
                m.train(); opts[s].zero_grad()
                loss_fn(m(Gb), yb).backward(); opts[s].step()
        for s, m in heads.items():
            scheds[s].step()
            ckpts[s].append({k: v.clone().cpu() for k, v in m.state_dict().items()})
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"    ground FINAL ep {ep:2d}", flush=True)
    return ckpts, (mu, sd)


def ground_predict_test(state_dict, mu, sd):
    GRID_TEST = np.load("feats/test.ground_image.general.grid1280x720.npy", mmap_mode="r")
    m = GroundHead().cuda(); m.load_state_dict(state_dict); m.eval()
    out = []
    with torch.no_grad():
        for j in range(0, len(GRID_TEST), 64):
            g = torch.from_numpy(np.asarray(GRID_TEST[j:j+64], dtype=np.float32)).cuda()
            Gb = (g - mu) / sd
            out.append(torch.sigmoid(m(Gb)).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------- DRONE ---
def drone_fold_run(rows_tr, rows_va, fold):
    rng = np.random.default_rng(fold)
    X = A_drone
    mu, sd = X[rows_tr].mean(0), X[rows_tr].std(0) + 1e-6
    Xtr = torch.tensor((X[rows_tr] - mu) / sd, dtype=torch.float32)
    Xva = torch.tensor((X[rows_va] - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(Y_pool[rows_tr], dtype=torch.float32)
    pw = pos_weight_for(Y_pool[rows_tr])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    n_va = len(rows_va)

    heads, opts, scheds = {}, {}, {}
    for s in SEEDS:
        torch.manual_seed(s); heads[s] = DroneHead()
        opts[s] = torch.optim.AdamW(heads[s].parameters(), lr=LR, weight_decay=WD)
        scheds[s] = torch.optim.lr_scheduler.CosineAnnealingLR(opts[s], EPOCHS)

    epoch_preds = {s: np.empty((EPOCHS, n_va, 8), dtype=np.float32) for s in SEEDS}
    for ep in range(EPOCHS):
        order = rng.permutation(len(rows_tr))
        for j in range(0, len(order), BATCH):
            idx = order[j:j+BATCH]
            for s, m in heads.items():
                m.train(); opts[s].zero_grad()
                loss_fn(m(Xtr[idx]), Yt[idx]).backward(); opts[s].step()
        for s, m in heads.items():
            scheds[s].step()
            m.eval()
            with torch.no_grad():
                epoch_preds[s][ep] = torch.sigmoid(m(Xva)).numpy()
    return epoch_preds


def drone_final_run():
    rng = np.random.default_rng(FINAL_RNG_SEED)
    X = A_drone
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(Y_pool, dtype=torch.float32)
    pw = pos_weight_for(Y_pool)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(rows_pool)

    heads, opts, scheds, ckpts = {}, {}, {}, {s: [] for s in SEEDS}
    for s in SEEDS:
        torch.manual_seed(s); heads[s] = DroneHead()
        opts[s] = torch.optim.AdamW(heads[s].parameters(), lr=LR, weight_decay=WD)
        scheds[s] = torch.optim.lr_scheduler.CosineAnnealingLR(opts[s], EPOCHS)

    for ep in range(EPOCHS):
        order = rng.permutation(n)
        for j in range(0, n, BATCH):
            idx = order[j:j+BATCH]
            for s, m in heads.items():
                m.train(); opts[s].zero_grad()
                loss_fn(m(Xt[idx]), Yt[idx]).backward(); opts[s].step()
        for s, m in heads.items():
            scheds[s].step()
            ckpts[s].append({k: v.clone() for k, v in m.state_dict().items()})
    return ckpts, (mu, sd)


def drone_predict_test(state_dict, mu, sd):
    X_test = np.load("feats/test.drone_image.satellite.npy").astype(np.float32)[:, :1024]
    Xs = torch.tensor((X_test - mu) / sd, dtype=torch.float32)
    m = DroneHead(); m.load_state_dict(state_dict); m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(Xs)).numpy()


# ---------------------------------------------------------------- DRIVER ---
def run_branch(name, fold_run_fn, final_run_fn, predict_test_fn):
    print(f"\n===== branch: {name} =====", flush=True)
    if not TIMING_ONLY:
        # 1) CV: collect per-epoch val predictions for every fold/seed
        reset_vram(); t_cv = time.time()
        per_fold_epoch_preds = {}   # fold -> {seed: (EPOCHS, n_va, 8)}
        per_fold_va_idx = {}
        for f in range(K):
            va = np.where(folds == f)[0]
            tr = np.where(folds != f)[0]
            print(f"  fold {f}  tr {len(tr)}  va {len(va)}", flush=True)
            per_fold_epoch_preds[f] = fold_run_fn(tr, va, f)
            per_fold_va_idx[f] = va
        log_stage(f"train_cv_{name}", time.time() - t_cv, note=f"{K} folds x {len(SEEDS)} seeds x {EPOCHS} epochs")

        # 2) E* = argmax of mean-over-seeds-then-mean-over-folds val macro-AP curve
        curve = np.zeros(EPOCHS)
        for f in range(K):
            va = per_fold_va_idx[f]
            fold_curve = np.zeros(EPOCHS)
            for ep in range(EPOCHS):
                seed_ap = []
                for s in SEEDS:
                    p = per_fold_epoch_preds[f][s][ep]
                    ap = average_precision_score(Y_pool[va], p, average="macro")
                    seed_ap.append(ap)
                fold_curve[ep] = np.mean(seed_ap)
            curve += fold_curve / K
        E_star = int(np.argmax(curve))
        print(f"  {name}: E* = {E_star}  (mean macro-AP {curve[E_star]:.4f})", flush=True)
        pd.DataFrame({"epoch": np.arange(EPOCHS), "mean_macro_AP": curve}).to_csv(
            f"results/item2_{name}_epoch_curve.csv", index=False)

        # 3) pooled OOF at E*, per-fold metrics at E*
        oof = {s: np.zeros(Y_pool.shape, dtype=np.float32) for s in SEEDS}
        per_fold_rows = []
        for f in range(K):
            va = per_fold_va_idx[f]
            for s in SEEDS:
                oof[s][va] = per_fold_epoch_preds[f][s][E_star]
            fold_P = np.mean([per_fold_epoch_preds[f][s][E_star] for s in SEEDS], axis=0)
            ap = average_precision_score(Y_pool[va], fold_P, average=None)
            f1 = f1_score(Y_pool[va], fold_P > 0.5, average=None, zero_division=0)
            per_fold_rows.append(dict(fold=f, macro_AP=ap.mean(), macro_F1=f1.mean(),
                                      **{f"AP_{c.split()[0]}": v for c, v in zip(CLASSES, ap)}))
        pd.DataFrame(per_fold_rows).to_csv(f"results/item2_{name}_perfold_at_Estar.csv", index=False)
        oof_ens = np.mean([oof[s] for s in SEEDS], axis=0)
        ap_pooled = average_precision_score(Y_pool, oof_ens, average=None)
        f1_pooled = f1_score(Y_pool, oof_ens > 0.5, average=None, zero_division=0)
        print(f"  {name} pooled OOF @ E*={E_star}: macroAP {ap_pooled.mean():.3f}  macroF1 {f1_pooled.mean():.3f}", flush=True)
    else:
        est_summary = pd.read_csv("results/item2_Estar_summary.csv", index_col=0)
        E_star = int(est_summary.loc[name, "E_star"])
        print(f"  {name}: [ITEM7_TIMING_FINAL_ONLY] using committed E* = {E_star}, skipping CV", flush=True)

    # 4) final model: full pool, checkpoint every epoch, select E*, predict test
    print(f"  {name}: training final model on full pool ...", flush=True)
    reset_vram(); t_final = time.time()
    ckpts, (mu, sd) = final_run_fn()
    log_stage(f"train_final_{name}", time.time() - t_final, note=f"full pool, {EPOCHS} epochs")

    reset_vram(); t_pred = time.time()
    test_preds = [predict_test_fn(ckpts[s][E_star], mu, sd) for s in SEEDS]
    test_pred_ens = np.mean(test_preds, axis=0)
    log_stage(f"test_predict_{name}", time.time() - t_pred, note=f"{len(test_pred_ens)} frames")

    if TIMING_ONLY:
        return None   # measurement-only pass: don't touch the already-committed result files

    np.savez(f"results/preds/item2_{name}_oof.npz", **{f"s{s}": oof[s] for s in SEEDS}, ens=oof_ens,
            frame_id=pool["frame_id"].to_numpy())
    np.savez(f"results/preds/item2_{name}_test.npz", **{f"s{s}": test_preds[i] for i, s in enumerate(SEEDS)},
            ens=test_pred_ens, frame_id=test_ds_frame_id)
    return dict(E_star=E_star, cv_macro_AP=ap_pooled.mean(), cv_macro_F1=f1_pooled.mean())


results = {}
results["drone"] = run_branch("drone", drone_fold_run, drone_final_run, drone_predict_test)
results["ground"] = run_branch("ground", ground_fold_run, ground_final_run, ground_predict_test)

if not TIMING_ONLY:
    pd.DataFrame(results).T.to_csv("results/item2_Estar_summary.csv")
    print("\n=== E* summary ===")
    print(pd.DataFrame(results).T.to_string(float_format=lambda x: f"{x:.3f}"))
print("\ndone.")
