"""Multi-resolution ground ensemble: dense-grid top-1 MIL (1024x768, local patch
pooling) vs a structurally different ground-alone model -- global CLS+mean-patch
pooling at 768x768 (feature already extracted). Different pooling mechanisms =
more likely to have decorrelated errors than the k-ensemble (which failed).
"""
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression

from assemble import pool, Y_pool

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
SEEDS = [0, 1, 2]
K = 5
folds = pool["fold"].values
rows_pool = pool.index.to_numpy()

X768 = np.load("feats/train.ground_image.general.r768.npy").astype(np.float32)[rows_pool]  # (N,1536)


def fit_predict(Xtr, Ytr, Xva, Yva, lr=1e-2, wd=1e-2, epochs=60, patience=10, batch=64, seed=0):
    torch.manual_seed(seed)
    head = nn.Linear(Xtr.shape[1], 8)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    p = Ytr.mean(0)
    pw = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(Xtr, dtype=torch.float32); Yt = torch.tensor(Ytr, dtype=torch.float32)
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
    with torch.no_grad():
        return torch.sigmoid(head(Xv)).numpy()


def cv_oof(X, tag):
    seed_oof = []
    for s in SEEDS:
        oof = np.zeros(Y_pool.shape, dtype=np.float32)
        for f in range(K):
            tr, va = folds != f, folds == f
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xtr, Xva = (X[tr] - mu) / sd, (X[va] - mu) / sd
            oof[va] = fit_predict(Xtr, Y_pool[tr], Xva, Y_pool[va], seed=s)
        seed_oof.append(oof)
        f1 = f1_score(Y_pool, oof > 0.5, average=None, zero_division=0)
        ap = average_precision_score(Y_pool, oof, average=None)
        print(f"  {tag} seed{s}  macroF1 {f1.mean():.3f}  macroAP {ap.mean():.3f}", flush=True)
    return np.mean(seed_oof, 0)


def score(P):
    f1 = f1_score(Y_pool, P > 0.5, average=None, zero_division=0)
    ap = average_precision_score(Y_pool, P, average=None)
    ff = [f1_score(Y_pool[folds == j], P[folds == j] > 0.5, average="macro", zero_division=0) for j in range(K)]
    return f1, ap, np.mean(ff), np.std(ff)


rows, apr = [], []
def report(tag, P):
    f1, ap, fm, fsd = score(P)
    print(f"{tag:28s} macroF1 {f1.mean():.3f}  macroAP {ap.mean():.3f}  perfold {fm:.3f}+/-{fsd:.3f}")
    rows.append(dict(config=tag, macro_F1=f1.mean(), macro_AP=ap.mean(), perfold_F1=fm, perfold_sd=fsd))
    apr.append(dict(config=tag, **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap)}))


d = np.load("results/armhead_ablation_oof.npz")
ground_grid = np.mean([d[f"ground_linear_s{s}"] for s in SEEDS], axis=0)   # dense-grid top-1 MIL, 1024x768
report("dense-grid top-1 MIL (ref)", ground_grid)

print("training ground alone, 768px global CLS+mean-patch pooling ...")
ground_global = cv_oof(X768, "ground768")
report("global-pool 768px (this run)", ground_global)


def late_fuse(a, b, tag):
    Ws = np.linspace(0, 1, 21)
    P_wavg = np.zeros(Y_pool.shape, dtype=np.float32)
    w_chosen = np.zeros((K, 8))
    for f in range(K):
        tr, va = folds != f, folds == f
        for j in range(8):
            y = Y_pool[tr, j]
            best_ap, best_w = -1, 0.5
            for w in Ws:
                p = w * a[tr, j] + (1 - w) * b[tr, j]
                apw = average_precision_score(y, p)
                if apw > best_ap:
                    best_ap, best_w = apw, w
            w_chosen[f, j] = best_w
            P_wavg[va, j] = best_w * a[va, j] + (1 - best_w) * b[va, j]
    report(f"weighted avg ({tag})", P_wavg)

    X = np.concatenate([a, b], axis=1)
    P_stack = np.zeros(Y_pool.shape, dtype=np.float32)
    for f in range(K):
        tr, va = folds != f, folds == f
        for j in range(8):
            clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
            clf.fit(X[tr], Y_pool[tr, j])
            P_stack[va, j] = clf.predict_proba(X[va])[:, 1]
    report(f"stacking LR ({tag})", P_stack)
    return w_chosen, P_wavg


w_chosen, P_best = late_fuse(ground_grid, ground_global, "grid+global768")

print("\nchosen w (w=1 -> pure dense-grid top-1, w=0 -> pure global-768):")
print(pd.DataFrame(w_chosen, columns=[c.split()[0] for c in CLASSES],
                   index=[f"fold{k}" for k in range(K)]).to_string(float_format=lambda x: f"{x:.2f}"))

df = pd.DataFrame(rows)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== for context: other project milestones ===")
print("ground+drone weighted-avg late fusion (current overall best): macro AP 0.823")
print("\n=== per-class AP ===")
print(pd.DataFrame(apr).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/res_ensemble.csv", index=False)
pd.DataFrame(apr).to_csv("results/res_ensemble_perclass_ap.csv", index=False)
np.savez("results/res_ensemble_oof.npz", ground_grid=ground_grid, ground_global=ground_global, best_fused=P_best)
print("\nwrote results/res_ensemble.csv")
