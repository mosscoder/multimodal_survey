"""Does a cleaner drone signal (CLS + mean-patch, 2048-d) help late fusion more
than CLS-only (1024-d) did? Retrains drone-alone on 'both' pooling (feature
already extracted -- no GPU grid work needed), then redoes the same nested
5-fold late fusion (weighted avg + stacking) against the existing ground-alone
OOF predictions.
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

A_full = np.load("feats/train.drone_image.satellite.npy").astype(np.float32)[rows_pool]  # (6089,2048)
A_cls = A_full[:, :1024]
A_both = A_full                                                                          # cls + patch-mean


def fit_predict(Xtr, Ytr, Xva, Yva, lr=1e-2, wd=1e-2, epochs=60, patience=10, batch=64, seed=0):
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
    ff = [f1_score(Y_pool[folds == k], P[folds == k] > 0.5, average="macro", zero_division=0) for k in range(K)]
    return f1, ap, np.mean(ff), np.std(ff)


def report(tag, P, rows, apr):
    f1, ap, fm, fsd = score(P)
    print(f"{tag:32s} macroF1 {f1.mean():.3f}  macroAP {ap.mean():.3f}  perfold {fm:.3f}+/-{fsd:.3f}")
    rows.append(dict(config=tag, macro_F1=f1.mean(), macro_AP=ap.mean(), perfold_F1=fm, perfold_sd=fsd))
    apr.append(dict(config=tag, **{c.split()[1][:6]: v for c, v in zip(CLASSES, ap)}))


rows, apr = [], []

print("training drone-alone, CLS-only (1024-d) ...")
drone_cls = cv_oof(A_cls, "drone/cls")
print("training drone-alone, CLS+patch-mean (2048-d, 'both') ...")
drone_both = cv_oof(A_both, "drone/both")

d = np.load("results/armhead_ablation_oof.npz")
ground = np.mean([d[f"ground_linear_s{s}"] for s in SEEDS], axis=0)

report("ground alone (ref)", ground, rows, apr)
report("drone alone, CLS (ref)", drone_cls, rows, apr)
report("drone alone, both (this run)", drone_both, rows, apr)


def late_fuse(g, a, tag):
    Ws = np.linspace(0, 1, 21)
    P_wavg = np.zeros(Y_pool.shape, dtype=np.float32)
    w_chosen = np.zeros((K, 8))
    for f in range(K):
        tr, va = folds != f, folds == f
        for j in range(8):
            y = Y_pool[tr, j]
            best_ap, best_w = -1, 0.5
            for w in Ws:
                p = w * g[tr, j] + (1 - w) * a[tr, j]
                apw = average_precision_score(y, p)
                if apw > best_ap:
                    best_ap, best_w = apw, w
            w_chosen[f, j] = best_w
            P_wavg[va, j] = best_w * g[va, j] + (1 - best_w) * a[va, j]
    report(f"weighted avg ({tag})", P_wavg, rows, apr)

    X = np.concatenate([g, a], axis=1)
    P_stack = np.zeros(Y_pool.shape, dtype=np.float32)
    for f in range(K):
        tr, va = folds != f, folds == f
        for j in range(8):
            clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
            clf.fit(X[tr], Y_pool[tr, j])
            P_stack[va, j] = clf.predict_proba(X[va])[:, 1]
    report(f"stacking LR ({tag})", P_stack, rows, apr)
    return w_chosen


w_cls = late_fuse(ground, drone_cls, "drone=CLS")
w_both = late_fuse(ground, drone_both, "drone=both")

print("\nchosen w, drone=both (w=1 -> pure ground, w=0 -> pure drone):")
print(pd.DataFrame(w_both, columns=[c.split()[0] for c in CLASSES],
                   index=[f"fold{k}" for k in range(K)]).to_string(float_format=lambda x: f"{x:.2f}"))

df = pd.DataFrame(rows)
print("\n=== summary ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\n=== per-class AP ===")
print(pd.DataFrame(apr).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
df.to_csv("results/drone_both_fusion.csv", index=False)
pd.DataFrame(apr).to_csv("results/drone_both_fusion_perclass_ap.csv", index=False)
print("\nwrote results/drone_both_fusion.csv")
