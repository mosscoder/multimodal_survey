"""5-fold leg-fold CV for the species multilabel probe.

Heads (user-supplied): "linear" = 8 one-vs-all logistic regressions,
"mlp" = 1 hidden layer (1792->512->8, GELU, dropout 0.2).

OOF predictions over folds_leg5.json, scored pooled + per-fold.
"""
import sys
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
)

from assemble import pool, Y_pool, K, features, standardize

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]


def make_head(kind, d_in, n_out=8, hidden=512, p_drop=0.2):
    if kind == "linear":
        return nn.Linear(d_in, n_out)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p_drop),
                         nn.Linear(hidden, n_out))


def fit_predict(X_tr, Y_tr, X_va, Y_va, kind, lr, wd=1e-2, epochs=60, patience=10, seed=0):
    torch.manual_seed(seed)
    head = make_head(kind, X_tr.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    p = Y_tr.mean(0)
    pos_weight = torch.tensor((1 - p) / np.maximum(p, 1e-3), dtype=torch.float32).clamp(max=20)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xt, Yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(Y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32)
    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        head.train()
        for idx in torch.randperm(len(Xt)).split(256):
            opt.zero_grad(); loss_fn(head(Xt[idx]), Yt[idx]).backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            P = torch.sigmoid(head(Xv)).numpy()
        f1 = f1_score(Y_va, P > 0.5, average="macro", zero_division=0)
        if f1 > best:
            best, state, bad = f1, {k: v.clone() for k, v in head.state_dict().items()}, 0
        elif (bad := bad + 1) >= patience:
            break
    head.load_state_dict(state)
    with torch.no_grad():
        return torch.sigmoid(head(Xv)).numpy(), ep


def cv_oof(modality, kind, lr, seed=0):
    X = features("train", modality, rows=pool.index)
    fold = pool["fold"].values
    oof = np.zeros_like(Y_pool, dtype=np.float32)
    stopped = []
    for k in range(K):
        va = fold == k
        Xtr, Xva = standardize(X[~va], X[va])
        P, ep = fit_predict(Xtr, Y_pool[~va], Xva, Y_pool[va], kind, lr, seed=seed)
        oof[va] = P
        stopped.append(ep)
    return oof, stopped


def report(Y, P):
    valid = [j for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    Yv, Pv = Y[:, valid], P[:, valid]
    pred = (Pv > 0.5).astype(int)
    macro_ap = average_precision_score(Yv, Pv, average="macro")
    micro_ap = average_precision_score(Yv, Pv, average="micro")
    macro_auc = roc_auc_score(Yv, Pv, average="macro")
    macro_f1 = f1_score(Yv, pred, average="macro", zero_division=0)
    micro_f1 = f1_score(Yv, pred, average="micro", zero_division=0)
    per_ap = average_precision_score(Yv, Pv, average=None)
    return dict(macro_ap=macro_ap, micro_ap=micro_ap, macro_auc=macro_auc,
                macro_f1=macro_f1, micro_f1=micro_f1), valid, per_ap


def per_fold_macro_f1(Y, P, fold):
    vals = []
    for k in range(K):
        m = fold == k
        valid = [j for j in range(Y.shape[1]) if 0 < Y[m, j].sum() < m.sum()]
        vals.append(f1_score(Y[m][:, valid], (P[m][:, valid] > 0.5).astype(int),
                             average="macro", zero_division=0))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    fold = pool["fold"].values
    configs = [
        ("concat", "linear", 1e-2),
        ("ground", "linear", 1e-2),
        ("drone",  "linear", 1e-2),
        ("concat", "mlp",    1e-3),
        ("ground", "mlp",    1e-3),
        ("drone",  "mlp",    1e-3),
    ]
    print(f"{'modality':8s} {'head':7s} {'lr':>6s} | {'macroAP':>8s} {'microAP':>8s} "
          f"{'macroAUC':>9s} {'macroF1':>8s} {'microF1':>8s} | {'perfold macroF1':>18s} | stop")
    rows = []
    for mod, kind, lr in configs:
        oof, stopped = cv_oof(mod, kind, lr)
        m, valid, per_ap = report(Y_pool, oof)
        mean_k, sd_k = per_fold_macro_f1(Y_pool, oof, fold)
        print(f"{mod:8s} {kind:7s} {lr:6.0e} | {m['macro_ap']:8.3f} {m['micro_ap']:8.3f} "
              f"{m['macro_auc']:9.3f} {m['macro_f1']:8.3f} {m['micro_f1']:8.3f} | "
              f"{mean_k:.3f} +/- {sd_k:.3f}      | {stopped}")
        rows.append((mod, kind, lr, m, valid, per_ap))
    print("\nper-class average precision (pooled OOF):")
    print(f"{'':16s}" + "".join(f"{c.split()[1][:6]:>8s}" for c in CLASSES))
    for mod, kind, lr, m, valid, per_ap in rows:
        cells = {j: v for j, v in zip(valid, per_ap)}
        print(f"{mod+'/'+kind:16s}" + "".join(
            f"{cells.get(j, float('nan')):8.3f}" for j in range(len(CLASSES))))


if __name__ == "__main__":
    main()
