"""Follow-up ablation — Fβ threshold objective (recall operating point).

Per-species thresholds are currently chosen to maximize F1. Fβ (β>1) weights recall more, so the
optimal cutoff drops -> higher recall, lower precision. Pure post-hoc on the deployed head's any-patch
scores (no re-extraction/retraining). β=1 reproduces the deployed F1 thresholds (control). Shows the
recall/precision trade per species so an operating point can be picked.
"""
from __future__ import annotations
import os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import common as C, data as D, train_mil as T

BETAS = [1.0, 1.5, 2.0, 3.0]
ck = torch.load(f"{D.ROOT}/checkpoints/mil.pt")
head = C.MILHead(n_classes=ck.get("n_out", len(ck["classes"])), hidden=ck["hidden"], r=ck["r"])
head.load_state_dict(ck["state"]); head.eval()
fids, Y = D.load_robot_gt(); vi, ti = D.iterative_stratification(Y)
scores = T.frame_scores(head, T.extract_robot_patches(None, fids, None))[:, :D.N]   # (533,9) any-patch


def prf(pred, y):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def fbeta(pred, y, b):
    p, r, _ = prf(pred, y); b2 = b * b; den = b2 * p + r
    return (1 + b2) * p * r / den if den > 0 else 0.0


def fit(beta):
    return np.array([max(D.THR_GRID, key=lambda t: fbeta(scores[vi, c] >= t, Y[vi, c], beta)) for c in range(D.N)])


thr_by_beta = {b: fit(b) for b in BETAS}
print("=== Fβ threshold sweep (per-species τ maximizes val Fβ), TEST over the 7 well-supported ===")
print(f"{'beta':>5}{'F1':>9}{'precision':>11}{'recall':>9}")
for b in BETAS:
    pr = [prf(scores[ti, c] >= thr_by_beta[b][c], Y[ti, c]) for c in range(D.N)]
    mP, mR, mF = (np.mean([pr[c][k] for c in D.SWEEP_IDX]) for k in (0, 1, 2))
    print(f"{b:>5}{mF:>9.3f}{mP:>11.3f}{mR:>9.3f}{'   (= F1 baseline)' if b == 1 else ''}")

for b in (1.0, 1.5, 2.0):
    T.report(scores[ti], Y[ti], thr_by_beta[b], f"beta={b} — TEST")
