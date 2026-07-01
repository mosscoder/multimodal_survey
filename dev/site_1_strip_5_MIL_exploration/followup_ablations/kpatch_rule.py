"""Follow-up ablation — relax the ANY-PATCH (k=1) rule to a k-of-N rule.

Present iff >= k patches score >= tau  <=>  the k-th-largest patch score >= tau. So sweeping k just
thresholds the k-th order statistic of the patch scores — pure post-hoc on the deployed head, no
re-extraction/retraining. Requiring k>=2 rejects single stray-patch FPs, which lets tau drop and
recall recover. We jointly pick (k, tau) per species to maximize val F1, then score test.

Reports: (1) fixed-k for all species (the global precision/recall trend), and (2) per-species
k optimized vs the k=1 baseline (the current deployed any-patch rule, which it reproduces).
"""
from __future__ import annotations
import os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import common as C, data as D, train_mil as T

KS = [1, 2, 3, 5, 8]
KMAX = max(KS)
ck = torch.load(f"{D.ROOT}/checkpoints/mil.pt")
head = C.MILHead(n_classes=ck.get("n_out", len(ck["classes"])), hidden=ck["hidden"], r=ck["r"])
head.load_state_dict(ck["state"]); head.eval()
fids, Y = D.load_robot_gt(); vi, ti = D.iterative_stratification(Y)
P = T.extract_robot_patches(None, fids, None)

# top-KMAX patch sigmoids per frame per target species -> tops[frame, k-1, species]
tops = np.zeros((len(fids), KMAX, D.N), np.float32)
with torch.no_grad():
    for i in range(0, len(fids), 16):
        s = torch.sigmoid(head(P[i:i + 16].float())[1])[:, :, :D.N]          # (b,4704,9)
        tops[i:i + 16] = torch.topk(s, KMAX, dim=1).values.numpy()           # (b,KMAX,9)


def best_tau(scores, y):
    bf, bt = -1.0, 0.5
    for t in D.THR_GRID:
        f = D._f1(scores >= t, y)
        if f > bf:
            bf, bt = f, float(t)
    return bf, bt


def macros(scores, thr):
    f1 = [D._f1(scores[ti, c] >= thr[c], Y[ti, c]) for c in range(D.N)]
    P_ = [ (lambda p, y: int((p & (y == 1)).sum()) / max(int(p.sum()), 1))(scores[ti, c] >= thr[c], Y[ti, c]) for c in range(D.N)]
    R_ = [ (lambda p, y: int((p & (y == 1)).sum()) / max(int((y == 1).sum()), 1))(scores[ti, c] >= thr[c], Y[ti, c]) for c in range(D.N)]
    sel7 = np.mean([f1[c] for c in D.SWEEP_IDX]); all9 = np.mean(f1)
    mp = np.mean([P_[c] for c in D.SWEEP_IDX]); mr = np.mean([R_[c] for c in D.SWEEP_IDX])
    return sel7, all9, mp, mr


print("=== fixed k-of-N (all species same k), TEST ===")
print(f"{'k':>3}{'test sel7':>11}{'test all9':>11}{'macroP(7)':>11}{'macroR(7)':>11}")
for k in KS:
    thr = np.array([best_tau(tops[vi, k - 1, c], Y[vi, c])[1] for c in range(D.N)])
    sel7, all9, mp, mr = macros(tops[:, k - 1, :], thr)
    tag = "  (any-patch)" if k == 1 else ""
    print(f"{k:>3}{sel7:>11.4f}{all9:>11.4f}{mp:>11.3f}{mr:>11.3f}{tag}")

# per-species k optimized on val F1
base_thr = np.zeros(D.N); kopt_thr = np.zeros(D.N); kstar = np.ones(D.N, int)
base_sc = tops[:, 0, :].copy(); kopt_sc = np.zeros((len(fids), D.N), np.float32)
for c in range(D.N):
    base_thr[c] = best_tau(tops[vi, 0, c], Y[vi, c])[1]
    bf = -1.0
    for k in KS:
        f, t = best_tau(tops[vi, k - 1, c], Y[vi, c])
        if f > bf:
            bf, kstar[c], kopt_thr[c] = f, k, t
    kopt_sc[:, c] = tops[:, kstar[c] - 1, c]

T.report(base_sc[ti], Y[ti], base_thr, "BASELINE k=1 (any-patch) — TEST")
T.report(kopt_sc[ti], Y[ti], kopt_thr, "k-of-N optimized per species — TEST")
print("\nchosen k per species:", {D.NAMES[c]: int(kstar[c]) for c in range(D.N)})
bs = macros(base_sc, base_thr); ks = macros(kopt_sc, kopt_thr)
print(f"baseline  sel7 {bs[0]:.4f} all9 {bs[1]:.4f}  macroP {bs[2]:.3f} macroR {bs[3]:.3f}")
print(f"k-opt     sel7 {ks[0]:.4f} all9 {ks[1]:.4f}  macroP {ks[2]:.3f} macroR {ks[3]:.3f}")
