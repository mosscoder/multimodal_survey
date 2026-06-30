"""Follow-up ablation B — SUPER-PATCH (token pooling).

Hypothesis (see RESEARCH_LOG): each scored unit is a 16 px patch; pooling neighborhoods of patch
tokens into "super-patches" lets each scored unit summarize a bigger chunk of plant, which may
align better with whole-plant cues and reduce speckle.

Design choices:
  - Reuse the deployed head (mil.pt) + the native robot tiling (cached robot_patches.pt). Pure
    inference ablation, NO retraining, NO backbone reload (features are cached).
  - KNOWN LIMITATION (deliberate, documented): the head was trained on individual tokens, so
    averaged "super-patch" tokens are off-distribution for it. This tests whether pooling helps
    *despite* that mismatch — a cheap upper-bound-ish probe before committing to retraining on
    pooled tokens.
  - Each native tile is 14x14 tokens; adaptive-average-pool each tile's token grid to g x g, so a
    super-patch spans ~224/g px. g in {14(control),7,4,2,1} -> 16,32,56,112,224 px units
    (g=1 = whole-tile mean, one token per 224 tile).
  - Any-patch max over super-patches per species; thresholds RE-DERIVED on val (scoring changed)
    and applied to test. g=14 is the control and must reproduce the deployed numbers.

Winner by val all-9 (flag if well-supported differs).
"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import common as C, data as D, geometry as G, train_mil as T

GRIDS = [14, 7, 4, 2, 1]                 # tokens per tile side after pooling (14 = control)
PG = G.TILE // C.PATCH                    # 14 native tokens per tile side
CKPT = f"{D.ROOT}/checkpoints/mil.pt"


def pooled_max(head, robotP, g):
    """robotP (nf, 24*196, D) -> adaptive-avg-pool each tile to gxg -> (nf, 9) any-patch max."""
    nf, _, Dd = robotP.shape
    out = []
    with torch.no_grad():
        for i in range(nf):
            x = robotP[i].float().reshape(-1, PG, PG, Dd).permute(0, 3, 1, 2)   # (24, D, 14, 14)
            xp = x if g == PG else F.adaptive_avg_pool2d(x, g)                  # (24, D, g, g)
            tok = xp.permute(0, 2, 3, 1).reshape(1, -1, Dd)                     # (1, 24*g*g, D)
            _, plog = head(tok)
            out.append(torch.sigmoid(plog).amax(1).numpy())
    return np.concatenate(out, 0)


def macro(scores, Y, thr=None):
    m, f1s, ths = D.macro_f1(scores, Y, thr)
    return m, T._measurable(f1s, Y.sum(0)), ths


def main():
    ck = torch.load(CKPT)
    head = C.MILHead(n_classes=len(ck["classes"]), hidden=ck["hidden"], r=ck["r"])
    head.load_state_dict(ck["state"]); head.eval()
    fids, Y = D.load_robot_gt(); vi, ti = D.iterative_stratification(Y)
    robotP = T.extract_robot_patches(None, fids, None)        # cache hit -> no backbone

    smax = {g: pooled_max(head, robotP, g) for g in GRIDS}
    cands, thrs = [], []
    print("\n=== super-patch pooling (val all9/meas | test all9/meas) ===")
    for g in GRIDS:
        valm, valmeas, thr = macro(smax[g][vi], Y[vi])
        testm, testmeas, _ = macro(smax[g][ti], Y[ti], thr)
        cands.append({"grid": g, "px": round(G.TILE / g), "val_all9": round(valm, 4),
                      "val_meas": round(valmeas, 4), "test_all9": round(testm, 4),
                      "test_meas": round(testmeas, 4)})
        thrs.append(thr)
        print(f"  g={g:<3} (~{round(G.TILE/g):3d}px) val {valm:.4f}/{valmeas:.4f}   test {testm:.4f}/{testmeas:.4f}")

    win = max(range(len(cands)), key=lambda i: cands[i]["val_all9"])
    wmeas = max(range(len(cands)), key=lambda i: cands[i]["val_meas"])
    T.report(smax[GRIDS[0]][ti], Y[ti], thrs[0], f"CONTROL g={GRIDS[0]} — TEST")
    if win != 0:
        T.report(smax[GRIDS[win]][ti], Y[ti], thrs[win], f"WINNER g={GRIDS[win]} — TEST")

    res = {"experiment": "superpatch_pool", "grids": GRIDS, "grid": cands,
           "winner": cands[win], "meas_winner": cands[wmeas], "disagree": win != wmeas}
    json.dump(res, open(f"{HERE}/superpatch_results.json", "w"), indent=2)
    print("\nRESULT_JSON: " + json.dumps(res))


if __name__ == "__main__":
    main()
