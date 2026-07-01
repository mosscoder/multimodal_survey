"""rev-4 FINAL sweep: choose the DEPLOYMENT tiling on robot frames.

Compares the three inference tilings (core/tiling: base / macro / macro+patch) at each method's OWN
per-species thresholds RE-DERIVED on val (F1-argmax — the deployed rule; denser tilings raise the
any-patch max, so base thresholds would over-fire). Selection = val sel7 (the 7 well-sampled species).
The winner is applied to the held-out test, then its tiling + re-derived thresholds are FROZEN into
checkpoints/mil.pt so infer_mil and the any-patch metric both use it.

Per-frame any-patch scores for all 533 frames are cached to cache/tiling_scores.npz (keyed to the
checkpoint), so re-deriving thresholds afterward is instant. Run AFTER `sweep_mil.py --stage final`.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import common as C
import data as D
import tiling as TL
import train_mil as T

CKPT = f"{D.ROOT}/checkpoints/mil.pt"
METRICS = f"{D.ROOT}/checkpoints/mil_metrics.json"
SCORES = f"{D.CACHE}/tiling_scores.npz"
STATE = os.path.join(HERE, "sweep_state.json")


def _sel7(f1):
    return float(np.mean([f1[c] for c in D.SWEEP_IDX]))


def _ckpt_sig(ck):
    return (f"{ck.get('species')}|{ck.get('hidden')}|{ck.get('r')}|{ck.get('n_per')}|"
            f"{ck.get('lr')}|{ck.get('best_epoch')}|{ck.get('n_out')}|{C.MODEL_ID}")


def frame_scores_all(ck, head, fids):
    """(n_frames, D.N) target-column any-patch scores per tiling, cached to npz keyed to the ckpt."""
    sig = _ckpt_sig(ck)
    if os.path.exists(SCORES):
        z = np.load(SCORES, allow_pickle=True)
        if str(z["sig"]) == sig and list(z["fids"]) == list(fids):
            print(f"  cached: tiling_scores.npz", flush=True)
            return {t: z[t] for t in TL.TILINGS}
    device = C.get_device(); model = C.load_backbone(device)
    n_out = ck.get("n_out", D.N)
    S = {t: np.zeros((len(fids), D.N), np.float32) for t in TL.TILINGS}
    for i, fid in enumerate(fids):
        pil = Image.open(os.path.join(C.CAPTURES, fid + ".jpg"))
        grids = TL.all_grids(model, head, pil, n_out, device)
        for t in TL.TILINGS:
            S[t][i] = grids[t][..., :D.N].max((0, 1))         # frame any-patch over the 9 target cols
        if i % 25 == 0:
            print(f"  tiling scores {i}/{len(fids)}", flush=True)
    np.savez(SCORES, sig=np.array(sig), fids=np.array(fids), **{t: S[t] for t in TL.TILINGS})
    print(f"  saved {SCORES}", flush=True)
    return S


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiling", choices=TL.TILINGS, default=None,
                    help="force this tiling as the deployed winner (default: pick by best val sel7)")
    args = ap.parse_args()
    ck = torch.load(CKPT)
    classes = ck["classes"]; n_out = ck.get("n_out", len(classes))
    head = C.MILHead(n_classes=n_out, hidden=ck["hidden"], r=ck["r"])
    head.load_state_dict(ck["state"]); head.eval()
    fids, Y = D.load_robot_gt(); vi, ti = D.iterative_stratification(Y)
    Yv, Yt = Y[vi], Y[ti]

    S = frame_scores_all(ck, head, fids)

    print(f"\n=== tiling sweep — thresholds RE-DERIVED on val (F1-argmax), applied to held-out test ===")
    print(f"{'tiling':14}{'val sel7':>10}{'val all9':>10}{'test sel7':>11}{'test all9':>11}")
    res = {}
    for t in TL.TILINGS:
        va9, vf1, ths = D.macro_f1(S[t][vi], Yv, thr=None)            # derive on val
        ta9, tf1, _ = D.macro_f1(S[t][ti], Yt, np.array(ths))         # apply to test
        res[t] = {"tiling": t, "val_sel7": round(_sel7(vf1), 4), "val_all9": round(va9, 4),
                  "test_sel7": round(_sel7(tf1), 4), "test_all9": round(ta9, 4),
                  "thr": [round(float(x), 4) for x in ths]}
        print(f"{t:14}{_sel7(vf1):>10.4f}{va9:>10.4f}{_sel7(tf1):>11.4f}{ta9:>11.4f}", flush=True)

    best_val = max(TL.TILINGS, key=lambda t: res[t]["val_sel7"])
    winner = args.tiling or best_val
    print(f"\nWINNER: {winner}" + (f"  (FORCED; best val sel7 was '{best_val}')" if args.tiling else "  (best val sel7)"))
    thr = np.array(res[winner]["thr"], np.float32)
    T.report(S[winner][ti], Yt, thr, f"{winner} — held-out TEST @ val-rederived thresholds")

    # freeze winner tiling + thresholds into the deployed checkpoint
    ck["tiling"] = winner
    ck["thresholds"] = res[winner]["thr"]
    torch.save(ck, CKPT)
    if os.path.exists(METRICS):
        mj = json.load(open(METRICS))
        mj.update({"tiling": winner, "thresholds": dict(zip(D.NAMES, res[winner]["thr"])),
                   "test_sel7_f1": res[winner]["test_sel7"], "test_macro_f1": res[winner]["test_all9"]})
        json.dump(mj, open(METRICS, "w"), indent=2)
    print(f"\nfroze tiling='{winner}' + re-derived thresholds into {os.path.basename(CKPT)}")

    if os.path.exists(STATE):
        st = json.load(open(STATE)); st.setdefault("stages", {})
        st["stages"]["tiling"] = {"stage": "tiling", "grid": res, "winner": res[winner]}
        json.dump(st, open(STATE, "w"), indent=2)
    print(f"\nRESULT_JSON: {json.dumps({'winner': winner, 'grid': res})}")


if __name__ == "__main__":
    main()
