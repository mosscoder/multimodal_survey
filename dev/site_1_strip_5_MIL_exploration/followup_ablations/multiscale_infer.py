"""Follow-up ablation A — MULTI-SCALE INFERENCE TILING.

Hypothesis (see RESEARCH_LOG): the deployed model tiles robot frames at native resolution only
(16 px / patch), yet training also showed the head a whole-plant "square" scale. Adding coarser
inference passes — downsample the frame so each 16 px patch spans a bigger chunk of plant —
should match that learned scale and improve detection, purely at inference.

Design choices:
  - Reuse the deployed head (checkpoints/mil.pt); NO retraining. This isolates the tiling effect,
    and the head already learned a coarse scale from the square training view.
  - Scales s in {1,2,4}. s=1 is the canonical native pass (cached robot_patches.pt). s>1
    downsamples the 1280x720 frame by s, reflect-pads to multiples of 224, tiles, encodes; a
    16 px patch then spans s*16 px of the frame (s=4 ~ 64 px ~ the square-view scale).
  - Fuse with the any-patch rule generalized across scales: per species, the frame score is the
    MAX patch-sigmoid over the union of all patches at all scales in the combo. Spatial grid
    alignment is irrelevant to the frame-level metric, so it is skipped.
  - Multi-scale shifts the score distribution, so thresholds are RE-DERIVED on val (same
    argmax-val-F1 protocol as training) and applied to test. Combo {1} is the control and must
    reproduce the deployed val/test numbers.

Combos: {1}(control), {1,2}, {1,4}, {1,2,4}. Winner by val all-9 (flag if well-supported differs).
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import common as C, data as D, geometry as G, train_mil as T

SCALES = [1, 2, 4]
COMBOS = [[1], [1, 2], [1, 4], [1, 2, 4]]
CKPT = f"{D.ROOT}/checkpoints/mil.pt"


def tiles_scaled(pil, s):
    """Frame -> 1280x720 -> downsample by s -> reflect-pad to multiples of 224 -> (n,3,224,224)."""
    pil = pil.convert("RGB")
    if pil.size != (G.NW, G.NH):
        pil = pil.resize((G.NW, G.NH))
    if s != 1:
        pil = pil.resize((round(G.NW / s), round(G.NH / s)), Image.BICUBIC)
    arr = np.asarray(pil, np.uint8)
    h, w = arr.shape[:2]
    arr = np.pad(arr, ((0, (-h) % G.TILE), (0, (-w) % G.TILE), (0, 0)), mode="reflect")
    hp, wp = arr.shape[:2]
    out = []
    for ty in range(hp // G.TILE):
        for tx in range(wp // G.TILE):
            t = arr[ty * G.TILE:(ty + 1) * G.TILE, tx * G.TILE:(tx + 1) * G.TILE]
            out.append((torch.from_numpy(t.astype(np.float32) / 255.0).permute(2, 0, 1) - C.MEAN[0]) / C.STD[0])
    return torch.stack(out, 0)


def extract_scale(model, fids, device, s):
    """Per-frame patch tokens at scale s -> (nframe, n_s*196, D). s=1 uses the canonical cache."""
    if s == 1:
        return T.extract_robot_patches(model, fids, device)
    sig = f"robotpatch_s{s}|{G.NW}x{G.NH}|{G.TILE}|{C.MODEL_ID}|{hashlib.md5(','.join(fids).encode()).hexdigest()[:10]}"

    def build():
        feats = []
        for i, fid in enumerate(fids):
            t = tiles_scaled(Image.open(os.path.join(C.CAPTURES, fid + ".jpg")), s)
            f, _, _ = C.patch_features(model, t, device)
            feats.append(f.reshape(1, -1, f.shape[-1]).half().cpu())
            if i % 80 == 0:
                print(f"  s{s} patches {i}/{len(fids)}", end="\r", flush=True)
        print()
        return {"X": torch.cat(feats, 0)}

    return D.cached(f"{D.CACHE}/robot_patches_s{s}.pt", sig, build)["X"]


def per_scale_max(head, P):
    """P (nframe, N, D) -> (nframe, 9) per-species MAX patch sigmoid (any-patch at this scale)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(P), T.SCORE_BATCH):
            _, plog = head(P[i:i + T.SCORE_BATCH].float())
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
    device = C.get_device(); model = C.load_backbone(device)
    smax = {s: per_scale_max(head, extract_scale(model, fids, device, s)) for s in SCALES}
    del model

    cands, thrs = [], []
    print("\n=== multi-scale tiling (val all9/meas | test all9/meas) ===")
    for combo in COMBOS:
        fused = np.maximum.reduce([smax[s] for s in combo])
        valm, valmeas, thr = macro(fused[vi], Y[vi])
        testm, testmeas, _ = macro(fused[ti], Y[ti], thr)
        cands.append({"combo": combo, "val_all9": round(valm, 4), "val_meas": round(valmeas, 4),
                      "test_all9": round(testm, 4), "test_meas": round(testmeas, 4)})
        thrs.append(thr)
        print(f"  scales {str(combo):10} val {valm:.4f}/{valmeas:.4f}   test {testm:.4f}/{testmeas:.4f}")

    win = max(range(len(cands)), key=lambda i: cands[i]["val_all9"])
    wmeas = max(range(len(cands)), key=lambda i: cands[i]["val_meas"])
    T.report(np.maximum.reduce([smax[s] for s in COMBOS[0]])[ti], Y[ti], thrs[0],
             f"CONTROL scales {COMBOS[0]} — TEST")
    if win != 0:
        T.report(np.maximum.reduce([smax[s] for s in COMBOS[win]])[ti], Y[ti], thrs[win],
                 f"WINNER scales {COMBOS[win]} — TEST")

    res = {"experiment": "multiscale_infer", "scales": SCALES, "grid": cands,
           "winner": cands[win], "meas_winner": cands[wmeas], "disagree": win != wmeas}
    json.dump(res, open(f"{HERE}/multiscale_results.json", "w"), indent=2)
    print("\nRESULT_JSON: " + json.dumps(res))


if __name__ == "__main__":
    main()
