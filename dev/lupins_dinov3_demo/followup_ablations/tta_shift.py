"""Follow-up ablation — TRANSLATION (sub-patch shift) TEST-TIME AUGMENTATION.

Hypothesis: inference uses one fixed 16-px patch lattice, so a feature straddling a patch boundary
is split across cells and diluted. Re-tiling with a HALF-PATCH (8 px) offset puts the second pass's
patch CENTERS on the first pass's CORNERS, giving boundary-straddling features a clean read. Union
the passes' patches and take the per-species MAX (the any-patch rule generalizes directly).

Design: deployed head (mil.pt), NO retraining. Offsets {(0,0),(8,0),(0,8),(8,8)} (8 = half of the
16-px patch). Configs: single {(0,0)} (control), 2-pass {(0,0),(8,8)}, 4-pass {all four}. Shifting
moves the reflect-pad split so the tile lattice slides by (ox,oy) with the canvas size unchanged.
Thresholds RE-DERIVED on val (the score distribution shifts), applied to test; selection = sel7.
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import common as C, data as D, geometry as G, train_mil as T

OFFSETS = [(0, 0), (8, 0), (0, 8), (8, 8)]
CONFIGS = {"single": [(0, 0)], "2-pass": [(0, 0), (8, 8)], "4-pass": [(0, 0), (8, 0), (0, 8), (8, 8)]}
CKPT = f"{D.ROOT}/checkpoints/mil.pt"


def tiles_offset(pil, ox, oy):
    """24 native 224 tiles with the lattice shifted by (ox,oy) px (canvas stays 1344x896)."""
    pil = pil.convert("RGB")
    arr = np.asarray(pil if pil.size == (G.NW, G.NH) else pil.resize((G.NW, G.NH)), np.uint8)
    padded = np.pad(arr, ((G.PAD_T - oy, G.PAD_B + oy), (G.PAD_L - ox, G.PAD_R + ox), (0, 0)), mode="reflect")
    out = []
    for ty in range(G.NTY):
        for tx in range(G.NTX):
            t = padded[ty * G.TILE:(ty + 1) * G.TILE, tx * G.TILE:(tx + 1) * G.TILE]
            out.append((torch.from_numpy(t.astype(np.float32) / 255.0).permute(2, 0, 1) - C.MEAN[0]) / C.STD[0])
    return torch.stack(out, 0)


def extract_offset(model, fids, device, off):
    if off == (0, 0):
        return T.extract_robot_patches(model, fids, device)              # canonical cache
    ox, oy = off
    sig = f"ttashift|{ox}_{oy}|{G.NW}x{G.NH}|{C.MODEL_ID}|{hashlib.md5(','.join(fids).encode()).hexdigest()[:10]}"

    def build():
        feats = []
        for i, fid in enumerate(fids):
            f, _, _ = C.patch_features(model, tiles_offset(Image.open(os.path.join(C.CAPTURES, fid + ".jpg")), ox, oy), device)
            feats.append(f.reshape(1, -1, f.shape[-1]).half().cpu())
            if i % 80 == 0:
                print(f"  shift{off} {i}/{len(fids)}", end="\r", flush=True)
        print()
        return {"X": torch.cat(feats, 0)}

    return D.cached(f"{D.CACHE}/robot_patches_tta_{ox}_{oy}.pt", sig, build)["X"]


def macro(scores, Y, thr=None):
    all9, f1s, ths = D.macro_f1(scores, Y, thr)
    return all9, float(np.mean([f1s[c] for c in D.SWEEP_IDX])), f1s, ths


def main():
    ck = torch.load(CKPT)
    head = C.MILHead(n_classes=ck.get("n_out", len(ck["classes"])), hidden=ck["hidden"], r=ck["r"])
    head.load_state_dict(ck["state"]); head.eval()
    fids, Y = D.load_robot_gt(); vi, ti = D.iterative_stratification(Y)
    device = C.get_device(); model = C.load_backbone(device)
    smax = {off: T.frame_scores(head, extract_offset(model, fids, device, off))[:, :D.N] for off in OFFSETS}
    del model

    cands, thrs = [], {}
    print("\n=== shift-TTA (val sel7/all9 | test sel7/all9) ===")
    for name, offs in CONFIGS.items():
        fused = np.maximum.reduce([smax[o] for o in offs])
        va9, vsel, _, thr = macro(fused[vi], Y[vi])
        ta9, tsel, tf1, _ = macro(fused[ti], Y[ti], thr)
        tmeas = T._measurable(tf1, Y[ti].sum(0))
        cands.append({"config": name, "passes": len(offs), "val_sel7": round(vsel, 4), "val_all9": round(va9, 4),
                      "test_sel7": round(tsel, 4), "test_all9": round(ta9, 4), "test_meas": round(tmeas, 4)})
        thrs[name] = thr
        print(f"  {name:8} ({len(offs)}p)  val {vsel:.4f}/{va9:.4f}   test {tsel:.4f}/{ta9:.4f}")

    win = max(range(len(cands)), key=lambda i: cands[i]["val_sel7"])
    for name, offs in CONFIGS.items():
        T.report(np.maximum.reduce([smax[o] for o in offs])[ti], Y[ti], thrs[name], f"{name} ({len(offs)}-pass) — TEST")
    res = {"experiment": "tta_shift", "offsets": OFFSETS, "grid": cands, "winner": cands[win]}
    json.dump(res, open(f"{HERE}/tta_shift_results.json", "w"), indent=2)
    print("\nRESULT_JSON: " + json.dumps(res))


if __name__ == "__main__":
    main()
