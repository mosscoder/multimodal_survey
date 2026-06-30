"""Train the MIL head on free iNaturalist imagery; evaluate presence/absence on robot frames.

Per CLAUDE.md:
  - Train on iNat only. Each crop VIEW of an image is a bag of 196 DINOv3 patch tokens labeled
    with the image's species; the head is trained with log-sum-exp bag pooling (BCE one-vs-all).
  - Eval is the ANY-PATCH rule on robot frames: a species is present iff >=1 patch scores >= its
    threshold. One threshold per species, argmax val-F1 per epoch, best epoch FROZEN.

Sweep SELECTION uses the macro over the 7 non-excluded species (D.SWEEP_IDX); the 2 rare species
(yarrow, tumble mustard) are reported only on the final test. Species sets: a config may train on
just the 9 TARGETS or on extra HF-dataset species as hard-negative / auxiliary one-vs-all classes.
TARGETS are always the FIRST classes, so scoring/metrics slice the first D.N logit columns.

Crop views: ("native", z) resize whole image by z then native 224 center; ("square",) short-side
crop -> 224. A config is an ordered list of views, each view -> its own labeled bag.

Features cached per (species, view, size) over the FULL iNat dataset (train+test); image picks are
NESTED across size. Exposes extract_robot_patches / extract_inat_bags / train_head / val_metrics /
save_run / load_splits.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
import common as C
import data as D
import geometry as G
from PIL import Image

R_LSE = 8.0
SCORE_BATCH = 16

DEF_VIEWS = [("native", 1.0)]
DEF_NPER = 128          # rev-5: default imgs/species (was 256)
DEF_LR = 5e-5            # rev-4 default LR (carried through hidden/crop/size/r; LR is swept LAST)
DEF_HIDDEN = 64
DEF_CLIP = 1.0          # rev-5: grad-norm clip LOCKED ON for every run (gradclip ablation found a
                        # near-free gain at the optimum); pass clip=None only to deliberately disable.
SKY_VIEWS = [("corner", "ul"), ("corner", "ur")]   # Sky (bald eagle) ALWAYS uses upper-corner sky crops
SKY_SEG_ID = "nvidia/segformer-b0-finetuned-ade-512-512"   # ADE20K segmenter (has a 'sky' class)
SKY_FRAC_MIN = 0.10                                          # keep a Sky corner crop iff >= this is sky


# --------------------------------------------------------------------------- #
# Feature extraction (cached)                                                  #
# --------------------------------------------------------------------------- #
def extract_robot_patches(model, fids, device):
    """Each robot frame -> 24 native 224 tiles -> 4704 patch tokens (one scene bag)."""
    sig = f"robotpatch|{G.NW}x{G.NH}|{G.TILE}|{G.NTX}x{G.NTY}|{C.MODEL_ID}|{hashlib.md5(','.join(fids).encode()).hexdigest()[:10]}"

    def build():
        feats = []
        for i, fid in enumerate(fids):
            t = G.tiles(Image.open(os.path.join(C.CAPTURES, fid + ".jpg")))   # (24,3,224,224)
            f, _, _ = C.patch_features(model, t, device)                       # (24,196,D)
            feats.append(f.reshape(1, -1, f.shape[-1]).half().cpu())           # (1,4704,D)
            if i % 40 == 0:
                print(f"  robot patches {i}/{len(fids)}", end="\r", flush=True)
        print()
        return {"X": torch.cat(feats, 0)}

    return D.cached(f"{D.CACHE}/robot_patches.pt", sig, build)["X"]


def _view_tag(view):
    if view[0] == "square":
        return "square"
    if view[0] == "corner":
        return f"corner{str(view[1]).upper()}"          # cornerUL / cornerUR
    return f"native{float(view[1]):.2f}"


def sky_pass(corner):
    """Boolean mask over the eagle (Sky) dataset: True where the `corner` crop is >= SKY_FRAC_MIN
    'sky' by SegFormer / ADE20K. Cached; SegFormer is loaded lazily, only on a cache miss. Gates the
    Sky training crops so the negative learns real sky, not perched-eagle backgrounds."""
    path = f"{D.CACHE}/sky_pass_{corner}.pt"
    sig = f"skypass|{D.SKY[1]}|{corner}|{SKY_FRAC_MIN}|{SKY_SEG_ID}|{D.ds_fingerprint(D.SKY[1])}"

    def build():
        import torch.nn.functional as Fnn
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        dev = C.get_device()
        proc = SegformerImageProcessor.from_pretrained(SKY_SEG_ID)
        seg = SegformerForSemanticSegmentation.from_pretrained(SKY_SEG_ID).eval().to(dev)
        sky_id = next((i for i, l in seg.config.id2label.items() if str(l).strip().lower() == "sky"), 2)
        tr = D.load_full(D.SKY[1]); n = len(tr)
        fracs = np.zeros(n, np.float32)
        for s in range(0, n, 16):
            crops = [C.crop_corner(tr[j]["image"], corner) for j in range(s, min(s + 16, n))]
            inp = proc(images=crops, return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = seg(**inp).logits                                  # (b,150,H/4,W/4)
            up = Fnn.interpolate(logits, size=crops[0].size[::-1], mode="bilinear", align_corners=False)
            pred = up.argmax(1)                                             # (b,224,224)
            for k in range(pred.shape[0]):
                fracs[s + k] = float((pred[k] == sky_id).float().mean())
            if s % 160 == 0:
                print(f"  sky-seg {corner} {s}/{n}", flush=True)
        passm = fracs >= SKY_FRAC_MIN
        print(f"  sky_pass[{corner}]: {int(passm.sum())}/{n} corners >= {SKY_FRAC_MIN:.0%} sky", flush=True)
        return {"pass": torch.from_numpy(passm), "frac": torch.from_numpy(fracs)}

    return D.cached(path, sig, build)["pass"].numpy()


def _month_balance(order, mon, months):
    """Reorder `order` (shuffled row indices) so its prefix is balanced across `months` (equal per
    month, round-robin in shuffle order), restricted to rows whose observed month is in `months`.
    Rows with a missing/out-of-window month are dropped. Used by the phenology stage."""
    want = set(months)
    by_m: dict[int, list[int]] = {}
    for i in order:
        mi = mon[int(i)]
        if mi in want:
            by_m.setdefault(int(mi), []).append(int(i))
    present = sorted(by_m)
    out, j = [], 0
    while any(j < len(by_m[m]) for m in present):
        for m in present:
            if j < len(by_m[m]):
                out.append(by_m[m][j])
        j += 1
    return np.array(out, dtype=np.int64)


def extract_species_view(model, device, count, view, hfdir, months=None):
    """Bags for ONE species, ONE view, up to `count` train images (nested fixed-shuffle prefix,
    seeded per-species). Features only (no labels); cached per (species, view, count, months, data
    fingerprint). `months` (e.g. [6,7,8]) restricts to that observed-month window and draws EQUALLY
    per month (phenology stage); None = all months in shuffle order. For the Sky class the per-corner
    SegFormer mask gates selection (only >= SKY_FRAC_MIN-sky corners), and the horizontal flip is
    skipped (it would swap which corner is sampled)."""
    tag = _view_tag(view)
    seed = D.SEED + D.HFDIR_IDX[hfdir]
    sky = (hfdir == D.SKY[1])
    skt = f"|sky{int(SKY_FRAC_MIN * 100)}" if sky else ""
    mtag = "" if months is None else "|m" + "_".join(str(m) for m in sorted(months))
    msuf = "" if months is None else "_m" + "_".join(str(m) for m in sorted(months))
    fp = D.ds_fingerprint(hfdir)                             # invalidate cache when data re-harvested
    sig = f"specview|{hfdir}|{tag}|{count}|{seed}|full{skt}{mtag}|{fp}|{C.MODEL_ID}"
    fname = f"specview_{hfdir}_{tag}_n{count}{('_sky' + str(int(SKY_FRAC_MIN * 100))) if sky else ''}{msuf}.pt"

    def build():
        tr = D.load_full(hfdir)                              # full iNat (train+test)
        order = np.random.default_rng(seed).permutation(len(tr))
        if sky:                                              # keep only sky-passing corners, in shuffle order
            order = order[sky_pass(view[1])[order]]
        if months is not None:                              # phenology window: equal draw per month
            order = _month_balance(order, tr["month"], months)
        idx = order[:min(count, len(order))]
        if len(idx) == 0:                                   # e.g. a spring-only negative under a summer window
            npatch = (C.TILE_SIZE // C.PATCH) ** 2
            print(f"  {hfdir} {tag}: 0 imgs in window {months} (empty bag)", flush=True)
            return {"X": torch.empty(0, npatch, C.EMBED_DIM, dtype=torch.float16)}
        flip = np.random.default_rng(seed + 1000)
        buf, bags = [], []

        def flush():
            if buf:
                f, _, _ = C.patch_features(model, torch.cat(buf, 0), device)
                bags.append(f.half().cpu()); buf.clear()

        for i in idx:
            pil = tr[int(i)]["image"]
            if not sky and flip.random() < 0.5:              # no flip for Sky (would swap UL<->UR)
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            buf.append(C.make_view(pil, view))
            if len(buf) >= D.EXTRACT_BATCH:
                flush()
        flush()
        print(f"  {hfdir} {tag} ({len(idx)} imgs{', sky-gated' if sky else ''})", flush=True)
        return {"X": torch.cat(bags, 0)}

    return D.cached(f"{D.CACHE}/{fname}", sig, build)["X"]


def extract_inat_bags(model, device, species, n_per=DEF_NPER, views=DEF_VIEWS, n_extract=None, months=None):
    """Bags for a species set: each species -> labeled bags (label = its index in `species`), one
    bag per crop view. Plant species use `views`; the auxiliary Sky class ALWAYS uses the upper-corner
    sky views (SKY_VIEWS), independent of the swept plant crop. `months` (phenology window) applies to
    every class including Sky — a summer deployment wants summer plants AND summer skies."""
    n_extract = n_extract or n_per
    Xs, ys = [], []
    for ci, (_, hfdir) in enumerate(species):
        cviews = SKY_VIEWS if hfdir == D.SKY[1] else views
        for v in cviews:
            X = extract_species_view(model, device, n_extract, v, hfdir, months=months)[:n_per]
            Xs.append(X); ys.append(torch.full((len(X),), ci, dtype=torch.long))
    return torch.cat(Xs, 0), torch.cat(ys, 0)


# --------------------------------------------------------------------------- #
# Any-patch scoring, metrics, training                                         #
# --------------------------------------------------------------------------- #
def frame_scores(head, patches):
    """patches (nf, P, D) -> (nf, n_out) per-class score = MAX patch sigmoid (any-patch rule)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(patches), SCORE_BATCH):
            _, plog = head(patches[i:i + SCORE_BATCH].float())                 # (b,P,n_out)
            out.append(torch.sigmoid(plog).amax(dim=1).numpy())                # max over patches
    return np.concatenate(out, 0)


def _measurable(f1s, support, min_sup=10):
    keep = [f for f, s in zip(f1s, support) if s >= min_sup]
    return float(np.mean(keep)) if keep else 0.0


def val_metrics(head, valP, valY, thr=None):
    """Any-patch metrics on robot-val over the 9 TARGET columns. `sel` (the sweep SELECTION score)
    is the macro over the 7 non-excluded species; `all9` is the full macro (final-test reporting)."""
    all9, f1s, ths = D.macro_f1(frame_scores(head, valP)[:, :D.N], valY, thr)
    support = valY.sum(0)
    return {"sel": float(np.mean([f1s[c] for c in D.SWEEP_IDX])), "all9": all9,
            "meas": _measurable(f1s, support), "f1s": f1s, "thr": ths,
            "support": [int(s) for s in support]}


def train_head(trX, trL, valP, valY, lr=DEF_LR, hidden=DEF_HIDDEN, r=R_LSE, n_classes=None, clip=DEF_CLIP):
    """Train the MIL head; keep the epoch with best val SELECTION macro (7 spp) + its frozen
    thresholds (all 9 thresholds are tuned, so the 2 excluded species can still be reported).
    Grad-norm clipping is ON by default (clip=DEF_CLIP=1.0); pass clip=None to disable."""
    n_classes = n_classes or D.N
    counts = np.bincount(trL.numpy(), minlength=n_classes); ntot = int(counts.sum())
    posw = torch.tensor([(ntot - c) / max(c, 1) for c in counts], dtype=torch.float32)
    torch.manual_seed(D.SEED)
    head = C.MILHead(n_classes=n_classes, hidden=hidden, r=r)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=D.WD)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=posw)
    trY = torch.nn.functional.one_hot(trL, n_classes).float()
    idx = np.arange(len(trX)); rng = np.random.default_rng(D.SEED); best = {"sel": -1.0}; bad = 0
    for ep in range(D.EPOCHS):
        head.train(); rng.shuffle(idx)
        for i in range(0, len(idx), D.BATCH):
            b = idx[i:i + D.BATCH]
            bag_logits, _ = head(trX[b].float())                              # LSE bag pooling
            loss = lossf(bag_logits, trY[b]); opt.zero_grad(); loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), clip)       # grad-norm clip (default 1.0)
            opt.step()
        head.eval()
        m = val_metrics(head, valP, valY)                                     # any-patch val tuning
        if m["sel"] > best["sel"] + 1e-4:
            best = {**m, "ep": ep, "state": {k: v.clone() for k, v in head.state_dict().items()}}; bad = 0
        else:
            bad += 1
            if bad >= D.PATIENCE:
                break
    return best


def report(scores, Yt, thr, tag):
    """scores = the 9 TARGET columns (nf, 9). Prints all 9; also the 7-species sweep macro."""
    rows = []
    for c, name in enumerate(D.NAMES):
        pred = scores[:, c] >= thr[c]; y = Yt[:, c]
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F = 2 * P * R / max(P + R, 1e-9)
        rows.append((name, thr[c], P, R, F, int(y.sum())))
    macro = float(np.mean([r[4] for r in rows]))
    sel = float(np.mean([r[4] for r in rows if r[0] not in D.SWEEP_EXCLUDE]))
    meas = float(np.mean([r[4] for r in rows if r[5] >= 10]))
    print(f"\n=== {tag} ===  all9 {macro:.4f}   sel7 {sel:.4f}   measurable(sup>=10) {meas:.4f}")
    print(f"{'species':24}{'tau':>5}{'prec':>7}{'rec':>7}{'F1':>7}{'sup':>5}")
    for name, t, P, R, F, sup in rows:
        tag2 = "  (excl)" if name in D.SWEEP_EXCLUDE else ""
        print(f"{name:24}{t:5.2f}{P:7.2f}{R:7.2f}{F:7.2f}{sup:5d}{tag2}")
    return macro, sel, meas, rows


# --------------------------------------------------------------------------- #
# Splits + a full run                                                          #
# --------------------------------------------------------------------------- #
def load_splits(model, device):
    """Robot GT -> 50/50 stratified -> (valP, valY, testP, testY). Robot patches cached."""
    fids, Y = D.load_robot_gt()
    vi, ti = D.iterative_stratification(Y)
    robotP = extract_robot_patches(model, fids, device)
    return robotP[vi], Y[vi], robotP[ti], Y[ti]


def save_run(species, views, n_per, lr, hidden, r, model, device, splits, n_extract=None, months=None):
    """Full training run for a fixed config: train on iNat (species set), eval robot val+test over
    ALL 9, freeze thresholds, save checkpoints/mil.pt + mil_metrics.json. Returns metrics.
    n_extract reuses a larger cached extraction (e.g. the sweep's SIZE_MAX) and subsets to n_per.
    months = phenology window (None = all months)."""
    valP, valY, testP, testY = splits
    n_classes = len(species)
    trX, trL = extract_inat_bags(model, device, species, n_per=n_per, views=views, n_extract=n_extract, months=months)
    best = train_head(trX, trL, valP, valY, lr=lr, hidden=hidden, r=r, n_classes=n_classes)
    head = C.MILHead(n_classes=n_classes, hidden=hidden, r=r); head.load_state_dict(best["state"]); head.eval()
    thr = best["thr"]
    val_all9, val_sel, _, _ = report(frame_scores(head, valP)[:, :D.N], valY, thr, f"VAL (n={len(valY)})")
    test_all9, test_sel, test_meas, _ = report(frame_scores(head, testP)[:, :D.N], testY, thr, f"TEST (n={len(testY)})")

    crop = [list(v) for v in views]
    os.makedirs(f"{D.ROOT}/checkpoints", exist_ok=True)
    torch.save({"state": best["state"], "classes": D.NAMES, "n_out": n_classes,
                "species": [h for _, h in species], "thresholds": thr, "hidden": hidden, "r": r,
                "lr": lr, "crop": crop, "n_per": n_per, "months": months, "clip": DEF_CLIP,
                "best_epoch": best["ep"], "model_id": C.MODEL_ID}, f"{D.ROOT}/checkpoints/mil.pt")
    json.dump({"test_macro_f1": round(test_all9, 4), "test_sel7_f1": round(test_sel, 4),
               "test_measurable_f1": round(test_meas, 4), "val_macro_f1": round(val_all9, 4),
               "val_sel7_f1": round(val_sel, 4), "best_epoch": best["ep"],
               "species_set_size": n_classes, "species": [h for _, h in species], "crop": crop,
               "n_per": n_per, "lr": lr, "hidden": hidden, "r": r, "months": months, "clip": DEF_CLIP,
               "thresholds": dict(zip(D.NAMES, [round(t, 4) for t in thr]))},
              open(f"{D.ROOT}/checkpoints/mil_metrics.json", "w"), indent=2)
    print("\nsaved checkpoints/mil.pt + mil_metrics.json")
    return {"val_all9": val_all9, "val_sel": val_sel, "test_all9": test_all9, "test_sel": test_sel,
            "test_meas": test_meas, "ep": best["ep"], "thr": thr}


def main():
    device = C.get_device(); model = C.load_backbone(device)
    print("robot scene patches (24 native tiles -> 4704/frame) ...", flush=True)
    splits = load_splits(model, device)
    print(f"iNat bags ({DEF_NPER}/class, crop {DEF_VIEWS}) ...", flush=True)
    save_run(D.species_set("targets"), DEF_VIEWS, DEF_NPER, DEF_LR, DEF_HIDDEN, R_LSE, model, device, splits)


if __name__ == "__main__":
    main()
