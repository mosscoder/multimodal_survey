"""Targets, robot ground truth, the 50/50 stratified split, the iNaturalist dataset
accessor, a feature cache, and the macro-F1 metric. Model-agnostic shared core.
"""
from __future__ import annotations

import hashlib
import json
import os
import random

import numpy as np
import torch
from datasets import load_from_disk

import common as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")                     # gitignored; feature caches live here
ROBOT_LABELS = ("/Users/kdoherty/multimodal_survey/missions/site_1/strip_5/runs/"
                "strip_5_2026-06-16_12-36-44/labels/image_multilabel.json")

# 9 target species: (display name, iNaturalist HF-dataset dir). All trained and scored.
TARGETS = [
    ("Lupinus sericeus",       "lupinus_sericeus"),
    ("Poa bulbosa",            "poa_bulbosa"),
    ("Tragopogon dubius",      "tragopogon_dubius"),
    ("Gaillardia aristata",    "gaillardia_aristata"),
    ("Balsamorhiza sagittata", "balsamorhiza_sagittata"),
    ("Bromus tectorum",        "bromus_tectorum"),
    ("Achillea millefolium",   "achillea_millefolium"),
    ("Sisymbrium altissimum",  "sisymbrium_altissimum"),
    ("Thinopyrum intermedium", "thinopyrum_intermedium"),
]
NAMES = [n for n, _ in TARGETS]
N = len(TARGETS)

# Sweep SELECTION excludes 2 rare, noise-prone species — Achillea millefolium (yarrow) and
# Sisymbrium altissimum (tumble mustard). Every sweep stage selects on the macro-F1 over the
# remaining 7; yarrow + tumble mustard are reported only on the FINAL test, never used to tune.
SWEEP_EXCLUDE = ("Achillea millefolium", "Sisymbrium altissimum")
SWEEP_IDX = [c for c, n in enumerate(NAMES) if n not in SWEEP_EXCLUDE]   # the 7 selection species

# --- species ablation -----------------------------------------------------
# The other iNaturalist species in the HF dataset, used as hard negatives / auxiliary
# one-vs-all classes. They are NEVER scored — the metric is always the 9 TARGETS, which
# are kept FIRST in every species set so target class indices 0..8 stay stable.
EXTRA_HFDIRS = [
    "antennaria_parvifolia", "bassia_scoparia", "chenopodium_album", "cirsium_undulatum",
    "collinsia_parviflora", "collomia_linearis", "erigeron_pumilus", "euphorbia_virgata",
    "filago_arvensis", "helianthus_annuus", "helianthus_maximiliani", "hesperostipa_comata",
    "holosteum_umbellatum", "linum_lewisii", "lithophragma_glabrum", "lomatium_triternatum",
    "microsteris_gracilis", "myosotis_stricta", "phacelia_linearis", "plantago_patagonica",
    "poa_secunda", "pseudoroegneria_spicata", "salsola_tragus", "silene_latifolia",
    "taraxacum_officinale", "verbascum_blattaria", "veronica_verna",
    # rev-5: appended (NOT alphabetical) to keep the existing species' seed indices stable. Two species
    # below the 1000 cap (Helianthella uniflora ~535, Polygonum douglasii ~313), now harvested
    # month-balanced and added as extra hard negatives. TARGETS / D.N (the 9 scored) are unchanged.
    "helianthella_uniflora", "polygonum_douglasii",
]
GRASS_HFDIRS = ["poa_secunda", "pseudoroegneria_spicata", "hesperostipa_comata"]  # target-grass look-alikes

# Auxiliary 'Sky' hard-negative class (rev-4): bald eagle (Haliaeetus leucocephalus) images, harvested
# into the SEPARATE birds/ dataset and cropped at the UPPER corners (sky/cloud) at train time. Appended
# LAST in every species set so target indices 0..8 are unchanged; scored only as a negative, never a target.
SKY = ("Sky", "haliaeetus_leucocephalus")


def _disp(hfdir):
    return hfdir.replace("_", " ").capitalize()


def species_set(name):
    """Ordered species list [(display, hfdir), ...] for a set name; TARGETS always first 9, the
    'Sky' negative always LAST. 'targets' = 9 + Sky; 'grasses' = 9 + look-alike grasses + Sky;
    'all' = every plant species + Sky. Sky is in EVERY set (rev-4) — scored only as a negative."""
    extra = {"targets": [], "grasses": GRASS_HFDIRS, "all": EXTRA_HFDIRS}[name]
    return list(TARGETS) + [(_disp(h), h) for h in extra] + [SKY]


CANON = species_set("all")                              # canonical order for stable per-species seeds
HFDIR_IDX = {h: i for i, (_, h) in enumerate(CANON)}

HF_DATASET = os.path.join(C.INAT_ROOT, "plants", "out", "dataset")     # 36 plant species
BIRDS_DATASET = os.path.join(C.INAT_ROOT, "birds", "out", "dataset")   # sky/cloud negatives (bald eagle)
SEED = 1312
VAL_FRAC = 0.50          # 50/50 val/test (was 0.80): stabilizes the held-out test for rare species
EPOCHS = 80
PATIENCE = 12
WD = 1e-4
BATCH = 256
EXTRACT_BATCH = 48
THR_GRID = np.linspace(0.02, 0.98, 97)


def load_robot_gt():
    """-> (sorted frame ids, Y (n, 9) multilabel presence matrix)."""
    d = json.load(open(ROBOT_LABELS))
    fids = sorted(d["labels"].keys())
    Y = np.zeros((len(fids), N), np.int64)
    for r, fid in enumerate(fids):
        present = set(d["labels"][fid]["present"])
        for c, name in enumerate(NAMES):
            if name in present:
                Y[r, c] = 1
    return fids, Y


def iterative_stratification(Y, val_frac=VAL_FRAC, seed=SEED):
    """Greedy iterative stratification (Sechidis 2011) -> (val idx, test idx), rarest first."""
    rng = random.Random(seed)
    n, k = Y.shape
    ratio = {"val": val_frac, "test": 1.0 - val_frac}
    t_fold = {f: ratio[f] * n for f in ratio}
    t_lab = {f: [ratio[f] * int(Y[:, c].sum()) for c in range(k)] for f in ratio}
    assigned = [None] * n
    remaining = set(range(n))
    while remaining:
        rem_pos = [(c, sum(1 for i in remaining if Y[i, c])) for c in range(k)]
        rem_pos = [(c, p) for c, p in rem_pos if p > 0]
        if rem_pos:
            c = min(rem_pos, key=lambda x: x[1])[0]
            samples = [i for i in remaining if Y[i, c]]
        else:
            c, samples = None, list(remaining)
        rng.shuffle(samples)
        for i in samples:
            if i not in remaining:
                continue
            fold = max(ratio, key=lambda f: (t_lab[f][c], t_fold[f])) if c is not None else max(ratio, key=lambda f: t_fold[f])
            assigned[i] = fold
            remaining.discard(i)
            t_fold[fold] -= 1
            for cc in range(k):
                if Y[i, cc]:
                    t_lab[fold][cc] -= 1
    return sorted(i for i in range(n) if assigned[i] == "val"), sorted(i for i in range(n) if assigned[i] == "test")


_DS = {}


def load_ds(hfdir):
    if hfdir not in _DS:
        base = BIRDS_DATASET if hfdir == SKY[1] else HF_DATASET    # route the Sky slug to the birds dataset
        _DS[hfdir] = load_from_disk(os.path.join(base, hfdir))
    return _DS[hfdir]


_DSF = {}


def load_full(hfdir):
    """Full iNaturalist dataset for a species = train + test splits concatenated (~2500 obs).
    The robot frames are the only held-out eval, so the iNat 'test' split is just more free
    training data — all of it is used to train the MIL head."""
    if hfdir not in _DSF:
        from datasets import concatenate_datasets
        ds = load_ds(hfdir)
        parts = [ds[k] for k in ("train", "test") if k in ds]
        _DSF[hfdir] = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return _DSF[hfdir]


_FP = {}


def ds_fingerprint(hfdir):
    """Short CONTENT hash of a species' full dataset (row count + occurrence ids). Folded into the
    feature-cache signature so caches invalidate automatically when the dataset is re-harvested
    (e.g. recency -> month-balanced) instead of silently reusing stale features. Memoized."""
    if hfdir not in _FP:
        tr = load_full(hfdir)
        ids = tr["occurrence_id"] if "occurrence_id" in tr.column_names else list(range(len(tr)))
        _FP[hfdir] = hashlib.md5((str(len(tr)) + "|" + ",".join(map(str, ids))).encode()).hexdigest()[:10]
    return _FP[hfdir]


def cached(path, sig, build):
    """Load a cached dict if its signature matches, else build + save it."""
    if os.path.exists(path):
        d = torch.load(path)
        if d.get("sig") == sig:
            print(f"  cached: {os.path.basename(path)}", flush=True)
            return d
    d = build(); d["sig"] = sig
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(d, path)
    return d


def _f1(pred, y):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def macro_f1(scores, Y, thr=None):
    """Per-(frame,species) scores (n, 9) + GT -> (macro F1, per-class F1, per-class thresholds).
    thr=None tunes each species' cutoff to maximize its F1; else applies the given cutoffs."""
    f1s, ths = [], []
    for c in range(N):
        sc, yc = scores[:, c], Y[:, c]
        if thr is None:
            bf, bt = -1.0, 0.5
            for t in THR_GRID:
                f = _f1(sc >= t, yc)
                if f > bf:
                    bf, bt = f, float(t)
            f1s.append(bf); ths.append(bt)
        else:
            ths.append(thr[c]); f1s.append(_f1(sc >= thr[c], yc))
    return float(np.mean(f1s)), f1s, ths
