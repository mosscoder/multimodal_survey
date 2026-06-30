"""Staged, ISOLATED model-tuning sweeps for the MIL head (CLAUDE.md crop + any-patch eval).

One axis at a time, each stage reusing the prior winners (carried in sweeps/sweep_state.json):

  --stage species   : species SET {targets(9), all(38)} + Sky (extras = hard negatives)
  --stage phenology : training-image month window {summer=JJA, extended=May-Sep, all=balanced 12mo},
                      drawn EQUALLY per month (requires the month-balanced harvest)
  --stage hidden    : hidden in {0,32,64,128,256}
  --stage crop      : crop = square + zoom(z), z in {0.5,0.75,1.0}
  --stage size      : images/species in {64,128,256,512}
  --stage r         : LSE bag-pooling temperature r in {2,4,8,16,32,64,128}
  --stage lr        : LR in {5e-6,1e-5,5e-5,1e-4,5e-4,1e-3}
  --stage final     : full run with all winners -> checkpoints/mil.pt (reports ALL 9 on test)

rev-5 run order: species -> phenology -> hidden -> crop -> size -> r -> lr -> final. The species stage
samples the May-Sep (extended) window by default (the robot deploys mid-June); the phenology stage then
picks the window, and every later stage draws from that winner. LR is swept LAST (after r), default 5e-5
carried through earlier stages. Grad-norm clip is LOCKED at 1.0 for every run (train_mil.DEF_CLIP). The
Sky hard-negative (bald-eagle upper-corner crops, SegFormer sky-gated) is in EVERY set and obeys the
window (a summer deployment wants summer skies too).

Winner each stage = highest robot-VAL **sel** macro-F1, the macro over the 7 SELECTION species
(Achillea/yarrow + Sisymbrium/tumble mustard are EXCLUDED from selection; reported only at final).
all9 (the full 9-species macro) is recorded for reference but does not pick winners.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import common as C
import data as D
import train_mil as T

STATE = os.path.join(HERE, "sweep_state.json")

SPECIES_SETS = ["targets", "all"]            # rev-5: 'grasses' dropped (irrelevant)
PHENOLOGY = {"summer": [6, 7, 8], "extended": [5, 6, 7, 8, 9], "all": list(range(1, 13))}
# observed-month windows; EACH level draws equally across its months (round-robin, capped by
# availability). "all" lists all 12 explicitly so it balances too, rather than relying on None
# (= no balancing, natural order) — matters for scarce species the harvest could not pre-balance.
PHENO_LEVELS = ["summer", "extended", "all"]
DEF_PHENO = "extended"                        # species stage samples May-Sep by default (covers the
                                              # mid-June deployment with enough data; no class starved at 128)
N_PER_TUNE = 128                              # rev-5: tune-stage imgs/species (was 256) — keeps the
                                              # phenology windows comparable (only Holosteum < 128, in JJA)
BASE_VIEWS = [["native", 1.0]]
HIDDENS = [0, 32, 64, 128, 256]
LRS = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
ZOOMS = [0.5, 0.75, 1.0]
SIZES = [64, 128, 256, 512]                  # rev-4: capped at 512 (rev-3 showed >512 hurt sel7)
R_LIST = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
SIZE_MAX = max(SIZES)
DEF_HIDDEN, DEF_LR, DEF_R = 64, 5e-5, 8.0     # rev-4: default LR 5e-5 (LR swept LAST, after r)


def fresh_state(best_species):
    return {"best_species": best_species, "best_pheno": DEF_PHENO, "best_hidden": DEF_HIDDEN,
            "best_lr": DEF_LR, "best_views": BASE_VIEWS, "best_n_per": N_PER_TUNE, "best_r": DEF_R,
            "stages": {}}


def load_state():
    return json.load(open(STATE))


def save_state(s):
    json.dump(s, open(STATE, "w"), indent=2)


def _views_t(views):
    return [tuple(v) for v in views]


def _species(state):
    sp = D.species_set(state["best_species"]); return sp, len(sp)


def _months(state):
    """The winning phenology month window (None = all months)."""
    return PHENOLOGY[state.get("best_pheno", DEF_PHENO)]


def _pick(cands):
    """Winner = highest val sel (7-species selection macro)."""
    return max(range(len(cands)), key=lambda i: cands[i]["sel"])


def _candidate(b, **fields):
    return {**fields, "sel": round(b["sel"], 4), "all9": round(b["all9"], 4), "ep": b["ep"]}


def _emit(stage, result, state):
    state["stages"][stage] = result
    save_state(state)
    print(f"\nRESULT_JSON: {json.dumps(result)}")


def _line(label, b):
    print(f"  {label}  val sel7 {b['sel']:.4f}  (all9 {b['all9']:.4f})  ep {b['ep']}")


# --------------------------------------------------------------------------- #
# Stages                                                                       #
# --------------------------------------------------------------------------- #
def stage_species(model, device, splits):
    valP, valY = splits[0], splits[1]
    months = PHENOLOGY[DEF_PHENO]                         # species stage samples the default (May-Sep) window
    cands = []
    for name in SPECIES_SETS:
        sp = D.species_set(name)
        trX, trL = T.extract_inat_bags(model, device, sp, N_PER_TUNE, _views_t(BASE_VIEWS), months=months)
        b = T.train_head(trX, trL, valP, valY, lr=DEF_LR, hidden=DEF_HIDDEN, r=DEF_R, n_classes=len(sp))
        cands.append(_candidate(b, species=name, n_species=len(sp))); _line(f"species {name:<8} ({len(sp):2d})", b)
    win = _pick(cands)
    state = fresh_state(SPECIES_SETS[win])
    _emit("species", {"stage": "species", "fixed": {"pheno": DEF_PHENO, "hidden": DEF_HIDDEN, "lr": DEF_LR,
          "r": DEF_R, "crop": BASE_VIEWS, "n_per": N_PER_TUNE}, "grid": cands, "winner": cands[win]}, state)


def stage_phenology(model, device, splits):
    valP, valY = splits[0], splits[1]
    state = load_state(); sp, nsp = _species(state)
    cands = []
    for name in PHENO_LEVELS:
        months = PHENOLOGY[name]
        trX, trL = T.extract_inat_bags(model, device, sp, N_PER_TUNE, _views_t(BASE_VIEWS), months=months)
        b = T.train_head(trX, trL, valP, valY, lr=DEF_LR, hidden=DEF_HIDDEN, r=DEF_R, n_classes=nsp)
        cands.append(_candidate(b, pheno=name, months=(months or "all"))); _line(f"pheno {name:<9}", b)
    win = _pick(cands); state["best_pheno"] = PHENO_LEVELS[win]
    _emit("phenology", {"stage": "phenology", "fixed": {"species": state["best_species"], "hidden": DEF_HIDDEN,
          "lr": DEF_LR, "r": DEF_R, "crop": BASE_VIEWS, "n_per": N_PER_TUNE}, "grid": cands, "winner": cands[win]}, state)


def stage_hidden(model, device, splits):
    valP, valY = splits[0], splits[1]
    state = load_state(); sp, nsp = _species(state); r = state["best_r"]; months = _months(state)
    trX, trL = T.extract_inat_bags(model, device, sp, N_PER_TUNE, _views_t(BASE_VIEWS), months=months)
    cands = []
    for h in HIDDENS:
        b = T.train_head(trX, trL, valP, valY, lr=DEF_LR, hidden=h, r=r, n_classes=nsp)
        cands.append(_candidate(b, hidden=h)); _line(f"hidden {h:<4}", b)
    win = _pick(cands); state["best_hidden"] = HIDDENS[win]
    _emit("hidden", {"stage": "hidden", "fixed": {"species": state["best_species"], "pheno": state["best_pheno"],
          "lr": DEF_LR, "r": r, "crop": BASE_VIEWS, "n_per": N_PER_TUNE}, "grid": cands, "winner": cands[win]}, state)


def stage_lr(model, device, splits):
    valP, valY = splits[0], splits[1]
    state = load_state(); sp, nsp = _species(state); hidden = state["best_hidden"]; r = state["best_r"]
    views = _views_t(state["best_views"]); n_per = state["best_n_per"]   # LR swept LAST -> on the full config
    months = _months(state)
    trX, trL = T.extract_inat_bags(model, device, sp, n_per=n_per, views=views, n_extract=SIZE_MAX, months=months)
    cands = []
    for lr in LRS:
        b = T.train_head(trX, trL, valP, valY, lr=lr, hidden=hidden, r=r, n_classes=nsp)
        cands.append(_candidate(b, lr=lr)); _line(f"lr {lr:<8}", b)
    win = _pick(cands); state["best_lr"] = LRS[win]
    _emit("lr", {"stage": "lr", "fixed": {"species": state["best_species"], "pheno": state["best_pheno"],
          "hidden": hidden, "r": r, "crop": state["best_views"], "n_per": n_per}, "grid": cands, "winner": cands[win]}, state)


def stage_crop(model, device, splits):
    valP, valY = splits[0], splits[1]
    state = load_state(); sp, nsp = _species(state); hidden = state["best_hidden"]
    lr = state["best_lr"]; r = state["best_r"]; months = _months(state)
    cands = []
    for z in ZOOMS:
        views = [("square",), ("native", z)]
        trX, trL = T.extract_inat_bags(model, device, sp, N_PER_TUNE, views, months=months)
        b = T.train_head(trX, trL, valP, valY, lr=lr, hidden=hidden, r=r, n_classes=nsp)
        cands.append(_candidate(b, zoom=z, crop=[["square"], ["native", z]])); _line(f"zoom {z:<5}", b)
    win = _pick(cands); state["best_views"] = [["square"], ["native", ZOOMS[win]]]
    _emit("crop", {"stage": "crop", "fixed": {"species": state["best_species"], "pheno": state["best_pheno"],
          "hidden": hidden, "lr": lr, "r": r, "n_per": N_PER_TUNE}, "grid": cands, "winner": cands[win]}, state)


def stage_size(model, device, splits):
    valP, valY = splits[0], splits[1]
    state = load_state(); sp, nsp = _species(state); hidden = state["best_hidden"]
    lr = state["best_lr"]; r = state["best_r"]; views = _views_t(state["best_views"]); months = _months(state)
    cands = []
    for n in SIZES:
        trX, trL = T.extract_inat_bags(model, device, sp, n_per=n, views=views, n_extract=SIZE_MAX, months=months)
        b = T.train_head(trX, trL, valP, valY, lr=lr, hidden=hidden, r=r, n_classes=nsp)
        cands.append(_candidate(b, n_per=n)); _line(f"n_per {n:<4}", b)
    win = _pick(cands); state["best_n_per"] = SIZES[win]
    _emit("size", {"stage": "size", "fixed": {"species": state["best_species"], "pheno": state["best_pheno"],
          "hidden": hidden, "lr": lr, "r": r, "crop": state["best_views"]}, "grid": cands, "winner": cands[win]}, state)


def stage_r(model, device, splits):
    valP, valY = splits[0], splits[1]
    state = load_state(); sp, nsp = _species(state); hidden = state["best_hidden"]
    lr = state["best_lr"]; views = _views_t(state["best_views"]); n_per = state["best_n_per"]; months = _months(state)
    trX, trL = T.extract_inat_bags(model, device, sp, n_per=n_per, views=views, n_extract=SIZE_MAX, months=months)
    cands = []
    for r in R_LIST:
        b = T.train_head(trX, trL, valP, valY, lr=lr, hidden=hidden, r=r, n_classes=nsp)
        cands.append(_candidate(b, r=r)); _line(f"r {r:<6}", b)
    win = _pick(cands); state["best_r"] = R_LIST[win]
    _emit("r", {"stage": "r", "fixed": {"species": state["best_species"], "pheno": state["best_pheno"],
          "hidden": hidden, "lr": lr, "crop": state["best_views"], "n_per": n_per}, "grid": cands, "winner": cands[win]}, state)


def stage_final(model, device, splits):
    state = load_state(); sp, _ = _species(state)
    views = _views_t(state["best_views"]); n_per = state["best_n_per"]; months = _months(state)
    lr = state["best_lr"]; hidden = state["best_hidden"]; r = state["best_r"]
    print(f"FINAL run  species {state['best_species']}  pheno {state['best_pheno']}  hidden {hidden}  "
          f"lr {lr}  crop {state['best_views']}  n_per {n_per}  r {r}  clip {T.DEF_CLIP}")
    out = T.save_run(sp, views, n_per, lr, hidden, r, model, device, splits, n_extract=SIZE_MAX, months=months)
    _emit("final", {"stage": "final", "config": {"species": state["best_species"], "pheno": state["best_pheno"],
          "hidden": hidden, "lr": lr, "crop": state["best_views"], "n_per": n_per, "r": r, "clip": T.DEF_CLIP},
          "val_sel7": round(out["val_sel"], 4), "val_all9": round(out["val_all9"], 4),
          "test_sel7": round(out["test_sel"], 4), "test_all9": round(out["test_all9"], 4),
          "test_measurable": round(out["test_meas"], 4), "best_epoch": out["ep"]}, state)


STAGES = {"species": stage_species, "phenology": stage_phenology, "hidden": stage_hidden,
          "crop": stage_crop, "size": stage_size, "r": stage_r, "lr": stage_lr, "final": stage_final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    a = ap.parse_args()
    device = C.get_device(); model = C.load_backbone(device)
    print("robot scene patches (cached after first run) ...", flush=True)
    splits = T.load_splits(model, device)
    STAGES[a.stage](model, device, splits)


if __name__ == "__main__":
    main()
