"""Leg-level 95% bootstrap confidence intervals and per-fold scores.

For the pooled out-of-fold predictions of ground, drone and fused, computes the
95% interval of each species' AP and of the macro AP by resampling the 38 legs
of the cross-validation pool (2,000 resamples, seed 0), and the per-fold macro
AP, macro F1 at 0.5 and per-species AP.

Reads results/preds/{ground,drone,fused}_cv.npz. Writes
results/ap_confidence_intervals.csv and results/ap_per_fold.csv. Runs on a CPU
in about a minute.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from assemble import pool, Y_pool

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
folds = pool["fold"].values
groups = pool["group"].values
K = 5
N_BOOT = 2000


def leg_bootstrap_ap(Yv, Pv, groups, n_boot=N_BOOT, seed=0):
    """95% intervals for per-species AP and macro AP, resampling survey LEGS, not
    frames: frames within a leg are near-duplicates (two capture events over the
    same marks, 5 m spacing), so a frame-level bootstrap would give intervals that
    are too tight. Macro AP is averaged WITHIN each resample, not from the
    per-species bounds."""
    rng = np.random.default_rng(seed)
    legs = pd.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in legs}
    boots, macro_boots = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[g] for g in rng.choice(legs, len(legs))])
        ok = Yv[idx].any(0) & ~Yv[idx].all(0)          # AP undefined without both classes
        ap = np.full(Yv.shape[1], np.nan)
        ap[ok] = average_precision_score(Yv[idx][:, ok], Pv[idx][:, ok], average=None)
        boots.append(ap)
        macro_boots.append(np.nanmean(ap))
    per_species_ci = np.nanpercentile(boots, [2.5, 97.5], axis=0)  # (2, 8)
    macro_ci = np.nanpercentile(macro_boots, [2.5, 97.5])          # (2,)
    return per_species_ci, macro_ci


# ---- load the pooled out-of-fold predictions (frame_id-aligned to `pool`) ----
configs = {}
for name in ["ground", "drone", "fused"]:
    d = np.load(f"results/preds/{name}_cv.npz", allow_pickle=True)
    assert (d["frame_id"] == pool["frame_id"].to_numpy()).all()
    configs[name] = d["ens"]

# ---- leg bootstrap CI: per-species + macro, per config ----
ci_rows = []
for name, P in configs.items():
    per_species_ci, macro_ci = leg_bootstrap_ap(Y_pool, P, groups)
    point = average_precision_score(Y_pool, P, average=None)
    for j, c in enumerate(CLASSES):
        ci_rows.append(dict(config=name, species=c, AP=point[j],
                             ci_lo=per_species_ci[0, j], ci_hi=per_species_ci[1, j]))
    ci_rows.append(dict(config=name, species="Macro average", AP=point.mean(),
                         ci_lo=macro_ci[0], ci_hi=macro_ci[1]))

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv("results/ap_confidence_intervals.csv", index=False)

print("=== leg-level 95% bootstrap CI (2000 resamples of 38 legs) ===")
for name in configs:
    print(f"\n-- {name} --")
    sub = ci_df[ci_df.config == name]
    for _, r in sub.iterrows():
        print(f"{r.species:24s} AP {r.AP:.3f}   95% CI [{r.ci_lo:.3f}, {r.ci_hi:.3f}]   width {r.ci_hi-r.ci_lo:.3f}")

# ---- per-fold scores: macro AP, macro F1@0.5, per-species AP, per config ----
fold_rows = []
for name, P in configs.items():
    for f in range(K):
        va = folds == f
        Yv, Pv = Y_pool[va], P[va]
        macro_ap = average_precision_score(Yv, Pv, average=None)
        macro_f1 = f1_score(Yv, Pv > 0.5, average="macro", zero_division=0)
        row = dict(config=name, fold=f, macro_AP=macro_ap.mean(), macro_F1=macro_f1)
        for j, c in enumerate(CLASSES):
            row[f"AP_{c.split()[0]}"] = macro_ap[j]
        fold_rows.append(row)

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv("results/ap_per_fold.csv", index=False)

print("\n=== per-fold macro AP / macro F1@0.5, per config ===")
for name in configs:
    sub = fold_df[fold_df.config == name]
    print(f"\n-- {name} --  mean(fold F1) {sub.macro_F1.mean():.3f} +/- {sub.macro_F1.std():.3f}")
    print(sub[["fold", "macro_AP", "macro_F1"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nwrote results/ap_confidence_intervals.csv, results/ap_per_fold.csv")
