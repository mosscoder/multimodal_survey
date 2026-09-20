import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from assemble import pool, Y_pool

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]


def leg_bootstrap_ap(Yv, Pv, groups, n_boot=2000, seed=0):
    """95% intervals for per-species AP, resampling survey LEGS, not frames: frames
    within a leg are near-duplicates (two capture events over the same marks, 5 m
    spacing), so a frame-level bootstrap would give intervals that are too tight."""
    rng = np.random.default_rng(seed)
    legs = pd.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in legs}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[g] for g in rng.choice(legs, len(legs))])
        ok = Yv[idx].any(0) & ~Yv[idx].all(0)          # AP undefined without both classes
        ap = np.full(Yv.shape[1], np.nan)
        ap[ok] = average_precision_score(Yv[idx][:, ok], Pv[idx][:, ok], average=None)
        boots.append(ap)
    return np.nanpercentile(boots, [2.5, 97.5], axis=0)  # (2, 8)


oof = np.load("results/topk_oof.npz")["k8_s0"]
ci = leg_bootstrap_ap(Y_pool, oof, pool["group"].values)

point = average_precision_score(Y_pool, oof, average=None)
for name, p, lo, hi in zip(CLASSES, point, ci[0], ci[1]):
    print(f"{name:24s} AP {p:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   width {hi-lo:.3f}")

pd.DataFrame({"species": CLASSES, "AP": point, "ci_lo": ci[0], "ci_hi": ci[1]}) \
    .to_csv("results/topk_k8_leg_bootstrap_ci.csv", index=False)
print("\nwrote results/topk_k8_leg_bootstrap_ci.csv")
