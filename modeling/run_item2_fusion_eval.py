"""Item 2 follow-up: ground-only / drone-only / 2-way fused numbers (CV + test)
using the epoch-selected (E*) predictions from run_item2_epoch_select.py.
Late fusion = per-class weighted average, nested 5-fold CV grid search (21-pt
weight grid), matching run_2way_test_eval.py's established recipe -- but run
on the item2 (native-res ground, E*-selected, freshly recomputed drone) preds
instead of the old armhead_ablation_oof.npz ones. Writes NEW files only.
"""
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

from assemble import pool, Y_pool

CLASSES = ["Lupinus sericeus", "Balsamorhiza sagittata", "Gaillardia aristata",
           "Achillea millefolium", "Euphorbia virgata", "Sisymbrium altissimum",
           "Bromus tectorum", "Poa bulbosa"]
folds = pool["fold"].values
K = 5

# ---- load item2 E*-selected OOF / test predictions (3-seed ensemble) ----
g_oof = np.load("results/preds/item2_ground_oof.npz", allow_pickle=True)
d_oof = np.load("results/preds/item2_drone_oof.npz", allow_pickle=True)
assert (g_oof["frame_id"] == pool["frame_id"].to_numpy()).all()
assert (d_oof["frame_id"] == pool["frame_id"].to_numpy()).all()
ground_oof, drone_oof = g_oof["ens"], d_oof["ens"]

g_test_npz = np.load("results/preds/item2_ground_test.npz", allow_pickle=True)
d_test_npz = np.load("results/preds/item2_drone_test.npz", allow_pickle=True)
assert (g_test_npz["frame_id"] == d_test_npz["frame_id"]).all()
g_test, d_test = g_test_npz["ens"], d_test_npz["ens"]

test_ds = load_dataset("mpg-ranch/multimodal-survey", split="test")
assert (g_test_npz["frame_id"] == np.array(test_ds["frame_id"])).all()
Y_test = np.zeros((len(test_ds), 8), dtype=np.int8)
for i, ids in enumerate(test_ds["species"]):
    Y_test[i, list(ids)] = 1


def per_class(Y, P, thresh=0.5):
    ap = average_precision_score(Y, P, average=None)
    f1 = f1_score(Y, P > thresh, average=None, zero_division=0)
    prec = precision_score(Y, P > thresh, average=None, zero_division=0)
    rec = recall_score(Y, P > thresh, average=None, zero_division=0)
    return ap, f1, prec, rec


# ---- nested 5-fold fusion weights (CV) ----
Ws = np.linspace(0, 1, 21)
P_cv = np.zeros(Y_pool.shape, dtype=np.float32)
w_per_fold = np.zeros((K, 8))
for f in range(K):
    tr, va = folds != f, folds == f
    for j in range(8):
        y = Y_pool[tr, j]
        best_ap, best_w = -1, 0.5
        for w in Ws:
            apw = average_precision_score(y, w * ground_oof[tr, j] + (1 - w) * drone_oof[tr, j])
            if apw > best_ap:
                best_ap, best_w = apw, w
        w_per_fold[f, j] = best_w
        P_cv[va, j] = best_w * ground_oof[va, j] + (1 - best_w) * drone_oof[va, j]

# ---- final weights refit on full pool, for test ----
w_final = np.zeros(8)
for j in range(8):
    y = Y_pool[:, j]
    best_ap, best_w = -1, 0.5
    for w in Ws:
        apw = average_precision_score(y, w * ground_oof[:, j] + (1 - w) * drone_oof[:, j])
        if apw > best_ap:
            best_ap, best_w = apw, w
    w_final[j] = best_w
P_test = w_final[None, :] * g_test + (1 - w_final[None, :]) * d_test

# ---- assemble per-modality + fused, CV + test ----
configs_cv = {"ground": ground_oof, "drone": drone_oof, "fused": P_cv}
configs_test = {"ground": g_test, "drone": d_test, "fused": P_test}

rows = []
for name in ["ground", "drone", "fused"]:
    ap_cv, f1_cv, prec_cv, rec_cv = per_class(Y_pool, configs_cv[name])
    ap_te, f1_te, prec_te, rec_te = per_class(Y_test, configs_test[name])
    for j, c in enumerate(CLASSES):
        rows.append(dict(config=name, species=c,
                          cv_AP=ap_cv[j], cv_F1=f1_cv[j], cv_precision=prec_cv[j], cv_recall=rec_cv[j],
                          test_AP=ap_te[j], test_F1=f1_te[j], test_precision=prec_te[j], test_recall=rec_te[j]))
    rows.append(dict(config=name, species="MACRO",
                      cv_AP=ap_cv.mean(), cv_F1=f1_cv.mean(), cv_precision=prec_cv.mean(), cv_recall=rec_cv.mean(),
                      test_AP=ap_te.mean(), test_F1=f1_te.mean(), test_precision=prec_te.mean(), test_recall=rec_te.mean()))

df = pd.DataFrame(rows)
df.to_csv("results/item2_ground_drone_fused_cv_vs_test.csv", index=False)

# ---- item 4: save fused OOF/test prediction arrays + fusion weights ----
np.savez("results/preds/item2_fused_oof.npz", ens=P_cv, frame_id=pool["frame_id"].to_numpy())
np.savez("results/preds/item2_fused_test.npz", ens=P_test, frame_id=g_test_npz["frame_id"])

pd.DataFrame(w_per_fold, columns=[c.split()[0] for c in CLASSES]).assign(fold=range(K)) \
    .to_csv("results/item2_fusion_weights_perfold.csv", index=False)
pd.DataFrame({"species": CLASSES, "ground_weight": w_final}) \
    .to_csv("results/item2_fusion_weights_final.csv", index=False)

print(f"final fusion weights (full pool, ground share): " +
      ", ".join(f"{c.split()[0]}={w:.2f}" for c, w in zip(CLASSES, w_final)))

for name in ["ground", "drone", "fused"]:
    sub = df[df.config == name]
    macro = sub[sub.species == "MACRO"].iloc[0]
    print(f"\n=== {name.upper()} ===")
    print(f"{'species':24s} {'CV AP':>7s} {'Test AP':>8s}   {'CV F1':>7s} {'Test F1':>8s}   "
          f"{'CV Prec':>8s} {'Test Prec':>9s}   {'CV Rec':>7s} {'Test Rec':>9s}")
    for _, r in sub.iterrows():
        print(f"{r.species:24s} {r.cv_AP:7.3f} {r.test_AP:8.3f}   {r.cv_F1:7.3f} {r.test_F1:8.3f}   "
              f"{r.cv_precision:8.3f} {r.test_precision:9.3f}   {r.cv_recall:7.3f} {r.test_recall:9.3f}")

print("\nwrote results/item2_ground_drone_fused_cv_vs_test.csv")
print("wrote results/preds/item2_fused_oof.npz, results/preds/item2_fused_test.npz")
print("wrote results/item2_fusion_weights_perfold.csv, results/item2_fusion_weights_final.csv")
