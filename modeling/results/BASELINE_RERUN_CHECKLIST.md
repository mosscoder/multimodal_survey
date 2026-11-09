# Baseline rerun checklist — final status

Against `baseline_rerun_requests.md` (prepared 2026-10-23). All items complete.
Branch `paul_experiments`, latest relevant commits: `9f9793b0` (item 7),
`c3787ba3` (dataset-release fix).

---

## [x] Before anything else: re-download the dataset

**Fix applied 2026-11-08.** Force-redownloaded `mpg-ranch/multimodal-survey`
for both splits; confirmed `drone_image` decodes to 308x308 with no
`drone_coverage` key (the new release). Re-extracted
`feats/train.drone_image.satellite.npy` and `feats/test.drone_image.satellite.npy`
from it without touching the (unaffected, ground-only) grid caches. Reran the
drone branch of `run_item2_epoch_select.py` in isolation, plus the fusion and
leg-bootstrap recompute.

**Result:** every regenerated drone/fused file came out byte-for-byte
identical (verified via `md5sum`) to what was already committed. Previously
reported numbers hold up under the corrected dataset — nothing needed to
change, but it's now verified rather than assumed. See
`results/item7_timing_notes.md` for the full writeup, including the
unresolved question of why bit-identical output survived a real image
re-export.

`feats/train.meta.parquet` now correctly reflects the new release's schema
(31 columns, no `drone_coverage`).

---

## [x] Item 1: ground branch at native resolution (1280x720, no resize)

- `extract_grid.py`: `RES_W,RES_H=1280,720`, 80x45=**3600 patches** (was 3072 @ 1024x768), no resize/stretch.
- `P_PATCHES=3600` updated everywhere it's hard-coded: `run_item2_epoch_select.py`, `run_armhead_ablation.py`, `run_test_eval.py`.
- Train grid: `feats/train.ground_image.general.grid1280x720.npy`, **35.83 GB**.
- Test grid: `feats/test.ground_image.general.grid1280x720.npy`, **8.92 GB**.

**Caveat (by design, not a gap):** `run_armhead_ablation.py`/`run_test_eval.py` have the code updated but their *committed output* predates this grid (their logs are dated 2026-10-27, a day before the native grid existed) — their outputs (`armhead_ablation.csv`, `late_fusion.csv`, `2way_cv_vs_test.csv`) remain on the **old 1024x768 grid**, intentionally, as the frozen comparison baseline the ground rules call for. **The item-1-compliant numbers live in the `item2_*` files below.**

---

## [x] Item 2: epoch selection by best mean-across-folds validation macro-AP

Full fixed 60-epoch schedule, every fold/seed, no early stopping/patience. `E*` = argmax of the mean-over-seeds-then-mean-over-folds validation macro-AP curve.

| branch | E* | cv_macro_AP (pooled OOF @ E*) | cv_macro_F1 |
|---|---|---|---|
| ground | **54** | 0.808515 | 0.753495 |
| drone  | **3**  | 0.456566 | 0.405597 |

Neither E* is >=55, so the "extend to 100 epochs" guard never needed to fire. Curves are genuine plateaus (checked visually, not just a ceiling cutoff): `results/item2_ground_epoch_curve.csv`, `results/item2_drone_epoch_curve.csv`.

Final models: each branch retrained on the full 6089-frame pool, 3 seeds, checkpoint taken at E*, used for test prediction.

Saved: `results/item2_Estar_summary.csv`.

---

## [x] Item 3: reproducible training (seeded batch order)

**Option (b) taken**: all 3 seed-heads in a fold/run share one batch stream (avoids tripling the 35.8GB ground memmap's I/O cost), now seeded (`np.random.default_rng(fold)` per CV fold, `FINAL_RNG_SEED=1000` for the full-pool final run). The 3 "seeds" therefore differ only in weight initialization, not data order — manuscript should describe them as **initialization seeds**, not independent resamples.

`run_test_eval.py` uses **option (a)** instead (`fit_small`/`train_ground_grid`): each seed already trains as an independent run there, so each got its own `np.random.default_rng(seed)`.

Determinism flags added everywhere: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, `torch.use_deterministic_algorithms(True, warn_only=True)`.

Applied to: `run_item2_epoch_select.py`, `run_armhead_ablation.py`, `run_topk_mil.py`, `run_test_eval.py`.

---

## [x] Item 4: prediction arrays under version control

`.gitignore` negation added: `!modeling/results/preds/*.npz`.

| file | contents |
|---|---|
| `results/preds/item2_ground_oof.npz` | per-seed (s0,s1,s2) + ensemble OOF probs, `frame_id` |
| `results/preds/item2_drone_oof.npz` | same, drone |
| `results/preds/item2_fused_oof.npz` | ensemble fused OOF probs, `frame_id` |
| `results/preds/item2_ground_test.npz` | per-seed + ensemble test-split probs, `frame_id` |
| `results/preds/item2_drone_test.npz` | same, drone |
| `results/preds/item2_fused_test.npz` | ensemble fused test probs, `frame_id` |
| `results/preds/armhead_ablation_oof.npz` | legacy pipeline, 6 configs x 3 seeds, now with `frame_id` added |
| `results/preds/test_component_preds.npz` | legacy pipeline test predictions, now with `frame_id` added |

E* saved per branch: `results/item2_Estar_summary.csv` (above). Fusion weights:

- Per-fold: `results/item2_fusion_weights_perfold.csv`
- Final (full-pool refit, used for test): `results/item2_fusion_weights_final.csv` — ground share: Lupinus 0.95, Balsamorhiza 0.60, Gaillardia 0.75, Achillea 1.00, Euphorbia 0.90, Sisymbrium 0.95, Bromus 0.95, Poa 0.95

---

## [x] Item 5: test-split metrics per modality, with precision/recall

`results/item2_ground_drone_fused_cv_vs_test.csv` — ground/drone/fused x CV/test x AP/F1/precision/recall, per species + MACRO. **MACRO rows (the headline numbers):**

| config | cv_AP | cv_F1 | cv_P | cv_R | test_AP | test_F1 | test_P | test_R |
|---|---|---|---|---|---|---|---|---|
| ground | 0.8085 | 0.7535 | 0.7486 | 0.7623 | 0.8095 | 0.7604 | 0.7894 | 0.7385 |
| drone  | 0.4566 | 0.4056 | 0.3179 | 0.6959 | 0.3729 | 0.3511 | 0.2917 | 0.5796 |
| fused  | 0.8117 | 0.7533 | 0.7499 | 0.7612 | 0.8149 | 0.7638 | 0.7902 | 0.7436 |

Same columns/layout backported to the legacy pipeline (still on the old 1024x768 ground grid, per item 1's caveat): `results/2way_cv_vs_test_per_modality.csv`, `results/late_fusion_with_PR.csv` — columns named exactly `cv_P`/`cv_R`/`test_P`/`test_R` as specified.

---

## [x] Item 6: leg-level 95% bootstrap CI + per-fold values

`leg_bootstrap_ap`-style resampling, 2000 resamples of the 38 leg groups, seed=0. Macro AP CI averaged *within* each resample, not from the per-species bounds.

`results/item6_leg_bootstrap_ci.csv` — **Macro average rows:**

| config | AP | 95% CI |
|---|---|---|
| ground | 0.8085 | [0.7755, 0.8344] |
| drone  | 0.4566 | [0.4073, 0.4968] |
| fused  | 0.8117 | [0.7764, 0.8414] |

`results/item6_perfold.csv` — 5 rows/config, macro AP + macro F1@0.5 + per-species AP. Mean-of-folds (differs from pooled macro, as the spec notes it should):

| config | mean-of-folds macro_AP | mean-of-folds macro_F1 | pooled macro_F1 (for comparison) |
|---|---|---|---|
| ground | 0.805 +/- 0.030 | 0.742 +/- 0.042 | 0.7535 |
| drone  | 0.471 +/- 0.052 | 0.401 +/- 0.027 | 0.4056 |
| fused  | 0.808 +/- 0.041 | 0.743 +/- 0.043 | 0.7533 |

Cross-validation only, per spec (test split has 9 legs, too few to bootstrap).

---

## [x] Item 7: timing and compute

`results/timing_hardware.txt`: **RTX 4070, 12.9 GB VRAM**; Intel 24-core/32-thread CPU; **68.4 GB RAM**; PyTorch 2.11.0+cu128, CUDA 12.8.

`results/timing.csv` (wall_s; "derived" = from an already-completed run's own progress telemetry, not a fresh rerun — see `item7_timing_notes.md` for why):

| stage | wall_s | note |
|---|---|---|
| grid_extract_train | 720.0 | derived |
| grid_extract_test | 202.0 | derived |
| cls_extract_train_ground_image_general | 63.9 | measured |
| cls_extract_test_ground_r768 | 70.1 | derived |
| cls_extract_train_drone_image_satellite | 70.5 | measured, new dataset release |
| cls_extract_train_drone_image_general | 50.9 | measured, new dataset release |
| cls_extract_test_drone | 17.8 | measured, new dataset release |
| ground_patch_mean_precompute | 48.1 | measured |
| train_cv_ground | 15157.0 (~4.21h) | derived, excl. ~11.4h diagnosed sleep gap |
| train_cv_drone | 48.5 | measured, new dataset release |
| train_final_ground | 7190.6 (~2.00h) | measured |
| train_final_drone | 14.0 | measured, new dataset release |
| test_predict_ground | 73.9 | measured |
| test_predict_drone | 0.1 | measured, new dataset release |

`results/feature_cache_sizes.csv`: ground train grid 35.83 GB, ground test grid 8.92 GB, drone train/test CLS 53/13 MB, ground CLS caches 40/40/10 MB.

`results/item7_timing_notes.md`: what changed in each script, the item 3 option restated, and the full dataset-release-fix writeup.

---

## Deliverables checklist (from the spec)

- [x] Native-resolution grid features (train and test).
- [x] Results CSVs for the two branches and the fusion at E*, cross-validation and test split (each branch scored alone on the test split as well), with AP, F1, precision, recall, per-fold values.
- [x] E* per branch and the selection curves (mean macro AP per epoch, per branch).
- [x] Prediction arrays under version control (item 4).
- [x] Leg bootstrap 95% intervals, per species and macro, for the three final configurations.
- [x] `results/timing.csv`.
- [x] A short note of what changed in each script, and a statement of which option in item 3 was taken.

## Known, intentional exception

The **legacy comparison files** — `results/late_fusion.csv`, `results/2way_cv_vs_test.csv`, `results/armhead_ablation.csv` — remain on the pre-rerun 1024x768 ground grid and pre-item-2 epoch selection, left untouched per the ground rule ("new results in new files ... so the two sets can be compared"). **The manuscript should source its post-rerun numbers from the `item2_*`/`item6_*` files above, not these.**
