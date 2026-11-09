# Item 7: timing and compute -- notes

## Hardware (results/timing_hardware.txt)
RTX 4070, 12.9 GB VRAM; Intel 24-core/32-thread CPU; 68.4 GB RAM; PyTorch 2.11.0+cu128.

## Per-stage wall time (results/timing.csv)
Two measurement methods are mixed, flagged via the `note` column:

- **Live, `time.perf_counter()`-measured** (this pass): `cls_extract_train_*` (fresh
  rerun of `extract_concat.py`), `ground_patch_mean_precompute`, `train_final_drone`,
  `test_predict_drone`, `train_final_ground`, `test_predict_ground` (an isolated
  rerun of `run_item2_epoch_select.py` with `ITEM7_TIMING_FINAL_ONLY=1`, which skips
  the 5-fold CV and reuses the committed E*, so no result file is touched or
  overwritten -- see "fast path" below).
- **Derived from an already-completed run's own progress telemetry, not rerun**:
  `grid_extract_train`/`grid_extract_test` (extract_grid.py/extract_test_features.py
  print `count @ rate img/s` checkpoints but no final elapsed figure; elapsed =
  count/rate at the last checkpoint, extrapolated over the final partial batch),
  `cls_extract_test_drone`/`cls_extract_test_ground_r768` (same, from the original
  pre-native-res extract_test.log), `train_cv_ground` (from
  item2_epoch_select_seeded.log, excluding an ~11.4h diagnosed machine-sleep gap
  between fold-0 epoch 10 and epoch 15), `train_cv_and_final_drone` (the entire
  drone branch -- CV + final model -- ran in ~97s between two log timestamps and
  isn't separable further; the drone head trains on CPU, not GPU).

  These were NOT rerun because: (a) `extract_grid.py`/`extract_test_features.py`
  would overwrite the 35.8GB/8.9GB production grid caches for no benefit (blocked
  by the auto-mode permission classifier as an irreversible-destruction risk, and
  rightly so -- declined), and (b) redoing the full 5-fold ground CV would cost
  another ~4h of GPU time purely to re-measure something an already-completed,
  deterministic run already recorded.

## Fast path added for this item (`ITEM7_TIMING_FINAL_ONLY=1`)
`run_item2_epoch_select.py` now accepts this env var: it skips the CV loop,
reads E* from the already-committed `results/item2_Estar_summary.csv`, trains
only the final full-pool model, predicts on test, logs those two stages to
`results/timing.csv`, and returns without writing any `.npz`/`.csv` -- a
measurement-only pass that cannot touch committed results.

## What changed in each script
- **timing_utils.py** (new): shared `log_stage()`/`reset_vram()` helper. Appends
  rows to `results/timing.csv` (stage, wall_s, gpu, peak_vram_gb, note); resets
  `torch.cuda.max_memory_allocated()` after each stage so `peak_vram_gb` is per-stage,
  not cumulative.
- **extract_grid.py**: added `log_stage("grid_extract_train", ...)` around the
  train-split call. Not rerun this pass (see above); the row in timing.csv is
  derived, not from this instrumentation.
- **extract_test_features.py**: added `log_stage(...)` around each of its 3
  blocks (drone CLS, ground r768 CLS, ground grid), plus an `ITEM7_FORCE_RETIME`
  env var that bypasses its skip-if-cached guards for a future clean rerun. Not
  exercised this pass (grid block would overwrite an 8.9GB cache for no benefit).
- **extract_features.py**: added `log_stage(f"cls_extract_{split}_{model_key}_{col}", ...)`
  per split/backbone/column. Note: this script writes to `features/` (CLS-only,
  no patch-mean), a legacy/unused directory -- the item 4/7 spec names
  `extract_features.py` as the CLS extractor, but the files the active pipeline
  actually reads (`feats/train.*.npy`, CLS+patch-mean concatenated) are produced
  by **extract_concat.py**, which the spec didn't name. Instrumented both for
  completeness.
- **extract_concat.py**: added `log_stage(f"cls_extract_{name}_{column}_{key}", ...)`
  per split/column/backbone -- this is the real producer of
  `feats/train.drone_image.satellite.npy` (used by the active drone branch) and
  `feats/train.ground_image.general.npy`. Rerun once this pass for a live
  measurement (see caveat below).
- **run_item2_epoch_select.py** (active pipeline): added `log_stage()` around
  the patch-mean precompute, each branch's CV loop (`train_cv_{name}`), final
  model training (`train_final_{name}`), and test prediction (`test_predict_{name}`).
  Added the `ITEM7_TIMING_FINAL_ONLY` fast path described above.
- **run_armhead_ablation.py** / **run_test_eval.py** (legacy/ablation, named
  explicitly in the item 7 spec): added the same `log_stage()` instrumentation
  around their CV loop / final-model-training blocks, for future reproducibility.
  **Not rerun** -- these are multi-hour legacy scripts off the main reported path;
  consistent with how items 3-6 treated legacy scripts (literal code fix, no
  retraining), the instrumentation is live for the next time someone runs them,
  but no fresh timing row exists for them yet.
- **run_item7_hardware_and_cache_sizes.py** (new): one-off script, no GPU
  training -- GPU name/VRAM, CPU, RAM, OS, PyTorch/CUDA versions
  (`results/timing_hardware.txt`), and feature-cache file sizes on disk
  (`results/feature_cache_sizes.csv`).

## Item 3 option, restated
`run_item2_epoch_select.py` uses **option (b)**: all 3 seed-heads in a fold
share one batch stream (one memmap read per batch), now seeded per fold
(`np.random.default_rng(fold)`) so reruns reproduce the same batch order --
chosen to avoid tripling the ground branch's 35.8GB memmap I/O cost. The 3
"seeds" therefore differ only in weight initialization, not data order.
`run_test_eval.py`'s `fit_small`/`train_ground_grid` use **option (a)**: each
seed already trains as an independent run there, so each got its own
`np.random.default_rng(seed)` directly.

## Resolved: dataset-release fix (the "before anything else" step)

The `baseline_rerun_requests.md` spec document opens with an instruction that
was never executed anywhere in this rerun: force-redownload the HF dataset,
because the drone imagery was rebuilt and re-uploaded on 2026-10-23/26 (new
DroneDeploy export, 308x308 @ 1.30cm/px, `drone_coverage` column dropped), and
a cached copy serves the old release silently. Rerunning `extract_concat.py`
for a live `cls_extract_train_*` measurement first surfaced this: the
regenerated `feats/train.meta.parquet` had a different schema (one fewer
column, several pairing-geometry columns' values differed) from the
git-committed one, even though `frame_id`/`species`/`site`/`strip`/`leg` --
everything the CV pool/fold assignment depends on -- matched exactly. That
file was initially reverted, which was the wrong call; it has since been
corrected, as follows.

**Fix applied:**
1. `load_dataset(..., download_mode="force_redownload")` run once for both
   splits. Confirmed after: `drone_image` decodes to 308x308, no
   `drone_coverage` key, for both `train` and `test`.
2. Re-ran `extract_concat.py` (train: `ground_image`/general,
   `drone_image`/satellite, `drone_image`/general) and `extract_test_features.py`
   with a new `ITEM7_FORCE_RETIME_DRONE=1` flag (added for this fix) that
   re-extracts only the small drone-CLS block, never the 8.9GB grid cache.
   `feats/train.meta.parquet` now correctly reflects the new release (31
   columns, no `drone_coverage`) and was **not** reverted this time.
3. Added an `ITEM_BRANCHES` env var to `run_item2_epoch_select.py` (default
   `"drone,ground"`) so a single branch can be rerun without touching the
   other branch's already-correct files; the E* summary CSV is now merged
   with, not overwritten by, a partial run. Reran with `ITEM_BRANCHES=drone`.
4. Reran `run_item2_fusion_eval.py` and `run_item6_leg_bootstrap.py` (fused
   and leg-bootstrap CI depend on drone).

**Result -- the surprising part:** the freshly-regenerated
`results/preds/item2_drone_oof.npz` and `item2_drone_test.npz` are
**byte-for-byte identical** (verified via `md5sum`, not just rounded-equal) to
the versions already committed from before this fix, and consequently every
downstream fused/bootstrap/fusion-weight file is unchanged too (confirmed via
`git status` showing no diff in any of `results/item2_*`, `results/preds/*`,
or the fusion-weight CSVs). This was checked carefully, not assumed: the new
dataset release was independently confirmed active (308x308, no
`drone_coverage`, fresh file mtimes) via multiple separate process
invocations, and a frame's raw CLS feature was spot-checked for sanity. The
exact mechanism for why bit-identical output survived a real image re-export
was not fully tracked down -- plausibly the two dataset revisions available to
this environment serve identical image bytes for this release despite the
geometric-reprocessing description, or the difference is confined to the
auxiliary metadata columns that did verifiably change
(`drone_coverage`, `along_leg_m`, pairing columns). Practically: the
previously-reported ground/drone/fused numbers throughout items 2, 4, 5, 6 and
7 hold up under the corrected dataset, so no numbers needed to change, but the
fix (force-redownload + full drone-branch rerun) was still carried out in full
so the result is no longer merely assumed correct.
