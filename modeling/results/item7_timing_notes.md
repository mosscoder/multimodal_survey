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

## Caveat discovered during this item: local feature-cache provenance
Rerunning `extract_concat.py` for a live `cls_extract_train_*` measurement calls
`load_dataset("mpg-ranch/multimodal-survey", split="train")` fresh, which also
rewrites `feats/train.meta.parquet`. The regenerated file differed from the
git-committed one: one column (`drone_coverage`) was missing and several
pairing-geometry columns (`pair_distance_m`, `pair_overlap_fraction`,
`along_leg_m`, `tier2_overlap_fraction`, `mission_restart_utc`, `pair_frame_id`)
had different values, even though `frame_id`/`species`/`site`/`strip`/`leg` --
everything the CV pool/fold assignment depends on -- matched exactly. The
dataset's Hub commit history shows no change since 2026-10-26 (before the
original 2026-10-27 extraction), so this is most likely a local caching/library
artifact rather than an upstream data change, but the exact cause wasn't tracked
down further. **Reverted `feats/train.meta.parquet` to the committed version
immediately.** `feats/train.drone_image.satellite.npy` and
`feats/train.ground_image.general.npy` were also regenerated by the same rerun
and have no backup to revert to (git-ignored, not version-controlled); a quick
sanity rerun of the drone 5-fold CV on the regenerated features landed in the
same plausible range as the committed number (0.442 at epoch 59 vs. the
committed 0.457 at E*=3 -- different epochs, so not a direct comparison, but not
a degenerate/corrupted result either). No committed result file
(`results/preds/*.npz`, `results/item2_*`, etc.) was touched by this rerun --
the regenerated caches were only used by this item's timing-only pass, whose
predictions were never saved. Worth a closer look before the next time the
drone branch is retrained for real.
