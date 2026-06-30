# Follow-up ablations

Inference-time ablations on the **deployed** MIL head (`checkpoints/mil.pt`) — **no retraining**.
Both change only how robot-frame patches are scored, then re-derive per-species thresholds on
robot-**val** (the scoring distribution shifts, so the frozen thresholds no longer apply) and
evaluate on the held-out **test** split. Each reports all-9 and well-supported (support ≥ 10)
macro-F1, picks a winner by val all-9, and flags well-supported disagreement. Findings are
written up in `../RESEARCH_LOG.md`.

- `multiscale_infer.py` — fuse the native tiling with downsampled passes so each 16 px patch
  spans a bigger chunk of plant (any-patch max across scales).
- `superpatch_pool.py`  — adaptive-average-pool each tile's token grid into coarser
  "super-patches" before the head.

The control config in each (`scales {1}` / `g=14`) is the canonical native pipeline and must
reproduce the deployed numbers (val all-9 0.636 / test 0.603) — a protocol sanity check.
