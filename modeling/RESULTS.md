# Modeling experiments — DINOv3 frozen-feature probe for 8-species multilabel

Frozen-backbone probing of `mpg-ranch/multimodal-survey`: given a robot ground photo
(`ground_image`) and its paired nadir drone crop (`drone_image`), predict which of 8 target
species are present (multilabel). All numbers below are **5-fold leg-grouped cross-validation
on `train`** (`make_folds.py` / `folds_leg5.json`) — folds are grouped by `site/strip/leg` so
both capture events of a leg always land in the same fold, and only odd (survey) legs are used
(6,089 of 6,480 train frames; even legs are turns). The official held-out `test` split (outermost
legs) was **not** extracted at the winning config — see "Not done" below.

Backbones: `facebook/dinov3-vitb16-pretrain-lvd1689m` (768-d, "general", used on `ground_image`)
and `facebook/dinov3-vitl16-pretrain-sat493m` (1024-d, "satellite", used on `drone_image`).

## Best result

**TopKHead**: per-patch linear logits on a dense 64×48 ground-image patch grid, mean of the
**top-1** patch per species, plus a per-frame offset from the drone/satellite CLS token.
3-seed ensemble, 5-fold leg CV.

| metric | value |
|---|---|
| macro AP | **0.766** |
| macro F1 @0.5 | **0.714** |
| per-fold macro F1 | 0.687 ± 0.016 |

Best exact hyperparameters (`run_topk_curves.py`, `k=1`):
- ground image resized to **1024×768** (square-distorted from native 1280×720), patch grid
  64×48 = 3,072 patches, `facebook/dinov3-vitb16-pretrain-lvd1689m`, patch tokens only
  (CLS + 4 register tokens dropped)
- drone image: CLS token only, `facebook/dinov3-vitl16-pretrain-sat493m`, **native resolution**
  (312 px rounded to 320, the nearest ViT/16-compatible size — resolution is irrelevant for
  this branch, see below)
- head: `TopKHead(k=1)` — `nn.Linear(768, 8, bias=False)` per-patch → `topk(1).mean(1)` +
  `nn.Linear(1024, 8)` on the drone CLS
- standardization: per-fold, train-stats only; ground grid standardized on the **per-frame
  patch mean** (an affine map, so it commutes correctly with any k)
- loss: `BCEWithLogitsLoss(pos_weight=(1-p)/p, clamped to 20)` per class
- optimizer: AdamW, `lr=1e-2`, `weight_decay=1e-2`, cosine annealing over 60 epochs
- batch size 64, early stopping on validation macro-F1, patience 10
- 3 seeds {0,1,2}, ensembled by averaging probabilities

### Per-class AP at k=1 (mean over seeds)

| species | n positive (train pool) | AP |
|---|---|---|
| Lupinus sericeus | 1248 | 0.919 |
| Balsamorhiza sagittata | 602 | 0.906 |
| Poa bulbosa | 3426 | 0.902 |
| Sisymbrium altissimum | 715 | 0.763 |
| Bromus tectorum | 418 | 0.750 |
| Gaillardia aristata | 238 | 0.829 |
| Achillea millefolium | 202 | 0.665 |
| Euphorbia virgata | 110 | 0.342 |

Euphorbia virgata is the persistent weak point — the rarest class (110 positives; one CV fold
holds only 3), unmoved by resolution or pooling changes on their own. See "What didn't move it."

## Experiment log, in order run

Each step's script and output are in this directory; see `results/` for the numeric CSVs.

| # | experiment | script(s) | key finding |
|---|---|---|---|
| 1 | CLS-only baseline, sklearn LogisticRegression, held-out test | `extract_features.py`, `fit_probe.py` | `results/probe_metrics.{csv,md}` — ground beats drone; drone-satellite domain-match hypothesis failed |
| 2 | Leg-fold CV scaffolding | `make_folds.py`, `fold_masks.py` | `folds_leg5.json` — grouped, multilabel-stratified 5-fold split; buffered variant costs ~30% of train |
| 3 | Torch CV harness (linear / MLP heads) | `assemble.py`, `head_cv.py`, `run_probe_cv.py`, `probe_lib.py` | baseline concat/MLP: macro AP 0.635, F1 0.610 |
| 4 | CLS vs CLS+mean-patch pooling | `run_pooling_compare.py` | wash on the headline number; free win on grasses |
| 5 | Learning-rate sweep | `run_lr_grid.py` | confirmed `lr=1e-2` (linear) / `1e-3` (MLP) near-optimal |
| 6 | **Resolution sweep 224→512→768** | `extract_concat_512.py`, `extract_concat_768.py`, `extract_drone_native.py`, `run_res_compare.py` | **+0.08 macro AP.** Small forbs (Gaillardia, Achillea) recovered; drone resolution proven irrelevant |
| 7 | Drone sub-crop ablation | `extract_drone_views.py`, `run_drone_views.py` | negative — every sub-crop lost to the full 4×4 m crop; drone works via whole-crop context, not local detail |
| 8 | MLP regularization + ensembling | `run_mlp_tune.py` | dropout 0.2→0.4 is the lever; linear+MLP ensemble beats either alone (AP 0.738) |
| 9 | Horizontal-flip augmentation | `extract_hflip.py`, `run_hflip.py` | marginal (+0.004 AP) — frozen DINOv3 features barely move under flip |
| 10 | Dense patch grid extraction | `extract_grid.py` | `feats/train.ground_image.general.grid1024x768.npy` (not committed — 30.6 GB, regenerate from this script) |
| 11 | **Top-k patch pooling — biggest lever** | `run_topk.py` (naive, thrashed — see note), `run_topk_mil.py`, `run_topk_curves.py` | **+0.10 macro AP** from k=3072 (mean) → k=1. Perfectly monotonic in k. `results/topk_curves.csv`, `topk_curves_perseed.csv` |

## What moved the needle, ranked

1. **Top-k patch pooling** (mean-pool → top-1): **+0.03 to +0.10 macro AP** depending on baseline. The dominant finding — small-species signal is concentrated in a handful of patches; mean-pooling 3,072 of them drowns it.
2. **Ground image resolution 224→768px**: **+0.05 to +0.10 macro AP**. Small forbs are sub-patch at 224px.
3. **Linear + MLP ensemble**: **+0.02 AP**, and the only thing (short of top-k) that moved Euphorbia.
4. **MLP regularization** (dropout 0.4, hidden 256): brought MLP to AP parity with linear while keeping its F1 edge.
5. **Horizontal-flip augmentation**: **+0.004 AP** — small, free, keep it, don't expect more from image-space augmentation on frozen features.

## What didn't move it

- **Drone image resolution / sub-cropping.** Native 320px ≈ 224px ≈ 512px-upscaled. Sub-cropping to plant scale made every species worse (monotonically) — the drone branch contributes whole-crop context, not localizable plant detail.
- **CLS vs CLS+patch-mean pooling** on its own — within noise, except a free lift for grasses (Bromus, Sisymbrium).
- **Euphorbia virgata specifically.** Stuck around AP 0.26–0.40 through every resolution/pooling/augmentation change; it moved most with `both`-pooling+linear (0.36–0.39) and with top-k (0.40), but never decisively. It's a long-tail data problem (110 positives, uneven per-fold), not a feature problem.

## Not done / next steps

- **Held-out `test` split was never extracted at the winning (top-k, 1024×768) configuration.** All numbers here are 5-fold CV on `train`; the reported held-out generalization number is still outstanding.
- Long-tail loss (asymmetric/focal) or per-class thresholds for Euphorbia virgata.
- Combine top-k ground pooling with the linear+MLP ensemble and/or CLS-token concatenation (not yet tried together).
- `fit_predict`'s early stopping selects the checkpoint on the same fold it's scored on — CV numbers here are consistent for comparing configs but mildly optimistic as an absolute estimate.
- Buffered leg-fold CV (drops legs within 5 m of the held-out leg), leave-one-strip-out CV, and a fixed site-2 validation split were scaffolded (`fold_masks.py`) but not carried through the full experiment matrix.

## Reproducing

Feature caches (`modeling/feats/`, `modeling/features/`) are **not committed** — they're
large (`.npy`/`.npz` are gitignored repo-wide) and are cheap to regenerate from the extraction
scripts above against the `mpg-ranch/multimodal-survey` HF dataset. Order: run the extraction
script for the config you want, then the matching `run_*.py`. `run_topk_curves.py` is the
current best-config script; it caches per-fold validation grids on the GPU (fp16) to avoid
re-streaming the ~30 GB dense grid memmap 24× per epoch — expect ~2–3 h wall time on a
12 GB GPU (RTX 4070) for the full 8-k × 3-seed × 5-fold sweep.
