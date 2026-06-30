## Overview

Predict presence/absence of target plant species in imagery from a quadrupedal
robot, training on crowd-sourced iNaturalist imagery ONLY. An image-level labeled
robot dataset exists, but it is used only for validation and testing — never for
training.

## Targets

All 9 focal species are trained and scored: Lupinus sericeus, Poa bulbosa,
Tragopogon dubius, Gaillardia aristata, Balsamorhiza sagittata, Bromus tectorum,
Achillea millefolium, Sisymbrium altissimum, Thinopyrum intermedium.
Visualizations paint only the three forbs: Lupinus sericeus, Gaillardia aristata,
Balsamorhiza sagittata.

These same three forbs — **lupine** (Lupinus sericeus), **arrowleaf** balsamroot (Balsamorhiza
sagittata), and **blanketflower** (Gaillardia aristata) — are the FAVORED HEADLINE species. All 9
focal species' performance is of genuine interest and is always reported, but results, write-ups,
and demos should foreground these three: they are the demo's visual story (the painted forbs), so a
gain/loss on lupine, arrowleaf, or blanketflower is a headline, weighted above the other six.

## Approach

Turn image-level (sparse) iNaturalist labels into dense per-patch predictions with
multiple-instance learning (MIL). Sweep learning rate, hidden-layer width, and
cropping strategy. MIL only — the non-MIL (CLS-token) classifier is retired.
Winnowing (pre-selecting field-like iNat images) is abandoned: it raised image
scores but made the head fire on vegetation context instead of the plant.

Training uses the FULL iNaturalist dataset per species — both HF train + test splits
(~2500 obs/species) — since the robot frames are the only held-out evaluation.

## Caching — reuse, never regenerate when avoidable

Backbone feature extraction is by far the expensive step; ALWAYS reuse cached features instead of
re-encoding through DINOv3 whenever possible. Caches live in `cache/` (gitignored) behind
`D.cached(path, sig, build)`, which loads on a signature match and rebuilds ONLY on a miss:

- **iNat features** (`specview_<species>_<view>_n<count>.pt`) are keyed by (species, view, count) and
  are INDEPENDENT of LR, head width, r, loss, label scheme, and the robot split. Extract ONCE at the
  largest count (`n_extract=SIZE_MAX`) and SUBSET down with `n_per` (nested fixed-shuffle prefix) — so
  every sweep/ablation that varies a head-only knob reuses the same files. A 2048-img run reuses the
  size sweep's 2048 caches verbatim.
- **Robot patches** (`robot_patches.pt`) are keyed by frame geometry + frame-set hash — reused across
  val/test scoring and all follow-ups; the 50/50 split just slices this cached tensor.

A `cached: <file>` log line is a HIT (load from disk); a cache MISS instead prints the build line
`<species> <view> (N imgs)`. If you see builds where a hit was expected, FIX THE SIGNATURE — don't
pay to re-extract. Before launching any extraction, check whether an existing cache already covers it.

## Evaluation

- Split robot frames 50/50, stratified, fixed seed: 50% validation, 50% test
  (held out, scored once). The 50/50 split (was 80/20) stabilizes the held-out test for
  rare species, whose 80/20 test support (2–6 frames) made the all-9 macro unmeasurable.
- Each frame is cut into 24 disjoint NATIVE-resolution 224x224 tiles (no
  downscaling) and encoded to a per-patch score per species.
- Any-patch rule: a species is predicted PRESENT in a frame if even ONE patch
  scores at/above that species' threshold.
  - True positive: predicted present AND in the image's labels.
  - False positive (hard fail): predicted present but NOT in the labels — one
    stray patch firing for an absent species is a failure.
  - False negative: in the labels but no patch fires.
- Metric: per-species precision/recall/F1. Hyperparameter SELECTION uses `sel7` = the macro over
  the 7 well-sampled species; Achillea millefolium (yarrow) + Sisymbrium altissimum (tumble
  mustard) are EXCLUDED from selection (too few robot frames -> noise) and reported only on the
  final held-out test, alongside the full all-9 macro and per-species table.

## Thresholds — one per species, set ONCE, no exceptions

Each species has exactly ONE probability threshold, set during model training:
every epoch, for each species, pick the cutoff that maximizes that species' F1 on
the validation set; keep the best epoch's cutoffs and FREEZE them into the
checkpoint. That single frozen threshold is the ONLY threshold ever used — applied
identically to test scoring, inference, and visualization. No Otsu, no fixed
values, no per-image adjustment, no second/post-hoc calibration, ever — the firing
DECISION is the same frozen threshold for test scoring, inference, and visualization.
(The overlay then draws the concave ALPHA-SHAPE of the fired patches + a centroid per
region as a PRESENTATION layer — morphologically smoothed, every fired region drawn down to a
single patch (no deletions) — which does not change the any-patch metric over all 9 species.)

## Findings (rev-5 sweeps of 2026-06-29 — details in RESEARCH_LOG.md)

Settled by isolated, one-axis-at-a-time sweeps. Selection = robot-val **sel7** macro-F1 (the 7
well-sampled species; yarrow + tumble mustard excluded — see Evaluation). Robot val/test 50/50;
**month-balanced, license-clean iNat data, 38 plant species + Sky** (equal-per-month harvest — see the
month-balanced entries in RESEARCH_LOG). rev-5 sweep order: species → **phenology** → hidden → crop →
size → r → lr → final → tiling. **Grad-norm clip = 1.0 locked on every run; tune size 128.**

- **Species:** ALL 38 HF plant species (29 non-targets as hard negatives) **+ a "Sky" negative** (always
  on), scored only on the 9. Broad negatives beat targets-only on sel7 (0.7366 vs 0.7036), as in rev-4.
  Two new small hard negatives added: Helianthella uniflora, Polygonum douglasii.
- **Phenology (NEW):** training images restricted to an observed-month window, drawn EQUALLY per month.
  **May–Sep wins** (sel7 0.7366) over JJA (0.7299) and all-12 (0.7191) — an inverted-U: matching the
  mid-June deployment season helps, and the FULL year HURTS (off-season senescent/dormant imagery is
  noise). This is the payoff of the month-balanced re-harvest.
- **Sky class:** auxiliary hard-negative from FREE iNaturalist **bald-eagle-in-flight** imagery — UL/UR
  native-224 **corner** crops, **SegFormer (ADE20K) gated to ≥10% sky** (UL 761 / UR 768 of 1000 on the
  month-balanced eagles). Teaches "bright sky ≠ lupine." Lives in `inat_dataset/birds/`; never scored (D.N=9).
- **Head:** hidden 64 (clean sel7 peak; 256 close, 0/32/128 dip).
- **Crop:** square whole-plant tile + native zoom; **z=1.0 wins (square + native(1.0))** — REVERSED from
  rev-4's z=0.5. All 3 zooms beat native-only, so it's the **square view** that helps.
- **Data:** 512 imgs/species (May–Sep-capped) — more data wins; convergence epoch shrinks 60→14 with more data.
- **r (LSE):** **2** — lower training temperature (bag pooling nearer the mean) wins again, but the curve is
  **U-shaped** (r=64 runner-up), not monotone. The all9-vs-sel7 split suggests species differ in optimal r
  (rare species favor high r) — a per-class-r head is a rev-6 candidate. (Inference is still any-patch MAX.)
- **LR:** **5e-5** — REVERSED from rev-4's 1e-5; clip 1.0 (now locked) stabilizes the higher rate. 5e-6
  under-fits; hot rates (5e-4 / 1e-3) overfit at epoch 0–1.
- **Tiling (deployment):** **macro+patch** (2-pass stride-112 + 8px offset) — best val sel7 AND monotone test
  gains (base 0.6968 → macro 0.7188 → macro+patch 0.7437). Per-tiling thresholds re-derived on val; frozen
  into mil.pt. See Visualizations.
- Winnowing abandoned.

Deployed checkpoint (`checkpoints/mil.pt`, 39-output head scored on the 9; **tiling macro+patch**, re-derived
thresholds; rev-4 preserved as `mil_rev4.pt`): val sel7 0.788 · **TEST sel7 0.744 / all-9 0.734** (50/50,
n=266) — **best in the program; beats rev-4 deployed (0.732 / 0.679) on both, all9 +0.055.** Per-species test:
Lupinus 0.94 (**P 0.97**), Thinopyrum 1.00, Balsamorhiza 0.77, Poa 0.70, Bromus 0.67, Gaillardia 0.58,
Tragopogon 0.56. **The rev-4 Achillea over-fire is FIXED** (P 0.23 → P 1.00 / F 0.80 — drives the all9 gain).
Open issues: **blanketflower (Gaillardia) recall REGRESSED** (rev-4 0.72 → 0.58, R 0.49 — the one headline
drop); Tragopogon recall-limited; larger val→test gap on low-support species.

## Visualizations

Run inference on the full robot dataset; for the three forbs, draw the concave ALPHA-SHAPE of the
fired patches (cv2: morphological close to bridge small gaps + concave contour that hugs the region)
with a centroid marker per region (a candidate map-anchor). The firing decision is the frozen
threshold; the alpha-shape is a presentation layer (does not change the any-patch metric). Per-region
centroids + areas are written to `inference_viz/summary.json` for downstream map placement.

**Do NOT delete small regions as "clutter."** An earlier size filter (drop regions < ~9 patches)
deleted small *real* detections (sparse plants that fire on a few patches), not just stray FPs — a size
filter cannot tell them apart — which broke painted-iff-fired faithfulness and hid real signal (e.g.
single-patch Fβ recall gains), actively misleading interpretation. Keep the overlay faithful: every
fired patch is drawn (`MIN_PATCHES=1`). If decluttering is ever needed, gate by CONFIDENCE, not size,
and make it explicit — never silently drop small detections.
