# Free-data → robot species detection (MIL)

Predict presence/absence of target plant species in robot imagery, training on
crowd-sourced iNaturalist data only. **`CLAUDE.md` is the spec** (targets, the
any-patch eval, the one-threshold-per-species rule) — read it first.

## Layout

```
core/           shared, model-agnostic
  common.py       frozen DINOv3 backbone, patch features, square/zoom/corner crop views, MILHead
  geometry.py     robot-frame -> 24 native 224 tiles -> 45x80 patch grid
  tiling.py       inference tilings: base / macro (stride-112) / macro+patch (8px) -> 45x80 grid
  data.py         targets + Sky negative, robot GT, 50/50 split, dataset routing, cache, macro-F1
  render.py       overlay: concave alpha-shape of fired patches + centroids (cv2)
(labeling)       MOVED -> repo-root multimodal_dataset/labeling (label + review apps)
training/       train_mil.py      MIL training + any-patch eval + frozen thresholds + sky gate
sweeps/         sweep_mil.py      staged species->hidden->crop->size->r->lr sweep -> sweep_state.json
                sweep_tiling.py   final tiling sweep -> freezes tiling+thresholds into mil.pt
inference_viz/  infer_mil.py      full-dataset inference (deployed tiling) + per-patch overlays
cache/          feature caches (gitignored)        checkpoints/  mil.pt (gitignored)
```

iNaturalist data lives upstream in `inat_dataset/{plants,birds}/out/dataset` (one HF config per
species; `pixelflora run <request.toml>` harvests it, license-filtered). The **Sky** hard-negative
class is bald-eagle-in-flight imagery (`birds/`), cropped at the upper corners and SegFormer-gated to
>=10% sky; it is scored only as a negative (the 9 targets are unchanged).

## Tuning (staged, one axis at a time)

`sweep_mil.py --stage {species,hidden,crop,size,r,lr,final}` tunes one knob at a time, each stage
reusing the prior winners (carried in `sweep_state.json`), picking the highest robot-val **sel7**
macro-F1 (the 7 well-sampled species; yarrow + tumble mustard reported only at final). rev-4 run
order: **species -> hidden -> crop -> size -> r -> lr -> final** (LR swept LAST, after the temperature
r; default LR **5e-5** carried through the earlier stages).

- **species** `{targets(9), grasses(12), all(36)}` + the **Sky** negative (always on)
- **hidden** `{0,32,64,128,256}` · **crop** — square + native zoom `z in {0.5,0.75,1.0}`
- **size** — images/species `{64,128,256,512}` (nested) · **r** — LSE temperature `{2..128}`
- **lr** `{5e-6,1e-5,5e-5,1e-4,5e-4,1e-3}` (swept last) · **final** — full run with all winners

Then `sweep_tiling.py` picks the deployment **tiling** (base / macro / macro+patch) on robot frames,
re-deriving thresholds per method, and freezes the winner's tiling + thresholds into `mil.pt`.
Each stage appends a dated entry to `RESEARCH_LOG.md` (objective, positive/negative results).

## Run

Backbone weights are Meta-gated; `HF_TOKEN` is read from `multimodal_survey/.env`.
Scripts put `core/` (and `training/`) on `sys.path` themselves.

```bash
set -a; . /Users/kdoherty/multimodal_survey/.env; set +a
export HF_TOKEN PYTORCH_ENABLE_MPS_FALLBACK=1
ENV="mamba run -n pixelflora python"

$ENV training/train_mil.py        # one full run with the module defaults -> checkpoints/mil.pt
for S in species hidden crop size r lr final; do $ENV sweeps/sweep_mil.py --stage $S; done  # staged tuning
$ENV sweeps/sweep_tiling.py       # pick deployment tiling -> freeze tiling+thresholds into mil.pt
$ENV inference_viz/infer_mil.py   # overlays for all robot frames (deployed tiling + frozen thresholds)
(cd ../.. && $ENV -m multimodal_dataset.labeling label --run <run_id>)  # relabel GT (repo root, port 8765)
```

Robot labels live in the run dir (`missions/.../labels/image_multilabel.json`),
not here. Feature caches and checkpoints regenerate on first run.
