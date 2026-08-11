# Getting started

Two workflows live in this repo. Paths are relative to the repo root. The
modeling and iNaturalist-harvest workflows moved out on 2026-08-11; they are
papers-bound and documented where they now live.

## 1. The labeling app (historical)

Labeling is **complete**: all 8,287 frames carry species and quality labels in
each run's `labels/`. The app stays as the provenance of how they were made,
and still runs if labels ever need revisiting. Needs only **Python 3** (standard
library, no install):

```bash
python -m multimodal_dataset.labeling label    # opens on the Species task; switch task + mission from the header
```

The app has **two switchable tasks** over the same frames — pick from the **Task**
dropdown in the header (no relaunch):

* **Species** (multilabel) — which of the 8 target species are present. Nothing
  is checked by default, so the labels are unbiased human ground truth. Writes
  `labels/species.json`.
* **Image quality** (single-select) — press **1–4** for how much of the frame is
  degraded by visual artifacts (blur, smear, glare, compression), quartile bands
  `0–25 / 25–50 / 50–75 / 75–100 %`. Writes `labels/quality.json`.

Both autosave and are resumable. `→`/`Space` confirm + next, `←` back, `c` copy
previous, `u` jump to next unreviewed. Task and mission are chosen in-app.

## 2. Build the Hugging Face dataset

The `multimodal_dataset/` package assembles the release from `missions/` and the
drone orthomosaics on GCS. The release-facing description of every column is
[`dataset-overview.html`](dataset-overview.html).

```bash
python -m multimodal_dataset build     # assemble to multimodal_dataset/out/
python -m multimodal_dataset verify    # assert the release invariants
python -m multimodal_dataset push      # upload to the private HF repo
```

Pushing needs a Hugging Face **write** token in `.env` (`HF_TOKEN=…`); create
one at <https://huggingface.co/settings/tokens>. Build and verify need no
credentials (the orthomosaics are public).

## 3. The manuscript (Overleaf submodule)

`manuscript/` is the Overleaf project as a git submodule. Overleaf is the
editor and viewer; the local checkout exists to add figures and other build
products. The loop:

```bash
git -C manuscript pull --rebase     # take Overleaf's latest edits first
cp <new figure> manuscript/figures/
git -C manuscript add figures && git -C manuscript commit -m "figures: ..."
git -C manuscript push              # appears in Overleaf immediately
git add manuscript && git commit -m "manuscript: advance pointer"   # parent repo tracks the state
```

Always pull before pushing: Overleaf commits every editor save, so the remote
moves on its own. Cloning fresh needs `git submodule update --init` and an
Overleaf git token (username `git`).
