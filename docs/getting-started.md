# Getting started

Below are quick starts for three workflows. All paths are relative to the `multimodal_survey` repo root. The
image labeler (1) is pure-Python and needs nothing installed; the dataset rebuild
(2) and the MIL experiments (3) each stand up their own environment, described in
their own sections.

## 1. Label images by quality

Needs only **Python 3** (the labeler is standard-library only — no install). Run it
as a module from the repo root:

```bash
python -m multimodal_dataset.labeling label    # opens on the Species task; switch task + mission from the header
```

The app has **two switchable tasks** over the same frames — pick from the **Task**
dropdown in the header (no relaunch). Choose **Image quality**: for each frame, press
**1–4** for how much of the image is degraded by visual artifacts (blur, smear,
glare, compression) — quartile bands `0–25 / 25–50 / 50–75 / 75–100 %`. Single-select,
autosaves, resumable. `→`/`Space` confirm + next, `←` back, `c` copy previous, `u`
jump to next unreviewed. Quality labels write to each run's `labels/quality.json`,
separate from species labels so the two tasks never collide. Switch missions from the
**Mission** dropdown. Both task and mission are chosen in-app — never on the CLI.

## 2. Rebuild the iNaturalist dataset

**What pixelflora is (one-time setup).** pixelflora is a standalone tool in its own
repo that turns a species list into a Hugging Face image dataset harvested from
iNaturalist. `multimodal_survey` depends on it, but it isn't part of this repo — you
install it separately into its own conda env:

```bash
# get it and build an isolated env (mamba/conda)
git clone https://github.com/mosscoder/pixelflora.git
cd pixelflora
mamba create -n pixelflora -c conda-forge python=3.12 pip
mamba run -n pixelflora pip install -e .          # editable install; the `pixelflora` CLI lands on PATH
```

Then edit `pixelflora/config.toml` (non-secret settings): set `user_agent`/`contact_email`
to **your** email so iNaturalist can reach you — required etiquette for their API, and
it avoids throttling. Everything is a **dry run / private by default**; you only need a
Hugging Face token (`export HF_TOKEN=…`) if you flip `push = true` to publish. Create a
token at <https://huggingface.co/settings/tokens> (a read token is enough to download
models; you need a write token to push a dataset). The
pipeline is `resolve → harvest → filter → download → assemble → split → publish`, and
written records are kept apart from image files so you can refilter/redivide without
re-downloading.

**Rebuilding.** The species list lives in `multimodal_survey/inat_dataset/plants/request.toml`
(currently **50 species**, each with an iNat-verified name + observation count). Edit it,
then run pixelflora against it (from its env):

```bash
cd inat_dataset/plants
mamba run -n pixelflora pixelflora --config ~/pixelflora/config.toml run request.toml
```

It **resumes**: already-downloaded species are reused (only new/changed ones fetched),
it rebuilds one HF config per species with its own train/test split, and publish stays a
dry run. Run `… resolve request.toml` first if you just want to confirm new names map to
the intended taxa. Note: `out/`, manifests, and `.pixelflora_cache/` are gitignored (the
provenance artifacts — `summary.json`, `README.md`, `CITATION.cff`, `bibliography.bib`,
`license_report.json` — are kept); don't commit images or manifests.

**The "Sky" negative class (birds).** There's a second request,
`inat_dataset/birds/request.toml`, that harvests bald-eagle-in-flight photos
(*Haliaeetus leucocephalus*) — very frequently shot against open sky/cloud — as a
single auxiliary hard-negative class. At train time the detector crops the
upper-left/upper-right 224-px corners of each image (the bird is centered, so the
corners are sky/cloud), teaching the head that bright sky/cloud is **not** a target
plant. It's the same pixelflora pipeline — just point it at the birds request:

```bash
cd inat_dataset/birds
mamba run -n pixelflora pixelflora --config ~/pixelflora/config.toml run request.toml
```

## 3. Initial work — the strip 5 MIL experiments

This work exists to **test multiple-instance-learning (MIL) methods on a single strip
(strip 5) in site 1** — a controlled, one-strip testbed for the robot model before
scaling to more strips/sites. It lives in **`dev/site_1_strip_5_MIL_exploration/`**: a
frozen DINOv3 backbone feeds a small MIL head, and the experiments compare MIL variants
(losses, pooling, tiling, inference tricks) against that one strip's ground truth. It has
heavier dependencies (PyTorch, transformers/DINOv3) — see the directory's own docs for its
environment.

> **Model access (required).** The backbone is `facebook/dinov3-vitb16-pretrain-lvd1689m`,
> a **gated** model on Hugging Face — you must **request access** on its model page
> (<https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>) and be approved
> before you can download it. Then authenticate so `transformers` can pull the weights:
> `huggingface-cli login` (or `export HF_TOKEN=…`) with a token that has access. Until
> access is granted the first `from_pretrained` call fails with a 401/403.

Start with **`RESEARCH_LOG.md`** and **`CLAUDE.md`** for the current state
(rev-5 is deployed), then:

- `sweeps/sweep_mil.py` — the sweep harness (config + `sweep_state.json`)
- `core/` — backbone, data, tiling, geometry, render
- `training/train_mil.py`, `inference_viz/infer_mil.py`

Ground-truth eval is the strip 5 human labels (50/50 val/test split — **never training
data**). Paths anchor to the package dir, so it runs from anywhere in the env. Large
artifacts (`cache/`, `checkpoints/`, `inference_viz/overlays/`) are already gitignored —
keep them out of commits.