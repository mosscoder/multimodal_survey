<p align="center">
  <img src="assets/multmodal_survey_logo.png" alt="multimodal_survey" width="400">
</p>

# Multimodal Survey

This project explores the Unitree Go2 robot as a platform for botanical surveys in a wildlands setting. Our Go2 traversed several kilometers of transects across the grasslands at MPG Ranch and photographed the vegetation at regular intervals. Aerial drone imagery of the same ground was gathered within hours of each walk. The repo holds exactly two things: the field data the robot recorded (`missions/`) and the pipeline that turns it into the paired ground-and-aerial Hugging Face dataset (`multimodal_dataset/`). Modeling work built on this data lives elsewhere and will surface in applied papers.

## Repo layout

```
multimodal_survey/
├── missions/              survey definitions + everything the robot recorded
│   ├── <site>/<strip>/    one folder per strip surveyed (config, runs, labels)
│   ├── inventory.csv      per-run status + quality tracker
│   └── planning/          QGIS project + basemap for laying out strips
├── multimodal_dataset/    builds the Hugging Face dataset, plus the (historical) labeling app
├── dev/                   the pre-reorg field shakedown (site_1_strip_3_test_run/)
└── docs/                  the dataset overview, field walkthrough, and figures
```

### `missions/` — what the robot did

One folder per strip surveyed, named `<site>/<strip>/` (e.g. `site_1/strip_3`).
Each holds:

* `mission.toml` — the run configuration (leg geometry, capture interval, steering).
* `seed_poly.geojson` — the strip's boundary, drawn by hand in QGIS.
* `waypoints.geojson` — the leg corners, generated from the seed polygon by
  `go2-survey make-waypoints` (regenerating this is a desk task, not a field
  task — don't hand-edit it).
* `mission_layout.png` — a preview image of the route.
* `runs/<run_id>/` — one folder per time the robot actually walked the strip (a
  strip is often walked more than once, on different days). Each run holds the
  photos (`captures/*.jpg`, geotagged and bearing-tagged, with a JSON sidecar per
  photo and a `captures.geojson` manifest), the field logs (`main.log`, `gps.log`,
  `imu.log`, `battery.log`), and, once a person has labeled it,
  `labels/species.json`.

`inventory.csv`, at the top of `missions/`, tracks every run's status (complete /
partial / aborted) and basic quality checks. A run only counts as usable once its
`main.log` reaches `MISSION COMPLETE`.

`missions/planning/` holds the QGIS project (`ranch_map.qgz`) and basemap
(`ranch_map.gpkg`) used to lay out strips before they become each mission's
`seed_poly.geojson`.

**Running a survey in the field** — the full procedure (clone onto the dog,
pre-flight, run, push the results) lives in
[`docs/field-walkthrough.md`](docs/field-walkthrough.md).

### `multimodal_dataset/` — building the Hugging Face dataset

A small, self-contained package that reads every completed mission out of
`missions/` and assembles the release: one row per photo, pairing the robot
frame with a nadir drone-orthomosaic crop of the ground ahead of it, plus
location, time, heading, and species labels. Run with `python -m
multimodal_dataset`. The release-facing description of every column lives in
[`docs/dataset-overview.html`](docs/dataset-overview.html).

It also holds the **labeling app**, the local web tool that produced the
shipped labels (labeling is complete; the app stays as provenance). It carries two switchable tasks over the same frames — pick either from
the header dropdowns, alongside the mission switcher, with no relaunch:

* **Species** (multilabel) — for each frame, which of the 8 target species are
  present. The checkboxes are grouped Wildflowers / Weeds and nothing is checked
  by default, so the labels are unbiased human ground truth (the model's own
  predictions are never shown). Hot-keys: `1`–`4` wildflowers, `A`/`S`/`D`/`F`
  weeds. Saved to `labels/species.json`.
* **Image quality** (single-select) — one ordinal judgement per frame: how much
  of the image is degraded by visual artifacts (motion blur, smear, glare,
  compression), in quartile bands `0–25` / `25–50` / `50–75` / `75–100`% of the
  frame affected, stored behind the scenes as `1`–`4`. Hot-keys `1`–`4` pick the
  band. Saved to `labels/quality.json`.

Each task keeps its own labels file, cursor, and progress; both autosave and are
resumable. Navigation is shared: `→`/`Space`/`Enter` confirm + next, `←` prev,
`c` copy the previous frame, `u` jump to the next unreviewed frame. The task and
the mission are both chosen from the header dropdowns in-app — never on the CLI;
`--mission`/`--run` only pick which mission to land on first.

```bash
python -m multimodal_dataset.labeling label    # opens on the Species task; switch task + mission from the header
```

## Roadmap

- [x] **Review the robot mission data for completeness.** Every run reached
  `MISSION COMPLETE` with its captures, sidecars, and logs intact (status
  tracked in `missions/inventory.csv`).
- [x] **Label the imagery.** All 8,287 frames hand-labeled for the 8 target
  species (`labels/species.json`) and image quality (`labels/quality.json`).
- [x] **Build and push the paired dataset.** One row per robot frame with its
  4 m nadir drone crop, 34 columns, real train/test splits. Live as a private
  Hugging Face dataset at `mpg-ranch/multimodal-survey`, with every release
  invariant verified by `python -m multimodal_dataset verify`.
- [ ] **Finish the release metadata.** Fill the remaining `CITATION.cff`
  author names and backfill the 5 missing `missions/inventory.csv` rows.
- [ ] **Take the dataset public.** Gated on a coordinate-precision review and
  a dataset-viewer check (the viewer activates once the repo is public).
- [ ] **Add the multispectral crop** (`drone_image_ms`) from the Mavic 3M
  orthomosaics as a follow-up column: bands Red, Green, NIR, RedEdge at
  2.24 cm GSD, values as scaled digital numbers rather than reflectance.

