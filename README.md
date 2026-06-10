# multimodal_survey

Mission definitions for Go2 robot line surveys at MPG Ranch. Each mission
directory under `dev/` holds everything the dog needs to walk a survey
(`mission.toml` + `waypoints.geojson`) plus the planning artifacts
(`seed_poly.geojson`, `mission_layout.png`). **Survey outputs (logs and
captures) nest under each mission's `runs/` folder on the datastick —
git-ignored, so they stay out of the repo history.**

---

## Field walkthrough: running `site_1_strip_3` on the dog

### TL;DR — routine field day (Jetson)

```
git -C /media/mpg-robodog/KINGSTON/multimodal_survey pull
go2-survey run /media/mpg-robodog/KINGSTON/multimodal_survey/dev/site_1_strip_3
```

Both commands work from any directory — no need to `cd` into either repo.
You only touch the `mpg-ai-edge` (go2-survey) checkout during one-time
setup, or when Kyle announces a software update.

### One-time setup (per datastick)

1. Insert the KINGSTON datastick into the dog's Jetson and confirm it
   mounted:

   ```
   ls /media/mpg-robodog/KINGSTON
   ```

2. Clone this repo onto the stick:

   ```
   cd /media/mpg-robodog/KINGSTON
   git clone https://github.com/mosscoder/multimodal_survey.git
   ```

3. Make sure the `go2-survey` software on the Jetson is current — it needs
   the **`refactor` branch of `mpg-ai-edge`, v0.29.0 or later** (the
   self-calibrating line-survey walk this mission uses). In the Jetson's
   `mpg-ai-edge` checkout:

   ```
   git fetch && git checkout refactor && git pull
   go2-survey list   # sanity check: command runs; the run banner shows the version
   ```

### Before every field day

```
cd /media/mpg-robodog/KINGSTON/multimodal_survey && git pull
```

Check free space on the stick — budget **~1 GB per run** (~300 JPEGs plus
logs):

```
df -h /media/mpg-robodog/KINGSTON
```

### Pre-flight checklist (each run)

- Dog powered and connected to the field hotspot; Jetson on the same
  network.
- GPS receiver plugged into the Jetson (shows up as `/dev/ttyACM0`).
- **Place the dog about 5 m outside the SOUTH corner of the strip, roughly
  facing it.** The route starts at the southern corner (`wp_001`) and the
  IMU self-calibrates from the approach walk, so the placement matters.
- `dev/site_1_strip_3/mission_layout.png` shows the route if you want to
  orient yourself: 4 parallel legs of ~150 m, snaking south → north.

### Run it

```
go2-survey run /media/mpg-robodog/KINGSTON/multimodal_survey/dev/site_1_strip_3
```

What you'll see, in order:

1. **GPS + NTRIP connect**, then an RTK stabilization dwell (up to 60 s —
   exits early once the fix is solid).
2. **Robot discovery + connect**, video stream on.
3. The dog walks to the south corner, then drives the 4 legs at 1 m/s,
   taking a geotagged photo every 2 m (~300 captures, ~600 m of track).
   Expect **15–20 minutes** end to end.
4. At mission end the photo compass bearings are recomputed from the RTK
   track and the capture manifest is written — automatic, even if the run
   was aborted.

### Outputs

Everything from one run lands in one timestamped folder inside the
mission directory (git-ignored — `git pull` stays clean regardless):

```
/media/mpg-robodog/KINGSTON/multimodal_survey/dev/site_1_strip_3/runs/site_1_strip_3_<timestamp>/
├── main.log              ← mission narrative (read this first)
├── gps.log               ← dense RTK/NTRIP telemetry
├── imu.log               ← IMU stream
└── captures/
    ├── legNN_mNNN_*.jpg  ← survey photos (EXIF geotag + true bearing)
    ├── legNN_mNNN_*.json ← per-photo sidecars
    └── captures.geojson  ← manifest of all captures
```

### If something goes wrong

- **Ctrl-C** aborts the mission; the dog stops, and all logs/captures up to
  that point are kept and finalized.
- **"GPS fix timeout"** — antenna needs open sky and the NTRIP caster needs
  cell coverage; reposition and rerun.
- **Robot not found** — confirm the dog and Jetson share the hotspot, then
  `go2-survey discover-ip` to diagnose.
- For anything else, send Kyle the whole run folder from the mission's
  `runs/` directory (it is self-contained).

---

## Repo layout

```
multimodal_survey/
├── README.md                      ← this walkthrough
├── dev/
│   └── site_1_strip_3/            ← one directory per survey mission
│       ├── mission.toml           ← run config
│       ├── seed_poly.geojson      ← survey area drawn in QGIS (EPSG:6514)
│       ├── waypoints.geojson      ← generated leg corners (do not hand-edit)
│       ├── mission_layout.png     ← route preview
│       └── runs/                  ← created by runs (git-ignored field data)
│           └── site_1_strip_3_<timestamp>/
└── planning/                      ← QGIS project + ranch basemap
```

Waypoints are generated from the seed polygon with `go2-survey
make-waypoints` (boxified grid, south start corner) — regenerating is a
dev task, not a field task.
