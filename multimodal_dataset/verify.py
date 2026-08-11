"""Release gates: assert the built dataset holds every published invariant.

    python -m multimodal_dataset verify

Loads the built dataset from ``out/`` and recomputes, from the shipped columns
themselves, every number the release documentation states: the partition, the
zero train/test overlap, the pair-join structure, the offset and separation
statistics, the per-split species table, and a random spot-check back against
the source sidecars. Any mismatch fails the run.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .build import MISSIONS_ROOT, OUT_DIR
from .crops import S, antenna_en, footprint_corners_en
from .schema import COLUMNS

# The published numbers (docs/dataset-overview.html + the plan). Exact where
# exact, (value, tolerance) where the page states a rounded figure.
PUBLISHED = {
    "counts": {"train": 6480, "test": 1613, "train_tier_2": 194},
    "pair_nonnull": 8134, "pair_null": 153,
    "pair_distance_median": (0.101, 0.001), "pair_distance_p95": (0.326, 0.002),
    "pair_distance_max": (2.12, 0.005),
    "offset_abs_median": (1.16, 0.005), "offset_abs_max": (20.29, 0.01),
    "offset_tod_median": (1.16, 0.005), "offset_tod_max": (4.35, 0.01),
    "restart_frames": 31,
    "clipped_frames": 33, "coverage_min": (0.078, 0.001),
    "tier2_touched_test_crops": 213,
    "tier2_overlap_median": (0.045, 0.002), "tier2_overlap_max": (0.626, 0.002),
    "artifact_band4": 1304,
    "species_per_split": {          # train, test, tier 2
        "Lupinus sericeus": (1331, 209, 39),
        "Balsamorhiza sagittata": (618, 245, 24),
        "Gaillardia aristata": (238, 70, 0),
        "Achillea millefolium": (216, 41, 1),
        "Euphorbia virgata": (112, 49, 11),
        "Sisymbrium altissimum": (728, 90, 15),
        "Bromus tectorum": (444, 140, 13),
        "Poa bulbosa": (3664, 1090, 105),
    },
}

_failures: list[str] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)


def _close(actual: float, spec: tuple[float, float]) -> bool:
    value, tol = spec
    return abs(actual - value) <= tol


def _quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    i = q * (len(xs) - 1)
    lo = int(math.floor(i))
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


def verify(out_dir: str | Path = OUT_DIR, spot_checks: int = 25) -> None:
    from datasets import load_from_disk
    from shapely import STRtree
    from shapely.geometry import Polygon

    ds = load_from_disk(str(out_dir))

    print("[verify] partition and schema")
    counts = {split: ds[split].num_rows for split in ds}
    _check("split counts", counts == PUBLISHED["counts"], str(counts))
    _check("total rows", sum(counts.values()) == 8287)
    for split in ds:
        _check(f"{split}: 34 columns in schema order", ds[split].column_names == COLUMNS)

    plain = {split: ds[split].remove_columns(["ground_image", "drone_image"])[:]
             for split in ds}

    def col(name, splits=("train", "test", "train_tier_2")):
        return [v for s in splits for v in plain[s][name]]

    print("[verify] geometry: zero train/test crop overlap")
    def polys(split):
        p = plain[split]
        return [Polygon(footprint_corners_en(*antenna_en(lat, lon), brg))
                for lat, lon, brg in zip(p["latitude"], p["longitude"], p["bearing_deg"])]
    test_polys, train_polys = polys("test"), polys("train")
    tree = STRtree(test_polys)
    overlapping = sum(
        1 for poly in train_polys
        if any(test_polys[i].intersection(poly).area > 0 for i in tree.query(poly)))
    _check("no train crop shares ground with any test crop", overlapping == 0,
           f"{overlapping} overlapping")
    tier2_polys = polys("train_tier_2")
    touched = {int(i) for poly in tier2_polys for i in tree.query(poly)
               if test_polys[int(i)].intersection(poly).area > 0}
    _check("tier-2 touches the published number of test crops",
           len(touched) == PUBLISHED["tier2_touched_test_crops"], str(len(touched)))
    t2 = plain["train_tier_2"]["tier2_overlap_fraction"]
    _check("tier-2 overlap median", _close(_quantile(t2, 0.5), PUBLISHED["tier2_overlap_median"]))
    _check("tier-2 overlap max", _close(max(t2), PUBLISHED["tier2_overlap_max"]))

    print("[verify] pairs")
    pair_ids = col("pair_frame_id")
    nonnull = [p for p in pair_ids if p is not None]
    _check("pair non-null count", len(nonnull) == PUBLISHED["pair_nonnull"], str(len(nonnull)))
    _check("pair null count", len(pair_ids) - len(nonnull) == PUBLISHED["pair_null"])
    back = dict(zip(col("frame_id"), pair_ids))
    _check("pairing is symmetric",
           all(back.get(p) == f for f, p in back.items() if p is not None))
    dists = [d for d in col("pair_distance_m") if d is not None]
    _check("pair distance median", _close(_quantile(dists, 0.5), PUBLISHED["pair_distance_median"]))
    _check("pair distance p95", _close(_quantile(dists, 0.95), PUBLISHED["pair_distance_p95"]))
    _check("pair distance max", _close(max(dists), PUBLISHED["pair_distance_max"]))

    print("[verify] timing")
    offs = col("capture_offset_absolute")
    _check("offset median", _close(_quantile([abs(o) for o in offs], 0.5),
                                   PUBLISHED["offset_abs_median"]))
    _check("offset max", _close(max(abs(o) for o in offs), PUBLISHED["offset_abs_max"]))
    tods = [abs(t) for t in col("capture_offset_time_of_day")]
    _check("time-of-day offset median", _close(_quantile(tods, 0.5), PUBLISHED["offset_tod_median"]))
    _check("time-of-day offset max", _close(max(tods), PUBLISHED["offset_tod_max"]))
    restarts = [r for r in col("mission_restart_utc") if r is not None]
    _check("restart frames", len(restarts) == PUBLISHED["restart_frames"], str(len(restarts)))

    print("[verify] coverage and labels")
    cov = col("drone_coverage")
    clipped = [c for c in cov if c < 1.0]
    _check("clipped frames", len(clipped) == PUBLISHED["clipped_frames"], str(len(clipped)))
    _check("worst coverage", _close(min(cov), PUBLISHED["coverage_min"]))
    _check("test coverage all 1.0", all(c == 1.0 for c in plain["test"]["drone_coverage"]))
    band4 = sum(1 for a in col("artifact_level") if a == 3)
    _check("artifact band-4 count", band4 == PUBLISHED["artifact_band4"], str(band4))
    names = ds["train"].features["species"].feature.names
    for sp, expected in PUBLISHED["species_per_split"].items():
        sid = names.index(sp)
        got = tuple(sum(1 for row in plain[s]["species"] if sid in row)
                    for s in ("train", "test", "train_tier_2"))
        _check(f"species counts: {sp}", got == expected, str(got))

    print("[verify] images (full decode pass)")
    bad = 0
    for split in ds:
        for row in ds[split]:
            if row["drone_image"].size != (312, 312) or row["ground_image"].size != (1280, 720):
                bad += 1
    _check("every drone crop 312x312 and every ground frame 1280x720", bad == 0, f"{bad} bad")

    print(f"[verify] spot-check {spot_checks} rows against source sidecars")
    rng = random.Random(0)
    rows = [(s, i) for s in ds for i in range(ds[s].num_rows)]
    mismatches = []
    for split, i in rng.sample(rows, spot_checks):
        row = {k: plain[split][k][i] for k in
               ("frame_id", "site", "strip", "run_id", "latitude", "longitude",
                "bearing_deg", "ground_captured_at", "pdop", "leg")}
        sidecar = (Path(MISSIONS_ROOT) / row["site"] / row["strip"] / "runs" /
                   row["run_id"] / "captures" / f"{row['frame_id']}.json")
        d = json.loads(sidecar.read_text())
        ok = (d["position"]["latitude"] == row["latitude"]
              and d["position"]["longitude"] == row["longitude"]
              and d["heading"]["course_degrees_true"] == row["bearing_deg"]
              and d["captured_at_utc"] == row["ground_captured_at"]
              and d["position"]["pdop"] == row["pdop"]
              and d["extra"]["line_survey"]["leg"] == row["leg"])
        if not ok:
            mismatches.append(row["frame_id"])
    _check("sidecar spot-checks", not mismatches, ", ".join(mismatches[:3]))

    if _failures:
        raise SystemExit(f"[verify] FAILED: {len(_failures)} gate(s): {_failures}")
    print("[verify] all gates passed")
