"""Assemble the release from ``missions/`` and the flight orthomosaics.

    python -m multimodal_dataset build

Pipeline: discover labeled runs -> extract rows -> merge labels -> crop
polygons -> cross-event pairs -> drone crops (one flight ortho at a time) ->
split assignment -> three real Hugging Face splits saved to ``out/``.
"""
from __future__ import annotations

from pathlib import Path

from .crops import CACHE, FLIGHTS, Ortho
from .discover import completed_runs
from .extract import add_pairs, add_polygons, extract_run
from .labeling.merge import apply_labels
from .schema import COLUMNS, features
from .splits import SPLITS, assign_splits

_PKG = Path(__file__).resolve().parent
MISSIONS_ROOT = _PKG.parent / "missions"
OUT_DIR = _PKG / "out"
CROPS_DIR = CACHE / "crops"


def labeled_runs(missions_root: str | Path):
    """Completed runs that carry labels, with each strip's capture-event
    ordinal (1-based, by run_id order — the release's flight-pairing rule)."""
    runs = [r for r in completed_runs(missions_root)
            if (r.run_dir / "labels" / "species.json").exists()]
    by_strip: dict[tuple, list] = {}
    for run in sorted(runs, key=lambda r: r.run_id):
        by_strip.setdefault((run.site, run.strip), []).append(run)
    return [(run, event) for strip_runs in by_strip.values()
            for event, run in enumerate(strip_runs, start=1)]


def build(missions_root: str | Path = MISSIONS_ROOT, out_dir: str | Path = OUT_DIR):
    from datasets import Dataset, DatasetDict
    from PIL import Image

    rows: list[dict] = []
    for run, event in labeled_runs(missions_root):
        run_rows = extract_run(run, event)
        apply_labels(run_rows, run.run_dir)
        rows.extend(run_rows)
        print(f"[build] {run.site}/{run.strip} event {event}: {len(run_rows)} frames")
    add_polygons(rows)
    add_pairs(rows)

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for (site, event), flight in FLIGHTS.items():
        flight_rows = [r for r in rows if r["site"] == site and r["capture_event"] == event]
        ortho = Ortho(flight)
        for row in flight_rows:
            crop, coverage = ortho.sample(row["_e"], row["_n"], row["bearing_deg"])
            png = CROPS_DIR / f"{row['frame_id']}.png"
            if not png.exists():
                Image.fromarray(crop).save(png, optimize=True)
            row["drone_image"] = str(png)
            row["drone_coverage"] = coverage
        ortho.close()
        print(f"[build] {flight}: {len(flight_rows)} crops")

    assign_splits(rows)

    feats = features()
    parts = {}
    for split in SPLITS:
        split_rows = [r for r in rows if r["_split"] == split]
        parts[split] = Dataset.from_dict(
            {col: [r[col] for r in split_rows] for col in COLUMNS}, features=feats)
    ds = DatasetDict(parts)
    ds.save_to_disk(str(out_dir))
    counts = {split: parts[split].num_rows for split in SPLITS}
    print(f"[build] saved {sum(counts.values())} rows {counts} -> {out_dir}")
    return ds
