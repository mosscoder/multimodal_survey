"""Extract release rows from completed runs.

Trusts the schema_version-3 sidecar for geometry and telemetry. Three stages,
each pure over row dicts keyed to ``schema.COLUMNS`` plus internal ``_``-keys
(``_mark``, ``_e``/``_n``, ``_poly``, ``_dt``) that ``build`` drops before
assembly:

* :func:`extract_run` — one row per capture sidecar, with the flight-anchored
  timing columns and the crop footprint.
* :func:`add_restarts` — per run, stamps ``mission_restart_utc`` on frames
  captured after a mid-run gap of more than an hour.
* :func:`add_pairs` — across a strip's two capture events, joins frames at the
  same (leg, mark) into the ``pair_*`` columns.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .crops import FLIGHTS, S, antenna_en, footprint_corners_en, footprint_wkt
from .discover import RunInfo

_REPO = Path(__file__).resolve().parent.parent
MEDIAN_TIMES = _REPO / "docs" / "figures" / "drone_flight_median_times.json"

RESTART_GAP_S = 3600.0


def flight_median_time(site: str, capture_event: int) -> datetime:
    """The paired flight's median source-photo timestamp (UTC)."""
    flights = json.loads(MEDIAN_TIMES.read_text())
    stamp = flights[FLIGHTS[(site, capture_event)]]
    return datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)


def _wrap_12h(hours: float) -> float:
    """Clock-time difference, date discarded, wrapped to (-12, +12]."""
    return -(((-hours + 12.0) % 24.0) - 12.0)


def _row(sidecar: Path, run: RunInfo, capture_event: int, drone_dt: datetime) -> dict | None:
    try:
        d = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    jpg = sidecar.with_suffix(".jpg")
    if not jpg.exists():
        return None
    pos = d.get("position") or {}
    head = d.get("heading") or {}
    line = ((d.get("extra") or {}).get("line_survey")) or {}

    lat, lon = pos["latitude"], pos["longitude"]
    bearing = head["course_degrees_true"]
    ground_dt = datetime.fromisoformat(d["captured_at_utc"])
    offset_h = (ground_dt - drone_dt).total_seconds() / 3600.0
    e, n = antenna_en(lat, lon)

    return {
        "frame_id": sidecar.stem,
        "ground_image": str(jpg),
        "site": run.site,
        "strip": run.strip,
        "latitude": lat,
        "longitude": lon,
        "altitude_ellipsoidal_m": pos.get("altitude"),
        "accuracy_horizontal_m": pos.get("accuracy_horizontal"),
        "accuracy_vertical_m": pos.get("accuracy_vertical"),
        "fix_type": pos.get("fix_type"),
        "pdop": pos.get("pdop"),
        "satellites_used": pos.get("satellites_used"),
        "bearing_deg": bearing,
        "footprint_wkt": footprint_wkt(lat, lon, bearing),
        "ground_captured_at": d["captured_at_utc"],
        "drone_captured_at": drone_dt.isoformat(),
        "capture_offset_absolute": offset_h,
        "capture_offset_time_of_day": _wrap_12h(offset_h),
        "capture_event": capture_event,
        "mission_restart_utc": None,
        "pair_frame_id": None,
        "pair_distance_m": None,
        "pair_overlap_fraction": None,
        "pair_offset_hours": None,
        "leg": line.get("leg"),
        "along_leg_m": line.get("along_track_m"),
        "frame_offset_s": line.get("frame_offset_from_mark_s"),
        "speed_mps": pos.get("speed_over_ground"),
        "run_id": run.run_id,
        "tier2_overlap_fraction": None,
        "drone_coverage": None,
        # internal, dropped before assembly
        "_mark": line.get("mark_index"),
        "_e": e,
        "_n": n,
        "_dt": ground_dt,
    }


def extract_run(run: RunInfo, capture_event: int) -> list[dict]:
    drone_dt = flight_median_time(run.site, capture_event)
    rows = []
    for sidecar in sorted(run.captures_dir.glob("*.json")):
        if sidecar.name == "captures.geojson":
            continue
        row = _row(sidecar, run, capture_event, drone_dt)
        if row is not None:
            rows.append(row)
    add_restarts(rows)
    return rows


def add_restarts(run_rows: list[dict]) -> None:
    """Stamp ``mission_restart_utc`` on every frame after a mid-run gap."""
    ordered = sorted(run_rows, key=lambda r: r["_dt"])
    restart = None
    for prev, cur in zip(ordered, ordered[1:]):
        if (cur["_dt"] - prev["_dt"]).total_seconds() > RESTART_GAP_S:
            restart = cur["ground_captured_at"]
        if restart is not None:
            cur["mission_restart_utc"] = restart


def add_pairs(rows: list[dict]) -> None:
    """Join each frame to its counterpart at the same (leg, mark) in the
    strip's other capture event. Single-visit marks keep null pair columns."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["site"], row["strip"], row["leg"], row["_mark"]), []).append(row)
    for key, pair in groups.items():
        if len(pair) == 1:
            continue
        if len(pair) != 2 or pair[0]["capture_event"] == pair[1]["capture_event"]:
            raise ValueError(f"unexpected repeat structure at {key}: "
                             f"{[r['frame_id'] for r in pair]}")
        a, b = pair
        dist = math.hypot(a["_e"] - b["_e"], a["_n"] - b["_n"])
        shared = a["_poly"].intersection(b["_poly"]).area / (S * S)
        hours = abs((a["_dt"] - b["_dt"]).total_seconds()) / 3600.0
        for row, other in ((a, b), (b, a)):
            row["pair_frame_id"] = other["frame_id"]
            row["pair_distance_m"] = dist
            row["pair_overlap_fraction"] = shared
            row["pair_offset_hours"] = hours


def add_polygons(rows: list[dict]) -> None:
    """Attach the shapely crop polygon (EPSG:6514) each row's geometry implies."""
    from shapely.geometry import Polygon
    for row in rows:
        row["_poly"] = Polygon(footprint_corners_en(row["_e"], row["_n"], row["bearing_deg"]))
