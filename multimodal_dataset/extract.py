"""Extract one dataset row per capture from a completed run.

Trusts the schema_version-3 sidecar for geometry; reads the JPEG bytes once for
sha256 + size. The capture stem is the ``frame_id`` join key. Returns plain dicts
keyed to ``schema.features``; omitted keys (e.g. the reserved label columns) become
null in the dataset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .discover import RunInfo

_RTK_FIXED_HACC_M = 0.05      # <= 5 cm horizontal accuracy => RTK-fixed


def _date_parts(iso: str | None):
    if not iso or len(iso) < 10:
        return None, None, None
    try:
        return int(iso[:4]), int(iso[5:7]), int(iso[8:10])
    except ValueError:
        return None, None, None


def _bearing_bucket(frame_id: str) -> str | None:
    for tok in frame_id.split("_"):
        if len(tok) > 1 and tok[0] == "b" and tok[1:].isdigit():
            return tok
    return None


def _row(sidecar: Path, run: RunInfo) -> dict | None:
    try:
        d = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    jpg = sidecar.with_suffix(".jpg")
    if not jpg.exists():
        return None
    data = jpg.read_bytes()
    pos = d.get("position") or {}
    head = d.get("heading") or {}
    frame = d.get("frame") or {}
    mission = d.get("mission") or {}
    line = ((d.get("extra") or {}).get("line_survey")) or {}
    hacc = pos.get("accuracy_horizontal")
    utc = d.get("captured_at_utc")
    year, month, day = _date_parts(utc)
    fid = sidecar.stem
    return {
        "frame_id": fid,
        "site": run.site, "strip": run.strip, "mission_name": run.mission_name,
        "run_id": run.run_id, "run_date": run.run_date,
        "strategy": mission.get("strategy"),
        "waypoint": mission.get("waypoint_name") or line.get("leg"),
        "leg": line.get("leg"), "mark_index": line.get("mark_index"),
        "along_track_m": line.get("along_track_m"), "interval_m": line.get("interval_m"),
        "bearing_bucket": _bearing_bucket(fid),
        "target_bearing": mission.get("target_bearing_deg_true") or head.get("target_degrees_true"),
        "captured_at_utc": utc, "captured_at_unix": d.get("captured_at_unix"),
        "year": year, "month": month, "day": day,
        "latitude": pos.get("latitude"), "longitude": pos.get("longitude"),
        "altitude": pos.get("altitude"), "altitude_msl": pos.get("altitude_msl"),
        "accuracy_horizontal": hacc, "accuracy_vertical": pos.get("accuracy_vertical"),
        "fix_type": pos.get("fix_type"), "satellites_used": pos.get("satellites_used"),
        "pdop": pos.get("pdop"), "position_interpolated": d.get("position_interpolated"),
        "correction_age_bin": pos.get("correction_age_bin"),
        "speed_over_ground": pos.get("speed_over_ground"),
        "course_over_ground": pos.get("course_over_ground"),
        "rtk_fixed": hacc is not None and hacc <= _RTK_FIXED_HACC_M,
        "heading_true": head.get("degrees_true"), "heading_target": head.get("target_degrees_true"),
        "heading_achieved": head.get("achieved_degrees_true"),
        "heading_residual": head.get("residual_degrees"),
        "heading_source": head.get("source"), "heading_course": head.get("course_degrees_true"),
        "image_width": frame.get("width"), "image_height": frame.get("height"),
        "corrupt": frame.get("corrupt"),
        "sha256": hashlib.sha256(data).hexdigest(), "file_size": len(data), "image_format": "jpg",
        "schema_version": d.get("schema_version"),
        "go2_survey_version": run.go2_survey_version, "git_sha": run.git_sha,
        "run_dir": str(run.run_dir),
        "is_labeled": False,                       # labels reserved; null until filled later
        "image": str(jpg),
    }


def extract_run(run: RunInfo) -> list[dict]:
    rows = []
    for sidecar in sorted(run.captures_dir.glob("*.json")):
        row = _row(sidecar, run)
        if row is not None:
            rows.append(row)
    return rows
