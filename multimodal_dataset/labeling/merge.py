"""Bridge: project a run's saved labels onto the release columns.

The labeling app wrote ``<run>/labels/species.json`` (schema ``strip-species/v1``,
name-keyed ``present`` lists) and ``<run>/labels/quality.json`` (schema
``strip-quality/v1``, quartile bins 1..4). This fills each row's ``species``
(present-species class ids in ``CLASSES`` order) and ``artifact_level``
(bin - 1), joined by ``frame_id``. Every frame must be present and reviewed in
both files — the release has no unlabeled rows. ``species.archived.json``
backups are ignored.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..classes import CLASSES


def _labels(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text())["labels"]


def apply_labels(rows: list[dict], run_dir: str | Path) -> None:
    """Fill ``species`` and ``artifact_level`` on a run's rows, in place."""
    labels_dir = Path(run_dir) / "labels"
    species = _labels(labels_dir / "species.json")
    quality = _labels(labels_dir / "quality.json")
    for row in rows:
        fid = row["frame_id"]
        s, q = species.get(fid), quality.get(fid)
        if not (s and s.get("reviewed")):
            raise ValueError(f"{fid}: no reviewed species label in {labels_dir}")
        if not (q and q.get("reviewed") and 1 <= q.get("bin", 0) <= 4):
            raise ValueError(f"{fid}: no reviewed quality label in {labels_dir}")
        row["species"] = sorted(CLASSES.index(name) for name in (s.get("present") or []))
        row["artifact_level"] = q["bin"] - 1
