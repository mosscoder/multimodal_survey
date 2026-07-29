"""Local web labelling apps for robot survey missions.

  * ``server.py`` — unbiased image-level multilabel labeller (stdlib only).
  * ``review.py`` — FP/FN audit against the MIL model's predictions (needs numpy +
    MIL prediction artifacts passed in via ``--summary``/``--overlays``).

Both require a mission (no default): pick one with ``--mission site_x/strip_y`` (or
``--run <run_id>`` when a mission has several completed runs); both read/write that
run's ``labels/species.json``. Run:

    python -m multimodal_dataset.labeling label  --mission site_1/strip_5 --run <run_id>
    python -m multimodal_dataset.labeling review --run <run_id> \\
        --summary <demo>/inference_viz/summary.json --overlays <demo>/inference_viz/overlays
"""
from __future__ import annotations

from pathlib import Path

from ..discover import RunInfo, completed_runs

_REPO_ROOT = Path(__file__).resolve().parents[2]                   # the repo root
_MISSIONS_ROOT = _REPO_ROOT / "missions"                           # the repo's missions/


def select_run(*, mission: str | None = None, run: str | None = None,
               missions_root: str | None = None) -> RunInfo:
    """Resolve ``--mission``/``--run`` to exactly one completed run, or exit with guidance."""
    runs = completed_runs(missions_root or _MISSIONS_ROOT)
    if run:
        matches = [r for r in runs if r.run_id == run]
    elif mission:
        key = mission.strip("/").replace("/", "_")
        matches = [r for r in runs if r.mission_name == key]
    else:
        raise SystemExit("specify a mission: --mission site_x/strip_y  (or --run <run_id>)")
    if not matches:
        missions = ", ".join(sorted({r.mission_name for r in runs})) or "(none)"
        raise SystemExit(f"no completed run matched. completed missions: {missions}")
    if len(matches) > 1:
        ids = "\n  ".join(r.run_id for r in matches)
        raise SystemExit(f"that mission has several completed runs; choose one with --run:\n  {ids}")
    return matches[0]
