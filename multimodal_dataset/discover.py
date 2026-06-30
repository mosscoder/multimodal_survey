"""Find completed robot missions under ``missions/<site>/<strip>/runs/<run_id>/``.

A run is COMPLETE iff its ``main.log`` holds the ``MISSION COMPLETE`` marker (one
run or resumed). The log header carries the robot software provenance
(``go2-survey vX | git YYYY``), denormalised onto every frame.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADER = re.compile(r"go2-survey\s+(v[\d.]+)\s*\|\s*git\s+([0-9a-f]+)", re.I)
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass
class RunInfo:
    site: str
    strip: str
    run_id: str
    run_dir: Path
    captures_dir: Path
    go2_survey_version: str | None = None
    git_sha: str | None = None

    @property
    def mission_name(self) -> str:
        return f"{self.site}_{self.strip}"

    @property
    def run_date(self) -> str | None:
        m = _DATE.search(self.run_id)
        return m.group(1) if m else None


def completed_runs(root: str | Path) -> list[RunInfo]:
    """Every completed mission under ``root`` (gated on the MISSION COMPLETE marker)."""
    root = Path(root)
    out: list[RunInfo] = []
    for run_dir in sorted(root.glob("*/*/runs/*")):
        parts = run_dir.relative_to(root).parts        # (site, strip, "runs", run_id)
        if not run_dir.is_dir() or len(parts) != 4 or parts[2] != "runs":
            continue
        log = run_dir / "main.log"
        text = log.read_text(errors="ignore") if log.exists() else ""
        if "MISSION COMPLETE" not in text:
            continue
        site, strip, _, run_id = parts
        m = _HEADER.search(text)
        out.append(RunInfo(site, strip, run_id, run_dir, run_dir / "captures",
                           *(m.groups() if m else (None, None))))
    return out
