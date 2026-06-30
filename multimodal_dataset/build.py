"""Build the robot-survey Hugging Face dataset from completed missions.

Scans the repo's ``missions/`` tree, extracts one row per capture with the target
features (see ``schema.features``), and writes a Hugging Face dataset to ``out/``.

    python -m multimodal_dataset
"""
from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from .discover import completed_runs
from .extract import extract_run
from .schema import features

_PKG = Path(__file__).resolve().parent
MISSIONS_ROOT = _PKG.parent / "missions"        # the repo's missions/ tree
OUT_DIR = _PKG / "out"


def build(missions_root: str | Path = MISSIONS_ROOT,
          out_dir: str | Path = OUT_DIR) -> Dataset:
    runs = completed_runs(missions_root)
    rows = [row for run in runs for row in extract_run(run)]
    if not rows:
        raise RuntimeError(f"no completed-mission frames under {missions_root}")
    feats = features()
    ds = Dataset.from_dict({k: [r.get(k) for r in rows] for k in feats}, features=feats)
    ds.save_to_disk(str(out_dir))
    print(f"[multimodal_dataset] {len(runs)} mission(s), {ds.num_rows} frame(s) -> {out_dir}")
    return ds
