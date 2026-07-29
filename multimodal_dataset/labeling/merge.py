"""Bridge: project a run's labels onto the dataset's reserved label columns.

The labelling apps write ``<run>/labels/species.json`` (schema
``strip-species/v1``, name-keyed ``present`` lists). This fills the
dataset's reserved label columns (see ``multimodal_dataset.schema.features``) for
frames present + reviewed in that file — joined by ``frame_id``, name-based
multi-hot over ``CLASSES``. An unlabelled frame stays ``labels = null`` (NOT
all-zeros). Not wired into the default build; call it when assembling a labelled
dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..classes import CLASSES


def load_labels(path: str | Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text()).get("labels", {})


def multi_hot(present: list[str]) -> list[int]:
    have = set(present)
    return [1 if c in have else 0 for c in CLASSES]


def apply_labels(rows: list[dict], labels: str | Path | dict[str, dict],
                 *, require_reviewed: bool = True) -> int:
    """Fill label columns in dataset row dicts (from ``extract``) in place, by
    ``frame_id``. Returns the number of frames labelled."""
    table = labels if isinstance(labels, dict) else load_labels(labels)
    merged = 0
    for row in rows:
        entry = table.get(row.get("frame_id"))
        if entry is None or (require_reviewed and not entry.get("reviewed")):
            continue
        present = entry.get("present") or []
        row["is_labeled"] = True
        row["labels"] = multi_hot(present)
        row["label_names"] = list(present)
        row["label_reviewed"] = bool(entry.get("reviewed"))
        row["label_source"] = entry.get("source")
        row["label_updated"] = entry.get("updated")
        merged += 1
    return merged
