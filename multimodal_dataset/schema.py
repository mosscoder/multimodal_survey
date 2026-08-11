"""The 34 release features of the multimodal-survey Hugging Face dataset.

One row per labeled robot capture: the ground photograph, its 4 x 4 m nadir
drone-orthomosaic crop, the species and quality labels, and the telemetry the
release documents. Column names, order, and semantics follow
``docs/dataset-overview.html`` exactly; that page is the spec, this file is its
executable form. The join key is ``frame_id`` (the capture stem).
"""
from __future__ import annotations

from datasets import ClassLabel, Features, Image, Sequence, Value

from .classes import ARTIFACT_LEVELS, CLASSES

_S, _F = Value("string"), Value("float64")
_I8 = Value("int8")


def features() -> Features:
    return Features({
        # identity & imagery
        "frame_id": _S,
        "ground_image": Image(),
        "drone_image": Image(),
        # labels
        "species": Sequence(ClassLabel(names=CLASSES)),
        "artifact_level": ClassLabel(names=ARTIFACT_LEVELS),
        # where
        "site": _S,
        "strip": _S,
        "latitude": _F,
        "longitude": _F,
        "altitude_ellipsoidal_m": _F,
        "accuracy_horizontal_m": _F,
        "accuracy_vertical_m": _F,
        "fix_type": _I8,
        "pdop": _F,
        "satellites_used": _I8,
        "bearing_deg": _F,
        "footprint_wkt": _S,
        # when
        "ground_captured_at": _S,
        "drone_captured_at": _S,
        "capture_offset_absolute": _F,
        "capture_offset_time_of_day": _F,
        "capture_event": _I8,
        "mission_restart_utc": _S,           # null except after the mid-run resume
        # cross-event pair
        "pair_frame_id": _S,                 # null on single-visit marks
        "pair_distance_m": _F,
        "pair_overlap_fraction": _F,
        "pair_offset_hours": _F,
        # along the transect
        "leg": _S,
        "along_leg_m": _F,
        "frame_offset_s": _F,
        "speed_mps": _F,
        # provenance & split
        "run_id": _S,
        "tier2_overlap_fraction": _F,        # populated on train_tier_2 rows only
        "drone_coverage": _F,
    })


COLUMNS = list(features().keys())
