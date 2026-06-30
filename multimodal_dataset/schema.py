"""The target features of the robot-survey Hugging Face dataset.

One row per geotagged robot capture: the native image plus denormalised mission /
GPS-RTK / heading / leg telemetry from the schema_version-3 sidecar. The join key
is ``frame_id`` (the capture stem). Label columns are reserved but nullable — an
unlabelled frame has ``labels = null`` (NOT all-zeros) — and are filled later from
the labelling app.
"""
from __future__ import annotations

from datasets import Features, Image, Sequence, Value

_S, _I, _F, _B = Value("string"), Value("int64"), Value("float64"), Value("bool")


def features() -> Features:
    return Features({
        # identity / join key
        "frame_id": _S,
        # mission / run provenance
        "site": _S, "strip": _S, "mission_name": _S, "run_id": _S, "run_date": _S,
        "strategy": _S, "waypoint": _S, "leg": _S, "mark_index": _I,
        "along_track_m": _F, "interval_m": _F, "bearing_bucket": _S, "target_bearing": _F,
        # temporal
        "captured_at_utc": _S, "captured_at_unix": _F, "year": _I, "month": _I, "day": _I,
        # position / GPS quality
        "latitude": _F, "longitude": _F, "altitude": _F, "altitude_msl": _F,
        "accuracy_horizontal": _F, "accuracy_vertical": _F, "fix_type": _I,
        "satellites_used": _I, "pdop": _F, "position_interpolated": _B,
        "correction_age_bin": _I, "speed_over_ground": _F, "course_over_ground": _F,
        "rtk_fixed": _B,
        # heading
        "heading_true": _F, "heading_target": _F, "heading_achieved": _F,
        "heading_residual": _F, "heading_source": _S, "heading_course": _F,
        # frame geometry + image-bytes provenance
        "image_width": _I, "image_height": _I, "corrupt": _B,
        "sha256": _S, "file_size": _I, "image_format": _S,
        # robot-software provenance
        "schema_version": _I, "go2_survey_version": _S, "git_sha": _S, "run_dir": _S,
        # labels — reserved, nullable (filled later; unlabelled = null, not zero)
        "is_labeled": _B, "labels": Sequence(Value("int8")), "label_names": Sequence(_S),
        "label_reviewed": _B, "label_source": _S, "label_updated": _S, "mil_role": _S,
        # the image itself
        "image": Image(),
    })
