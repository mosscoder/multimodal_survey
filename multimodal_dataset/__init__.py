"""Build the robot-survey Hugging Face dataset from completed missions in ``missions/``.

Self-contained (only ``datasets`` + ``pillow``). Run: ``python -m multimodal_dataset``.
See ``schema.features`` (target features), ``discover`` (mission gate),
``extract`` (per-frame row), ``build`` (assemble + save).
"""
from __future__ import annotations
