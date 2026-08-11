"""Build the multimodal-survey Hugging Face release from ``missions/``.

Run: ``python -m multimodal_dataset {build|verify|push}``. See ``schema``
(the 34 release features), ``discover`` (mission gate), ``extract`` (rows,
pairs, restarts), ``crops`` (drone-crop geometry), ``splits`` (test-leg rule),
``verify`` (release gates), ``push`` (private HF upload). Dependencies in
``requirements.txt``; the release spec is ``docs/dataset-overview.html``.
"""
from __future__ import annotations
