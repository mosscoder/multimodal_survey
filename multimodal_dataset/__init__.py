"""Build the multimodal-survey Hugging Face release from ``missions/``.

Run: ``python -m multimodal_dataset {build|verify|push}``. See ``schema``
(the 33 release features), ``discover`` (mission gate), ``extract`` (rows,
pairs, restarts, planned marks), ``crops`` (drone-image geometry), ``splits``
(test-leg rule), ``verify`` (release gates), ``push`` (private HF upload).
Dependencies in ``requirements.txt``; the release spec is
``docs/dataset-overview.html``.
"""
from __future__ import annotations
