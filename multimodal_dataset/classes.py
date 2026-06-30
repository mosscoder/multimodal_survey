"""Canonical focal species — the one ordered class list shared by the labelling
apps and the dataset's label columns.

Order is the pipeline's order; it matches the MIL program's ``core/data.NAMES`` by
VALUE (not imported across packages, to keep this list dependency-free / stdlib).
``DEFAULT_ON`` is the always-present grass the labeller checks by default on every
fresh frame.
"""
from __future__ import annotations

CLASSES = [
    "Lupinus sericeus",
    "Poa bulbosa",
    "Tragopogon dubius",
    "Gaillardia aristata",       # blanketflower
    "Balsamorhiza sagittata",    # arrowleaf balsamroot
    "Bromus tectorum",
    "Achillea millefolium",
    "Sisymbrium altissimum",
    "Thinopyrum intermedium",    # last: default-on, ubiquitous
]
DEFAULT_ON = "Thinopyrum intermedium"
