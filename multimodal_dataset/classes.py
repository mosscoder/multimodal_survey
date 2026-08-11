"""Canonical focal species — the one ordered class list shared by the labelling
apps and the dataset's label columns.

Refocused onto 8 target species: 4 wildflowers + 4 weeds. ``GROUPS`` carries the
wildflower/weed split the labelling UI renders; ``CLASSES`` is the flat pipeline
order (wildflowers, then weeds) used for label vectors and dataset columns.
``DEFAULT_ON`` is the species pre-checked on every fresh frame, or ``None`` for a
blank start — here ``None``, since none of the 8 targets is ubiquitous (the old
default-on matrix grass, Thinopyrum intermedium, was dropped in the refocus).

Saved labels migrate by species NAME, so this list can be reordered or changed
without scrambling existing annotations.
"""
from __future__ import annotations

GROUPS = [
    ("Wildflowers", [
        "Lupinus sericeus",          # silky lupine
        "Balsamorhiza sagittata",    # arrowleaf balsamroot
        "Gaillardia aristata",       # blanketflower
        "Achillea millefolium",      # common yarrow
    ]),
    ("Weeds", [
        "Euphorbia virgata",         # leafy spurge
        "Sisymbrium altissimum",     # tall tumblemustard
        "Bromus tectorum",           # cheatgrass
        "Poa bulbosa",               # bulbous bluegrass
    ]),
]

# Flat pipeline order (label-vector / dataset-column order): wildflowers, then weeds.
CLASSES = [sp for _group, species in GROUPS for sp in species]

# Species pre-checked on a fresh frame; None = blank start.
DEFAULT_ON = None


# --------------------------------------------------------------------------- #
# Image-quality task                                                          #
# --------------------------------------------------------------------------- #
# A second, independent labelling task: instead of which species are present, a
# single ordinal judgement per frame — how much of the image is degraded by
# visual artifacts (motion blur, smear, glare, compression, exposure). Four equal
# quartile bands over the *share of the frame affected*; stored behind the scenes
# as the integer 1..4 (0 = unset). Hot-keys 1–4 map to the four bands in order.
QUALITY_LABEL = "Artifact prevalence"
QUALITY_BINS = [
    ("0–25%",   "clean — few/no visual artifacts"),
    ("25–50%",  "light — artifacts over a minority of the frame"),
    ("50–75%",  "heavy — artifacts over most of the frame"),
    ("75–100%", "severe — frame dominated by artifacts"),
]

# Release names for the quality bands (the dataset's `artifact_level` ClassLabel).
# Plain hyphens, unlike the en-dashed UI strings above; index = stored bin - 1.
ARTIFACT_LEVELS = ["0-25%", "25-50%", "50-75%", "75-100%"]
