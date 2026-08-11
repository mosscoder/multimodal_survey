"""Dispatch the labelling app (historical: labelling is complete; kept as the
provenance of how the shipped labels were produced).

    python -m multimodal_dataset.labeling label --mission site_x/strip_y [...]
"""
from __future__ import annotations

import sys

_USAGE = "usage: python -m multimodal_dataset.labeling label --mission site_x/strip_y [...]"


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] != "label":
        print(_USAGE)
        raise SystemExit(0 if argv[:1] in (["-h"], ["--help"]) else 2)
    from .server import main as run_main
    run_main(argv[1:])


if __name__ == "__main__":
    main()
