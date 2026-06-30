"""Dispatch the labelling apps.

    python -m multimodal_dataset.labeling label  --mission site_x/strip_y [...]
    python -m multimodal_dataset.labeling review --run <run_id> --summary ... --overlays ...
"""
from __future__ import annotations

import sys

_USAGE = ("usage: python -m multimodal_dataset.labeling {label|review} "
          "--mission site_x/strip_y | --run <run_id> [...]")


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("label", "review"):
        print(_USAGE)
        raise SystemExit(0 if argv[:1] in (["-h"], ["--help"]) else 2)
    cmd, rest = argv[0], argv[1:]
    if cmd == "label":
        from .server import main as run_main
    else:
        from .review import main as run_main
    run_main(rest)


if __name__ == "__main__":
    main()
