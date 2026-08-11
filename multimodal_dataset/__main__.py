from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="python -m multimodal_dataset",
        description="Build, verify, and push the multimodal-survey HF dataset.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="assemble the release into out/")
    sub.add_parser("verify", help="assert the release invariants against out/")
    sub.add_parser("push", help="upload the verified build to the private HF repo")
    args = parser.parse_args()

    if args.command == "build":
        from .build import build
        build()
    elif args.command == "verify":
        from .verify import verify
        verify()
    else:
        from .push import push
        push()


if __name__ == "__main__":
    main()
