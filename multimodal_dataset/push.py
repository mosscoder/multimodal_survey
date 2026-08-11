"""Push the verified build and the dataset card to the private HF repo.

    python -m multimodal_dataset push

Reads the write token from the repo-root ``.env`` (``HF_TOKEN=...``), creates
the private repo if needed, uploads the three splits (image columns embedded
into parquet by ``push_to_hub``), then uploads the card: ``card/README.md``
with its YAML front matter merged over the hub-generated ``dataset_info``
block and ``SOURCE_COMMIT`` replaced by the current git commit, plus the
figures, ``species.csv``, and ``CITATION.cff``.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

from .build import OUT_DIR

REPO_ID = "mpg-ranch/multimodal-survey"
_ROOT = Path(__file__).resolve().parent.parent
_CARD = Path(__file__).resolve().parent / "card"


def hf_token() -> str:
    for line in (_ROOT / ".env").read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key == "HF_TOKEN" and value:
            return value
    raise RuntimeError(f"no HF_TOKEN in {_ROOT / '.env'}")


def source_commit() -> str:
    """The commit the build ran from: stamped by ``build`` into ``out/``,
    falling back to the current HEAD for builds made before the stamp existed."""
    stamp = Path(OUT_DIR) / "source_commit.txt"
    if stamp.exists():
        return stamp.read_text().strip()
    return subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _split_front_matter(text: str) -> tuple[dict, str]:
    import yaml
    if text.startswith("---\n"):
        head, _, body = text[4:].partition("\n---\n")
        return yaml.safe_load(head) or {}, body.lstrip("\n")
    return {}, text


def merged_readme(hub_readme: str) -> str:
    """The card body under YAML that keeps the hub's dataset_info/configs and
    adds the card's own metadata (license, tags, ...)."""
    import yaml
    hub_meta, _ = _split_front_matter(hub_readme)
    card_meta, card_body = _split_front_matter((_CARD / "README.md").read_text())
    meta = {**card_meta, **hub_meta}         # hub keys (dataset_info) win
    card_body = card_body.replace("SOURCE_COMMIT", source_commit())
    return f"---\n{yaml.safe_dump(meta, sort_keys=False)}---\n\n{card_body}"


def push(out_dir: str | Path = OUT_DIR, repo_id: str = REPO_ID):
    from datasets import load_from_disk
    from huggingface_hub import HfApi

    token = hf_token()
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    ds = load_from_disk(str(out_dir))
    ds.push_to_hub(repo_id, private=True, token=token)

    hub_readme = api.hf_hub_download(repo_id, "README.md", repo_type="dataset",
                                     force_download=True)
    readme = merged_readme(Path(hub_readme).read_text())
    api.upload_folder(folder_path=str(_CARD), repo_id=repo_id, repo_type="dataset",
                      ignore_patterns=["README.md"],
                      commit_message="card: figures, species.csv, CITATION.cff")
    api.upload_file(path_or_fileobj=io.BytesIO(readme.encode()),
                    path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
                    commit_message="card: README")
    print(f"[push] {sum(ds[s].num_rows for s in ds)} rows + card -> "
          f"https://huggingface.co/datasets/{repo_id} (private, source {source_commit()[:8]})")
