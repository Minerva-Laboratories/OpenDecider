"""Disk-guarded, revision-pinned model download.

    python -m opendecider.fetch --repo Qwen/Qwen3.5-2B --revision <sha> --out models/qwen3.5-2b

Checks the total download size from the Hub API first and refuses if it would leave < 4 GB free.
"""
from __future__ import annotations

import argparse

from huggingface_hub import HfApi, snapshot_download

from .guards import require_free_gb

SKIP = ["*.md", "*.gitattributes", "video_preprocessor_config.json"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    info = HfApi().model_info(a.repo, revision=a.revision, files_metadata=True)
    total = sum((s.size or 0) for s in info.siblings) / 2**30
    free = require_free_gb(total + 0.5, path=a.out)
    print(f"{a.repo}@{info.sha}: {total:.2f} GB, {free:.1f} GB free -> OK")
    snapshot_download(a.repo, revision=info.sha, local_dir=a.out, ignore_patterns=SKIP)
    print(f"pinned revision: {info.sha}")


if __name__ == "__main__":
    main()
