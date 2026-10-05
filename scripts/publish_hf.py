"""Publish the released checkpoints to Hugging Face model repos (private by default).

    huggingface-cli login                                   # once, with a token that can write to the organization
    .venv/bin/python scripts/publish_hf.py --org <org> --dry-run    # what would be uploaded, no network writes
    .venv/bin/python scripts/publish_hf.py --org <org>              # create <org>/<checkpoint> repos and upload

Each folder under checkpoints/ becomes one repo with the same name. Uploaded: the folder's files (model card, config,
safetensors shards) plus the repository's LICENSE and NOTICE. The working tree must be clean, so every upload maps to
one git commit, which is recorded in the commit message on Hugging Face. Repos are created private unless --public.
"""
from __future__ import annotations

import argparse
import os
import subprocess

FILES_FROM_ROOT = ("LICENSE", "NOTICE")


def git(*args) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def plan(org: str, root: str = "checkpoints", only: list[str] | None = None) -> list[dict]:
    """One entry per checkpoint folder: repo id and the (local path, path in repo) pairs to upload."""
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d) or (only and name not in only):
            continue
        files = [(os.path.join(d, f), f) for f in sorted(os.listdir(d)) if not f.startswith(".")]
        if not any(f.endswith(".safetensors") for _, f in files) or not any(f == "README.md" for _, f in files):
            raise ValueError(f"{d}: needs README.md and safetensors shards (run scripts/export_checkpoints.py)")
        files += [(f, f) for f in FILES_FROM_ROOT if os.path.exists(f)]
        out.append({"repo_id": f"{org}/{name}", "files": files,
                    "bytes": sum(os.path.getsize(p) for p, _ in files)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="Hugging Face organization (or user) name")
    ap.add_argument("--public", action="store_true", help="create public repos (default: private)")
    ap.add_argument("--only", nargs="*", help="checkpoint folder names to publish (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if git("status", "--porcelain"):
        msg = "working tree has uncommitted changes: commit first, so the upload maps to one git commit"
        if not a.dry_run:
            raise SystemExit(msg)
        print(f"warning: {msg}")
    sha = git("rev-parse", "HEAD")
    entries = plan(a.org, only=a.only)
    for e in entries:
        print(f"{e['repo_id']} ({'public' if a.public else 'private'}): {len(e['files'])} files, "
              f"{e['bytes'] / 2 ** 20:.1f} MB")
        for p, f in e["files"]:
            print(f"   {f}  <-  {p}")
    if a.dry_run:
        print(f"dry run: nothing uploaded (git commit {sha[:12]})")
        return
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    for e in entries:
        api.create_repo(e["repo_id"], repo_type="model", private=not a.public, exist_ok=True)
        ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=p) for p, f in e["files"]]
        info = api.create_commit(e["repo_id"], operations=ops, repo_type="model",
                                 commit_message=f"OpenDecider release, github commit {sha}")
        print(f"published {e['repo_id']} @ {info.oid} (github {sha[:12]})")


if __name__ == "__main__":
    main()
