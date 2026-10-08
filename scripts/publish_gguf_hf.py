"""Upload the Mercurius-1-4B GGUFs to the model's Hugging Face repo, under gguf/ (the repo must exist; its
visibility is not changed).

    HF_TOKEN=... .venv/bin/python scripts/publish_gguf_hf.py --dry-run    # checks + shows the root-card change
    HF_TOKEN=... .venv/bin/python scripts/publish_gguf_hf.py              # upload

The token is read from the environment only (never stored). Uploaded: the GGUF files listed in SHA256SUMS (verified
before upload), the GGUF model card (README.md -> gguf/README.md) and SHA256SUMS, plus the repo's root README.md with a
short "GGUF (llama.cpp)" section and gguf / llama.cpp tags added (left as is if the section already exists). After
the upload, the LFS checksums on the Hub are compared with the local ones.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tempfile

from huggingface_hub import HfApi, hf_hub_download

FORK = "https://github.com/alexhergomz/llama-mercurius"
MARK = "## GGUF (llama.cpp)"
SECTION = f"""{MARK}

Ready-to-run GGUF files are in [`gguf/`](gguf/): exact NF4 (recommended), IQ4_NL, Q4_K, Q5_K and f16. They need the
Mercurius llama.cpp fork, [alexhergomz/llama-mercurius]({FORK}) (branch `mercurius`); stock llama.cpp, Ollama, LM Studio
and llama-cpp-python cannot load them. Runs on NVIDIA GPUs (CUDA; Ampere and newer, tested on Jetson AGX Orin; Turing
expected to work) and on CPU (slow). Not supported yet: AMD (ROCm), Apple Metal, Vulkan, SYCL. Usage, benchmarks and
hardware notes: [`gguf/README.md`](gguf/README.md).

"""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def root_card(text: str) -> str:
    """Root README with the GGUF section (before '## Files', else appended) and gguf / llama.cpp tags."""
    if MARK in text:
        return text
    m = re.match(r"(---\n.*?\n---\n)", text, re.S)
    if m:
        fm = m.group(1)
        t = re.search(r"^tags: \[(.*)\]$", fm, re.M)
        if t:
            tags = [x.strip() for x in t.group(1).split(",") if x.strip()]
            tags += [x for x in ("gguf", "llama.cpp") if x not in tags]
            fm = fm.replace(t.group(0), f"tags: [{', '.join(tags)}]")
        text = fm + text[m.end():]
    if "\n## Files" in text:
        text = text.replace("\n## Files", "\n" + SECTION + "## Files", 1)
        text = re.sub(r"(\| `MANIFEST.md` \|[^\n]*\n)", r"\1| `gguf/` | GGUF builds for llama.cpp (needs the fork, see above) |\n", text, 1)
    else:
        text = text.rstrip("\n") + "\n\n" + SECTION
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Minerva-Laboratories/Mercurius-1-4B")
    ap.add_argument("--src", default="/mnt/ext/opendecider/gguf")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN in the command's environment")
    api = HfApi(token=token)
    info = api.model_info(a.repo)                                   # fails if the repo does not exist / no access
    print(f"{a.repo}: {'private' if info.private else 'PUBLIC'}, {len(info.siblings)} files now")

    sums = {}
    for line in open(os.path.join(a.src, "SHA256SUMS")):
        h, name = line.split()
        sums[name] = h
    for name, h in sums.items():
        got = sha256(os.path.join(a.src, name))
        if got != h:
            raise SystemExit(f"checksum mismatch: {name}")
        print(f"   ok {name} ({os.path.getsize(os.path.join(a.src, name)) / 1e9:.2f} GB)")
    card = open(os.path.join(a.src, "README.md")).read()
    if "<" + "FORK URL>" in card or "<BENCH TABLE>" in card:
        raise SystemExit("GGUF card still has placeholders")

    old_root = open(hf_hub_download(a.repo, "README.md", token=token)).read()
    new_root = root_card(old_root)
    print("root README:", "unchanged (section exists)" if new_root == old_root else "GGUF section + tags added")
    if a.dry_run:
        import difflib
        print("".join(difflib.unified_diff(old_root.splitlines(True), new_root.splitlines(True), "README.md (Hub)",
                                           "README.md (new)")))
        print("dry run: nothing uploaded")
        return

    with tempfile.TemporaryDirectory(dir=os.path.dirname(a.src.rstrip("/"))) as stage:
        os.makedirs(os.path.join(stage, "gguf"))
        for name in sums:                                           # hard links: no copy of the large files
            os.link(os.path.join(a.src, name), os.path.join(stage, "gguf", name))
        shutil.copy(os.path.join(a.src, "README.md"), os.path.join(stage, "gguf", "README.md"))
        shutil.copy(os.path.join(a.src, "SHA256SUMS"), os.path.join(stage, "gguf", "SHA256SUMS"))
        open(os.path.join(stage, "README.md"), "w").write(new_root)
        api.upload_folder(repo_id=a.repo, folder_path=stage, commit_message="Add GGUF builds for llama.cpp (gguf/)")

    remote = {p.path: p for p in api.get_paths_info(a.repo, [f"gguf/{n}" for n in sums])}
    for name, h in sums.items():
        p = remote.get(f"gguf/{name}")
        ok = p is not None and p.lfs is not None and p.lfs.sha256 == h
        print(f"   {'verified' if ok else 'MISMATCH'} gguf/{name}")
    print("done")


if __name__ == "__main__":
    main()
