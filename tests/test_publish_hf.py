import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from publish_hf import plan  # noqa: E402


def test_plan_maps_each_checkpoint_folder_to_one_repo(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    for name in ("opendecider-2b", "opendecider-9b"):
        d = root / name
        d.mkdir(parents=True)
        (d / "README.md").write_text("card")
        (d / "config.json").write_text("{}")
        (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "LICENSE").write_text("apache")
    monkeypatch.chdir(tmp_path)
    p = plan("lab", root="checkpoints")
    assert [e["repo_id"] for e in p] == ["lab/opendecider-2b", "lab/opendecider-9b"]
    assert {f for _, f in p[0]["files"]} == {"README.md", "config.json", "model-00001-of-00001.safetensors", "LICENSE"}
    assert [e["repo_id"] for e in plan("lab", root="checkpoints", only=["opendecider-9b"])] == ["lab/opendecider-9b"]


def test_plan_rejects_a_folder_without_weights(tmp_path, monkeypatch):
    d = tmp_path / "checkpoints" / "broken"
    d.mkdir(parents=True)
    (d / "README.md").write_text("card")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        plan("lab", root="checkpoints")
