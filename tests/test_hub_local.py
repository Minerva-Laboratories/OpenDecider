"""Backbones the user already has (hub source with local_env): resolved from backbone_path or the env var."""
import pytest

from opendecider.hub import resolve_backbone

SPEC = {"weight_quant": "mercurius", "dtype": "bfloat16",
        "source": {"repo_id": "Org/Model", "revision": "abc123", "local_env": "OD_TEST_ROOT"}}


def test_local_backbone_from_argument_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OD_TEST_ROOT", raising=False)
    out = resolve_backbone(SPEC, str(tmp_path))
    assert out["path"] == str(tmp_path) and out["repo_id"] == "Org/Model" and out["revision"] == "abc123"
    assert "source" not in out and "source" in SPEC                    # the spec itself is not modified
    monkeypatch.setenv("OD_TEST_ROOT", str(tmp_path))
    assert resolve_backbone(SPEC)["path"] == str(tmp_path)


def test_local_backbone_missing_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OD_TEST_ROOT", raising=False)
    with pytest.raises(ValueError, match="OD_TEST_ROOT"):
        resolve_backbone(SPEC)
    with pytest.raises(ValueError, match="does-not-exist"):
        resolve_backbone(SPEC, "/does-not-exist")
