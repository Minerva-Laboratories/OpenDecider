"""Copy a checkpoint with another backbone config (same trained head), so every eval tool can test a backbone variant.

    python scripts/swap_backbone.py --ckpt runs/x/model.pt --backbone configs/backbone_9b_awq.yaml --out runs/x-awq/model.pt
"""
import argparse
import os

import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", required=True, help="backbone yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    new = yaml.safe_load(open(a.backbone))
    keep = set(ck["backbone_cfg"])
    if new.get("feature_layers") != ck["backbone_cfg"].get("feature_layers"):
        raise SystemExit(f"feature_layers differ: {new.get('feature_layers')} vs {ck['backbone_cfg'].get('feature_layers')}")
    ck["backbone_cfg"] = {**ck["backbone_cfg"], **{k: v for k, v in new.items() if k in keep or k == "path"}}
    ck.setdefault("extra", {})["backbone_swapped_from"] = a.ckpt
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(ck, a.out)
    print(f"{a.out}: backbone {ck['backbone_cfg']['repo_id']} ({ck['backbone_cfg']['weight_quant']})")


if __name__ == "__main__":
    main()
