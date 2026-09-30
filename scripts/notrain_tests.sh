#!/usr/bin/env bash
# No-training tests for the recipe update: (1) frozen 2B label-sequence baseline (knowledge ceiling check),
# (2) x2b predictions saved on every public benchmark (for per-deployment calibration-profile fitting).
cd "$(dirname "$0")/.."
P=data/public
.venv/bin/python -m eval.evaluate --baseline label_seq --backbone-config configs/backbone_2b.yaml \
    --data $P/banking77_8way.jsonl $P/pubmedqa.jsonl $P/openbookqa.jsonl $P/commonsenseqa.jsonl $P/prompt_injections.jsonl \
    --limit 1000 --n-boot 500 --name baseline2b-labelseq > runs/baseline2b-labelseq.log 2>&1
.venv/bin/python -m eval.evaluate --ckpt runs/x2b/model.pt --save-preds --n-boot 200 --limit 1000 --name preds-x2b \
    --data $P/banking77_77way.jsonl $P/banking77_8way.jsonl $P/pubmedqa.jsonl $P/openbookqa.jsonl $P/commonsenseqa.jsonl \
    $P/prompt_injections.jsonl > runs/preds-x2b.log 2>&1
