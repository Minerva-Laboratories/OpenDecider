#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for r in "$@"; do
  .venv/bin/python -m eval.probes --ckpt "runs/$r/model.pt" --items runs/probe_items.jsonl --name "probes-$r" \
      --n-perm 20 --n-repeat 50 --no-latency > "runs/probes-$r.log" 2>&1
done
