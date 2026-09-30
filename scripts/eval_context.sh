#!/usr/bin/env bash
# Eval-only context-learning test: single vs unlabelled batch vs labelled ICL, on held-out benchmark items.
cd "$(dirname "$0")/.."
for r in "$@"; do
  .venv/bin/python -m eval.evaluate --ckpt "runs/$r/model.pt" --data data/public_context/*.jsonl --n-boot 500 \
      --name "context-$r" > "runs/eval-context-$r.log" 2>&1
done
