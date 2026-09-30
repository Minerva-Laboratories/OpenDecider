#!/usr/bin/env bash
# Evaluate checkpoints on the held-out public benchmarks, one after another (each takes the GPU lock).
#   scripts/eval_public.sh run_name [run_name ...]
cd "$(dirname "$0")/.."
D="data/public/banking77_77way.jsonl data/public/banking77_8way.jsonl data/public/pubmedqa.jsonl data/public/openbookqa.jsonl data/public/commonsenseqa.jsonl data/public/prompt_injections.jsonl"
for r in "$@"; do
  while [ -f "runs/$r.pid" ] && kill -0 "$(cat runs/$r.pid)" 2>/dev/null; do sleep 30; done   # wait for training
  .venv/bin/python -m eval.evaluate --ckpt "runs/$r/model.pt" --data $D --temperature-from data/synthetic/calib.jsonl \
      --limit 1000 --name "public-$r" > "runs/eval-public-$r.log" 2>&1
done
