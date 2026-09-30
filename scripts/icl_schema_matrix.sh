#!/usr/bin/env bash
# Schema-consistent ICL test (NO training): single / labelled 4 / labelled 8 / unlabelled batch of 8 (empty answers).
cd "$(dirname "$0")/.."
until [ -f runs/icl-x-answer/eval_prompt_injections_single.json ]; do sleep 60; done     # after the first matrix
D=$(ls data/public_context/*_schema.jsonl)
.venv/bin/python -m eval.evaluate --baseline label_seq --data $D --n-boot 500 --name icls-baseline > runs/icls-baseline.log 2>&1
for r in mix-v0 mix2-v3branch-vera x-robust x-answer; do
  .venv/bin/python -m eval.evaluate --ckpt "runs/$r/model.pt" --data $D --n-boot 500 --name "icls-$r" > "runs/icls-$r.log" 2>&1
done
