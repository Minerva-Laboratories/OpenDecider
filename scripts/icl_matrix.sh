#!/usr/bin/env bash
# In-context-learning test, NO training: every model x {single, unlabelled batch (json/native), labelled 4/8 (json/native)}.
cd "$(dirname "$0")/.."
D=$(ls data/public_context/*.jsonl)
.venv/bin/python -m eval.evaluate --baseline label_seq --data $D --n-boot 500 --name icl-baseline > runs/icl-baseline.log 2>&1
for r in mix-v0 mix2-v3branch-vera x-robust x-answer; do
  .venv/bin/python -m eval.evaluate --ckpt "runs/$r/model.pt" --data $D --n-boot 500 --name "icl-$r" > "runs/icl-$r.log" 2>&1
done
