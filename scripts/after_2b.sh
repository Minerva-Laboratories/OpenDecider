#!/usr/bin/env bash
# After the 2B run finishes: public benchmarks, probe battery, schema-consistent ICL matrix (no training).
cd "$(dirname "$0")/.."
r=${1:-x2b}
while [ -f "runs/$r.pid" ] && kill -0 "$(cat runs/$r.pid)" 2>/dev/null; do sleep 60; done
[ -f "runs/$r/model.pt" ] || { echo "no checkpoint for $r"; exit 1; }
scripts/eval_public.sh "$r"
scripts/probes.sh "$r"
.venv/bin/python -m eval.evaluate --ckpt "runs/$r/model.pt" --data $(ls data/public_context/*_schema.jsonl) --n-boot 500 \
    --name "icls-$r" > "runs/icls-$r.log" 2>&1
# batch-dependence probe (eval only): same item alone vs in batches with ~10/50/90% positives, two renderings
for m in "$r" x-answer; do
  for fmt in items schema; do
    .venv/bin/python -m eval.evaluate --ckpt "runs/$m/model.pt" --data "data/public_context/batchdep_$fmt.jsonl" \
        --n-boot 200 --save-preds --name "bdep-$fmt-$m" > "runs/bdep-$fmt-$m.log" 2>&1
    .venv/bin/python -m eval.batch_dependence "runs/bdep-$fmt-$m" >> runs/batch_dependence.txt 2>&1
  done
done
