#!/usr/bin/env bash
# After the long 2B run: full public benchmarks (+ saved predictions for calibration profiles), probes,
# schema ICL, retrieval ICL, batch dependence, calibration-profile cross-fit.
cd "$(dirname "$0")/.."
r=${1:-long2b}
while [ -f "runs/$r.pid" ] && kill -0 "$(cat runs/$r.pid)" 2>/dev/null; do sleep 60; done
[ -f "runs/$r/model.pt" ] || { echo "no checkpoint for $r"; exit 1; }
P=data/public; X=data/public_context
.venv/bin/python -m eval.evaluate --ckpt runs/$r/model.pt --save-preds --n-boot 1000 --name "public-$r" \
    --data $P/banking77_77way.jsonl $P/banking77_8way.jsonl $P/pubmedqa.jsonl $P/openbookqa.jsonl \
    $P/commonsenseqa.jsonl $P/prompt_injections.jsonl > runs/eval-public-$r.log 2>&1
.venv/bin/python -m eval.profile_cv runs/public-$r > runs/profile-cv-$r.log 2>&1
scripts/probes.sh "$r"
.venv/bin/python -m eval.evaluate --ckpt runs/$r/model.pt --n-boot 500 --save-preds --name "icl-$r" \
    --data $(ls $X/*_icl*_schema.jsonl $X/*_single_schema.jsonl $X/*_batch8_schema.jsonl $X/*_knn*_schema.jsonl) \
    > runs/icl-$r.log 2>&1
for fmt in items schema; do
  .venv/bin/python -m eval.evaluate --ckpt runs/$r/model.pt --data $X/batchdep_$fmt.jsonl --n-boot 200 --save-preds \
      --name "bdep-$fmt-$r" > runs/bdep-$fmt-$r.log 2>&1
  .venv/bin/python -m eval.batch_dependence runs/bdep-$fmt-$r >> runs/batch_dependence-$r.txt 2>&1
done
echo "after_long done"
