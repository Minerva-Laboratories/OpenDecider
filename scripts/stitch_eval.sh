#!/usr/bin/env bash
# Evaluate the stitched 9B OpenDecider (runs/x9b-stitch) on typed-decisions and the public benchmarks,
# then ensemble it with the 9B zero-shot predictions (runs/td-9b, runs/zs-9b).
cd "$(dirname "$0")/.."
P=data/public
.venv/bin/python -m eval.typed_decisions od --split test --name td-x9b-stitch --ckpt runs/x9b-stitch/model.pt
.venv/bin/python -m eval.typed_decisions od --split train --limit 100 --name td-x9b-stitch-train --ckpt runs/x9b-stitch/model.pt
.venv/bin/python -m eval.evaluate --ckpt runs/x9b-stitch/model.pt --save-preds --n-boot 1000 --name public-x9b-stitch \
    --data $P/banking77_8way.jsonl $P/prompt_injections.jsonl $P/openbookqa.jsonl $P/commonsenseqa.jsonl $P/pubmedqa.jsonl
echo "== typed-decisions: stitched 9B vs 9B zero-shot"
.venv/bin/python -m eval.typed_decisions score --a runs/td-x9b-stitch --b runs/td-9b \
    --fit-on runs/td-x9b-stitch-train,runs/td-9b-train --name td-ens-stitch
echo "== public: stitched 9B + 9B zero-shot ensemble"
.venv/bin/python -m eval.ensemble_cv --a runs/public-x9b-stitch --b runs/zs-9b --name ens-x9b-stitch-9b
echo STITCH_EVAL_DONE
