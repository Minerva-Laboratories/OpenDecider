#!/usr/bin/env bash
# 9B pipeline on the GGUF weights: stitch the 2B head, then every evaluation the 2B has.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CK=runs/x9b-gguf-stitch/model.pt
P=data/public
mkdir -p /dev/shm/stitch
$PY scripts/stitch_backbone.py targets --ckpt runs/x2b/model.pt --out /dev/shm/stitch/targets.pt --n-states 900
$PY scripts/stitch_backbone.py fit --ckpt runs/x2b/model.pt --targets /dev/shm/stitch/targets.pt \
    --backbone-path models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_0.gguf --layers 8,16,24,final --out $CK
rm -f /dev/shm/stitch/targets.pt
echo "== typed-decisions"
$PY -m eval.typed_decisions od --split test --name td-x9b-gguf --ckpt $CK
echo "== public"
OPENDECIDER_EVAL_GPU_GB=34 $PY -m eval.evaluate --ckpt $CK --save-preds --n-boot 1000 --name public-x9b-gguf \
    --data $P/banking77_8way.jsonl $P/prompt_injections.jsonl $P/openbookqa.jsonl $P/commonsenseqa.jsonl $P/pubmedqa.jsonl
echo "== ensembles with the 9B zero-shot letter scores"
$PY -m eval.ensemble_cv --a runs/public-x9b-gguf --b runs/zs-9b --name ens-x9b-gguf-9b
$PY -m eval.tree_head --typed --sources runs/td-x2b runs/td-x9b-gguf runs/td-9b --name tree-typed-gguf
echo "== none option"
$PY scripts/train_none.py --ckpt $CK --explicit "none of the above" --eval-out runs/x9b-gguf-stitch/none_eval.json
echo "== memory / latency / quantization"
for c in int8:int8 int8:int4; do
  $PY scripts/bench_quant.py --ckpt $CK --configs $c --out runs/quant/9b_${c/:/_}.json
done
echo "== explanations"
$PY -m eval.explain_eval --ckpt $CK --n 24 --samples 4 --name explain-9b
echo EVAL_9B_DONE
