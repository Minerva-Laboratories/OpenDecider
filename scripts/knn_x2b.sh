#!/usr/bin/env bash
# Retrieved (TF-IDF nearest) labelled examples, no training, on x2b; compare with random icl4/icl8 and single.
cd "$(dirname "$0")/.."
X=data/public_context
.venv/bin/python -m eval.evaluate --ckpt runs/x2b/model.pt --n-boot 500 --limit 300 --save-preds --name knn-x2b \
    --data $X/banking77_8way_knn4_schema.jsonl $X/banking77_8way_knn8_schema.jsonl $X/prompt_injections_knn4_schema.jsonl \
    $X/prompt_injections_knn8_schema.jsonl $X/banking77_77way_knn4_schema.jsonl > runs/knn-x2b.log 2>&1
