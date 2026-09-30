#!/usr/bin/env bash
# Deploy-time benchmark: quantized weights on efficient kernels, CUDA graphs, one fresh process per config.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for c in int8:int8 w8:int8 w8:int8:graphs w8:int4:graphs int8:int4:graphs nf4:int4:graphs w4:int4:graphs; do
  $PY scripts/bench_quant.py --ckpt runs/x2b/model.pt --configs $c --out runs/deploy/2b_${c//:/_}.json
done
for c in int8:int8:graphs w8:int8:graphs w8:int4:graphs; do
  $PY scripts/bench_quant.py --ckpt runs/x9b-gguf-stitch/model.pt --configs $c --out runs/deploy/9b_${c//:/_}.json
done
echo BENCH_DEPLOY_DONE
