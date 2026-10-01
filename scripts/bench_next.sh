#!/usr/bin/env bash
# Follow-up deploy experiments: kernel microbenchmark, then the question-cache row path (exact) on the 2B.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/bench_gemm.py
for c in int8:int8:qcache w8:int8:qcache; do
  $PY scripts/bench_quant.py --ckpt runs/x2b/model.pt --configs $c --out runs/deploy/2b_${c//:/_}.json
done
$PY scripts/bench_quant.py --ckpt runs/x9b-gguf-stitch/model.pt --configs int8:int8:qcache --out runs/deploy/9b_int8_int8_qcache.json
echo BENCH_NEXT_DONE
