#!/usr/bin/env bash
# Build libod_llama.so against a llama.cpp build (LLAMA_DIR, default ~/llama.cpp, built with CUDA in build/).
#   deploy/llamacpp/build.sh [out_dir]          -> <out_dir>/libod_llama.so (default: deploy/llamacpp/build)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
L="${LLAMA_DIR:-$HOME/llama.cpp}"
OUT="${1:-$HERE/build}"
mkdir -p "$OUT"
g++ -O2 -std=c++17 -shared -fPIC "$HERE/od_llama.cpp" -o "$OUT/libod_llama.so" \
  -I"$L/include" -I"$L/ggml/include" -L"$L/build/bin" -lllama -lggml -lggml-base -lpthread \
  -Wl,-rpath,"$L/build/bin"
echo "$OUT/libod_llama.so"
