#!/usr/bin/env bash
# Wait for the no-training test script (by PID), then the retrieval eval on x2b, then the ablation ladder.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 60; done
scripts/knn_x2b.sh
scripts/ladder.sh
