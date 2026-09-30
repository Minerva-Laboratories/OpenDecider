#!/usr/bin/env bash
# Wait until x-answer's public eval AND probes are finished, then run the eval-only context test.
cd "$(dirname "$0")/.."
until [ -f runs/probes-x-answer/probes.json ]; do sleep 60; done
scripts/eval_context.sh mix2-v3branch-vera x-answer
