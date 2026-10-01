#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "BENCH_NEXT_DONE\|Traceback" runs/bench_next.log 2>/dev/null; do sleep 30; done
.venv/bin/python scripts/check_release.py > runs/check_release.log 2>&1
echo CHECK_RELEASE_DONE >> runs/check_release.log
