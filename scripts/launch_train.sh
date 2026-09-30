#!/usr/bin/env bash
# Launch a training run detached, record the REAL python PID (runs/<name>.pid), attach the memory watchdog.
#   scripts/launch_train.sh <name> <config> [--set k=v ...]
#   stop with:  kill $(cat runs/<name>.pid)
set -e
name=$1; cfg=$2; shift 2
cd "$(dirname "$0")/.."
setsid .venv/bin/python -m opendecider.train --config "$cfg" --set train.name="$name" "$@" \
  > "runs/$name.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "runs/$name.pid"
setsid scripts/memwatch.sh "$pid" 12 > "runs/memwatch-$name.log" 2>&1 < /dev/null &
echo "launched $name pid $pid"
