#!/usr/bin/env bash
# Kill PID (and all its descendants) if system MemAvailable drops below FLOOR_GB.
# Orin unified memory: protect the OS.  usage: scripts/memwatch.sh <pid> [floor_gb=12]
pid=$1; floor=${2:-12}
tree() { local p=$1; echo "$p"; for c in $(pgrep -P "$p"); do tree "$c"; done; }
while kill -0 "$pid" 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt "$floor" ]; then
    echo "[memwatch] MemAvailable ${avail}GB < ${floor}GB: killing tree of $pid" >&2
    pids=$(tree "$pid"); kill $pids 2>/dev/null; sleep 5; kill -9 $pids 2>/dev/null; exit 1
  fi
  sleep 2
done
