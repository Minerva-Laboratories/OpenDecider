#!/usr/bin/env bash
# OpenDecider on the local network (run by the opendecider user service; see scripts/server.sh).
# Settings come from ~/.config/opendecider/server.env (created by `scripts/server.sh install`):
#   MODEL     checkpoint directory or HF repo id            (default checkpoints/opendecider-9b)
#   BACKBONE  backbone variant, empty = the checkpoint's default
#   MAX_BATCH most concurrent requests batched together (default 16; 1 turns batching off)
#   PORT      (default 8000)   HOST (default 0.0.0.0)
#   LOG_DIR   request + feedback log                        (default runs/server)
set -e
cd "$(dirname "$0")/.."
MODEL=${MODEL:-checkpoints/opendecider-9b}
MAX_BATCH=${MAX_BATCH:-16}
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
LOG_DIR=${LOG_DIR:-runs/server}
exec .venv/bin/python -m opendecider.serve --model "$MODEL" ${BACKBONE:+--backbone "$BACKBONE"} \
  --host "$HOST" --port "$PORT" --max-batch "$MAX_BATCH" --warmup \
  --api-key-file "$HOME/.config/opendecider/api_key" --log-dir "$LOG_DIR"
