#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PIDS=()

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup INT TERM EXIT

start_script() {
  local script_path="$1"
  "$script_path" &
  PIDS+=("$!")
}

start_script "$SCRIPT_DIR/scripts/linux/redis.sh"
start_script "$SCRIPT_DIR/scripts/linux/uvicorn.sh"
start_script "$SCRIPT_DIR/scripts/linux/celery.sh"
start_script "$SCRIPT_DIR/scripts/linux/flower.sh"
# start_script "$SCRIPT_DIR/scripts/linux/gradio.sh"
start_script "$SCRIPT_DIR/scripts/linux/example_pipeline.sh"

wait
