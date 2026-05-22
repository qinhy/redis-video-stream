#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
FLOWER_UNAUTHENTICATED_API=true exec uv run celery -A redis_video_stream.tasks flower -b redis://127.0.0.1 --loglevel=INFO "$@"
