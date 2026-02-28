#!/usr/bin/env bash
# Launch the APEX Dashboard on :8050
#
# Usage:
#   ./scripts/dashboard.sh              # default: PostgreSQL, 0.0.0.0:8050
#   ./scripts/dashboard.sh --debug      # with Dash debug/hot-reload
#   ./scripts/dashboard.sh --port 8051  # custom port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv/bin/python"

# Database URL: APEX_DATABASE_URL env > --db flag > dashboard config file
DB="${APEX_DATABASE_URL:-}"
HOST="0.0.0.0"
PORT=8050
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --db)     DB="$2";   shift 2 ;;
        --host)   HOST="$2"; shift 2 ;;
        --port)   PORT="$2"; shift 2 ;;
        --debug)  EXTRA_ARGS+=(--debug); shift ;;
        *)        EXTRA_ARGS+=("$1");    shift ;;
    esac
done

echo "Starting APEX Dashboard — http://${HOST}:${PORT}/"
if [[ -n "$DB" ]]; then
    exec "$VENV" -m apex dashboard "$DB" --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}"
else
    exec "$VENV" -m apex dashboard --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}"
fi
