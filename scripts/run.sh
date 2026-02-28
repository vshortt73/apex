#!/usr/bin/env bash
# Launch an APEX probe run from a config file.
#
# Usage:
#   ./scripts/run.sh configs/my_run.yaml           # foreground
#   ./scripts/run.sh configs/my_run.yaml --bg       # background (detached)
#   ./scripts/run.sh configs/my_run.yaml --bg --log  # background with log file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv/bin/python"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [--bg] [--log]"
    echo ""
    echo "Options:"
    echo "  --bg    Run in background (detached from terminal)"
    echo "  --log   Write stdout/stderr to logs/run_<timestamp>.log (implies --bg)"
    exit 1
fi

CONFIG="$1"
shift

if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG"
    exit 1
fi

BG=false
LOG=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bg)  BG=true;  shift ;;
        --log) LOG=true; BG=true; shift ;;
        *)     echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$BG" == true ]]; then
    if [[ "$LOG" == true ]]; then
        mkdir -p "$PROJECT_ROOT/logs"
        LOGFILE="$PROJECT_ROOT/logs/run_$(date +%Y%m%d_%H%M%S).log"
        echo "Launching in background — logging to $LOGFILE"
        nohup "$VENV" -m apex run "$CONFIG" > "$LOGFILE" 2>&1 &
    else
        echo "Launching in background (no log file)"
        nohup "$VENV" -m apex run "$CONFIG" > /dev/null 2>&1 &
    fi
    PID=$!
    echo "PID: $PID"
    echo "$PID" > "$PROJECT_ROOT/.last_run_pid"
else
    exec "$VENV" -m apex run "$CONFIG"
fi
