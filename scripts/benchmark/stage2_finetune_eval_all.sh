#!/usr/bin/env bash
# Deprecated: use run_all.sh --stage 2 instead.
# This script is kept for backward compatibility.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/run_all.sh" --stage 2 "$@"
