#!/usr/bin/env bash
# Start the ToxFox OCR FastAPI service locally (CPU).
# Usage: ./run_api.sh   (logs to /tmp/toxfox_api.log, serves on http://127.0.0.1:8502)
set -euo pipefail
cd "$(dirname "$0")"

# Bound thread pools so EasyOCR/torch stay within modest RAM on CPU-only hosts.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false

# Bind address. Defaults to 0.0.0.0 (all interfaces) so the service is reachable over the
# network without hardcoding any address in this repo. To bind to one specific IP only,
# run e.g.  HOST=<your-ip> ./run_api.sh  (or set HOST in .env).
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8502}"

PY="${PY:-.venv/bin/python}"
LOG="${LOG:-/tmp/toxfox_api.log}"

echo "Starting ToxFox OCR API (python -m zug_toxfox) on http://$HOST:$PORT -> $LOG"
exec "$PY" -u -m zug_toxfox
