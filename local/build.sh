#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
python3 "$repo/local/verify-source.py"
[[ "${1:-}" == --verify-only ]] && exit 0
# Run inside the pinned ROCm/Python SDK environment or an isolated build venv.
python -m pip install --require-hashes -r "$repo/local/requirements-json.lock"
cd "$repo"
exec bash build.sh
