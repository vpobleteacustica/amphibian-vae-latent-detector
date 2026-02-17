#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "==[01] Normalize dataset (RMS peak)=="
cd "${LSE_DIR}"
python 00_normalize_dataset_rms.py
echo "Done."
