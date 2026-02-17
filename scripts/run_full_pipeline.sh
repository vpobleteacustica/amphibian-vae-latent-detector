#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root, no matter where user calls it from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================"
echo " Amphibian VAE Thesis Baseline Pipeline"
echo " Release: v1.0-thesis-baseline"
echo "============================================"

# 0) Ensure env file exists
if [[ ! -f "scripts/_env.sh" ]]; then
  echo "ERROR: Missing scripts/_env.sh"
  echo "Create it from scripts/_env.sh.example (if provided) or define paths manually."
  exit 1
fi

# Load environment variables (paths, device, q_in, grid, etc.)
source scripts/_env.sh

echo ""
echo "[0/5] Pre-flight checks"

# A) Check required env vars exist
: "${CHUNKS_ROOT:?ERROR: CHUNKS_ROOT is not set in scripts/_env.sh}"
: "${VAE_ENCODER_PT:?ERROR: VAE_ENCODER_PT is not set in scripts/_env.sh}"
: "${VAE_DECODER_PT:?ERROR: VAE_DECODER_PT is not set in scripts/_env.sh}"

echo "Using:"
echo "  CHUNKS_ROOT     = ${CHUNKS_ROOT}"
echo "  VAE_ENCODER_PT  = ${VAE_ENCODER_PT}"
echo "  VAE_DECODER_PT  = ${VAE_DECODER_PT}"
if [[ "${VAE_YAML:-}" != "" ]]; then
  echo "  VAE_YAML        = ${VAE_YAML}"
fi

# B) Check required scripts exist
for f in \
  scripts/01_normalize_chunks.sh \
  scripts/03_encode_latents.sh \
  scripts/04_calibrate_qout_grid.sh \
  scripts/05_make_plots.sh
do
  [[ -f "$f" ]] || { echo "ERROR: Missing script: $f"; exit 1; }
done

# C) Check chunk folders exist (raw chunks)
for d in "${CHUNKS_ROOT}/train" "${CHUNKS_ROOT}/val" "${CHUNKS_ROOT}/test"
do
  [[ -d "$d" ]] || { echo "ERROR: Missing chunk folder: $d"; exit 1; }
done

# Optional: ensure folders contain at least one WAV (prevents “runs but does nothing”)
for split in train val test; do
  if ! find "${CHUNKS_ROOT}/${split}" -type f \( -iname "*.wav" -o -iname "*.flac" \) | head -n 1 | grep -q .; then
    echo "ERROR: No audio files found in ${CHUNKS_ROOT}/${split}"
    echo "Expected .wav (or .flac). This repo assumes chunks already exist."
    exit 1
  fi
done

# D) Check pretrained VAE files exist (required for encoding)
for f in "${VAE_ENCODER_PT}" "${VAE_DECODER_PT}"
do
  [[ -f "$f" ]] || { echo "ERROR: Missing VAE weight file: $f"; exit 1; }
done

# YAML is optional (only if your code actually needs it)
if [[ "${VAE_YAML:-}" != "" ]]; then
  [[ -f "${VAE_YAML}" ]] || { echo "ERROR: Missing VAE YAML: ${VAE_YAML}"; exit 1; }
fi

echo "OK: scripts, chunks, and VAE weights found."

echo ""
echo "[1/5] Normalize chunks (RMS peak)"
bash scripts/01_normalize_chunks.sh

echo ""
echo "[2/5] Encode latents"
bash scripts/03_encode_latents.sh

echo ""
echo "[3/5] Calibrate q_out grid (includes fitting + benchmark per q_out)"
bash scripts/04_calibrate_qout_grid.sh

echo ""
echo "[4/5] Summaries + plots"
bash scripts/05_make_plots.sh

echo ""
echo "============================================"
echo " Pipeline finished successfully ✅"
echo "============================================"