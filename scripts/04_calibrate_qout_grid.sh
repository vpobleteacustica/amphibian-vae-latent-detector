#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "==[04] Calibrate q_out grid=="

QIN="${QIN:-$QIN_DEFAULT}"
DEVICE="${DEVICE:-$DEVICE_DEFAULT}"
MAX_PER_CLASS="${MAX_PER_CLASS:-$MAX_PER_CLASS_DEFAULT}"

GRID_ROOT="${OUTPUTS_DIR}/qout_grid_$(date +%Y%m%d)"
mkdir -p "${GRID_ROOT}"

cd "${LSE_DIR}"

for qout in 0.10 0.15 0.20 0.25
do
  echo "Running q_out=${qout}"

  outdir="${GRID_ROOT}/qout_${qout}"
  mkdir -p "${outdir}"

  runlog="${outdir}/run.log"

  {
    python 08_fit_radial_detector.py \
      --root "${TRAIN_NORM}" \
      --q-in "${QIN}" \
      --q-out "${qout}" \
      --device "${DEVICE}" \
      --max-per-class "${MAX_PER_CLASS}" \
      --cache

    python 10_benchmark_folder_detection.py \
      --root "${VAL_NORM}" \
      --device "${DEVICE}"
  } 2>&1 | tee "${runlog}"

  echo "Saved to ${outdir}"
done

echo "Grid complete."
