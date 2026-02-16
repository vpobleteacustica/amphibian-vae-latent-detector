#!/bin/bash
set -euo pipefail

# Corre desde: modelos_VAE/latent_space_exploration

QIN="0.95"
DEVICE="cpu"
MAX_PER_CLASS="400"

GRID_ROOT="../outputs/qout_grid_20260213"
mkdir -p "${GRID_ROOT}"

for qout in 0.10 0.15 0.20 0.25
do
  echo "=============================="
  echo "Running q_out=${qout} (q_in=${QIN})"
  echo "=============================="

  outdir="${GRID_ROOT}/qout_${qout}"
  mkdir -p "${outdir}"

  # Log por corrida (clave para extraer rk)
  runlog="${outdir}/run.log"
  echo "📝 Logging to: ${runlog}"

  # Ejecutar y capturar stdout+stderr
  {
    python 08_fit_radial_detector.py \
      --root train_chunks \
      --q-in "${QIN}" \
      --q-out "${qout}" \
      --device "${DEVICE}" \
      --max-per-class "${MAX_PER_CLASS}" \
      --cache

    python 10_benchmark_folder_detection.py \
      --root val_chunks \
      --device "${DEVICE}"
  } 2>&1 | tee "${runlog}"

  # Copiar outputs del benchmark
  cp -f ../outputs/detection_benchmark/summary.txt "${outdir}/summary.txt"
  cp -f ../outputs/detection_benchmark/results.csv "${outdir}/results.csv"
  cp -f ../outputs/detection_benchmark/confusion_matrix.png "${outdir}/confusion_matrix.png"
  cp -f ../outputs/detection_benchmark/accuracy_by_class.png "${outdir}/accuracy_by_class.png"
  cp -f ../outputs/detection_benchmark/no_detect_rate_by_class.png "${outdir}/no_detect_rate_by_class.png"
  cp -f ../outputs/detection_benchmark/global_counts.png "${outdir}/global_counts.png"

  # Guardar el config tal como quedó (aunque no tenga rk, igual sirve)
  cp -f ../config.json "${outdir}/config_used.json"

  # Generar snapshot con rk parseado desde el log
  python 9105_make_config_snapshot_from_log.py \
    --log "${runlog}" \
    --q-in "${QIN}" \
    --q-out "${qout}" \
    --out "${outdir}/config_snapshot.json"

  echo "✅ Saved -> ${outdir}"
done