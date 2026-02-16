#!/bin/bash
set -e

for qout in 0.10 0.15 0.20 0.25
do
  echo "=============================="
  echo "Running q_out=${qout} (q_in=0.95)"
  echo "=============================="

  python 08_fit_radial_detector.py \
    --root train_chunks \
    --q-in 0.95 \
    --q-out ${qout} \
    --device cpu \
    --max-per-class 400 \
    --cache

  python 10_benchmark_folder_detection.py \
    --root val_chunks \
    --device cpu

  outdir="../outputs/qout_grid_20260213/qout_${qout}"
  mkdir -p "${outdir}"

  cp -f ../outputs/detection_benchmark/summary.txt "${outdir}/summary.txt"
  cp -f ../outputs/detection_benchmark/results.csv "${outdir}/results.csv"
  cp -f ../outputs/detection_benchmark/confusion_matrix.png "${outdir}/confusion_matrix.png"
  cp -f ../outputs/detection_benchmark/accuracy_by_class.png "${outdir}/accuracy_by_class.png"
  cp -f ../outputs/detection_benchmark/no_detect_rate_by_class.png "${outdir}/no_detect_rate_by_class.png"
  cp -f ../outputs/detection_benchmark/global_counts.png "${outdir}/global_counts.png"

  cp -f ../config.json "${outdir}/config_used.json"

  echo "Saved -> ${outdir}"
done
