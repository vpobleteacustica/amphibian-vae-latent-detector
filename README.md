# Amphibian VAE Latent Detector

**Official baseline implementation for Jose's Thesis (Computer Engineering degree)**

This repository contains the reproducible implementation of the latent-space
detector used in the thesis work based on:

- VAE latent embeddings  
- Geometric radial decision rule  
- q_in / q_out calibration workflow  

This version corresponds to the **official frozen thesis baseline (v1.0-thesis-baseline)**.

---

## 1. Method Overview

The complete detection pipeline consists of:

1. Dataset normalization (RMS peak normalization)  
2. Latent encoding of audio chunks using a trained VAE  
3. Radial centroid fitting per species in latent space  
4. q_in / q_out calibration for in-distribution vs. out-of-distribution control  
5. Detection benchmarking on test data  

The decision rule is a **geometric radial heuristic** operating directly in the learned latent space.

No probabilistic MAP models or discriminative classifiers are included in this baseline version.

---

## 2. Repository Structure

```text
latent_space_exploration/
│
├── 00_normalize_dataset_rms.py
│   RMS peak normalization of train/val/test datasets
│
├── 06_print_latent_coords.py
│   Utility for inspecting latent embeddings
│
├── 07_encode_wav_to_latent.py
│   Encode audio chunks into VAE latent space
│
├── 08_fit_radial_detector.py
│   Fit species centroids and radial statistics
│
├── 09n_evaluate_wav_detection.py
│   Evaluate single WAV files using calibrated thresholds
│
├── 10b_benchmark_folder_detection_map.py
│   Folder-level detection benchmark
│
├── 9200_run_qout_grid_with_snapshot.py
│   Grid search for q_out calibration
│
└── map_detector_core.py
    Core utilities for detector logic and configuration handling

Important
• Only source code is included.
• No datasets, trained models, or outputs are stored in this repository.
• External assets (VAE encoder weights, YAML configs) must be provided separately.

## 3. Reproducibility

To reproduce the thesis baseline:
1. Normalize dataset
2. Encode train chunks
3. Fit radial detector
4. Calibrate q thresholds
5. Run detection benchmark

The exact calibration procedure is documented in the thesis and associated figures.

## Citation

If you use this implementation, please cite:

Aillapi, J. (2026).
VAE-based latent radial detector for amphibian acoustic classification.
Thesis in Computer Engineering.
