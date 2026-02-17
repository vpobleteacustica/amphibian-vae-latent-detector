# # Amphibian VAE Latent Detector

**Baseline implementation for José’s Thesis**

This repository contains the reproducible implementation of the latent-space
detector used in the thesis work based on:

- VAE latent embeddings
- Radial heuristic decision rule
- q_in / q_out calibration workflow

This version corresponds to the **official thesis baseline**.

---

## 1. Method Overview

The pipeline consists of:

1. Normalize dataset (RMS peak normalization)
2. Encode audio chunks into VAE latent space
3. Fit radial centroids per species
4. Calibrate q_in / q_out thresholds
5. Run detection benchmark

The decision rule is a **geometric radial heuristic** operating directly
in the learned latent space.

---

## 2. Repository Structure: 

latent_space_exploration/

00_normalize_dataset_rms.py
07_encode_wav_to_latent.py
08_fit_radial_detector.py
09n_evaluate_wav_detection.py
9200_run_qout_grid_with_snapshot.py 

---
Only source code is included.
No data, models, or outputs are stored in this repository.

---

## 3. Reproducibility Steps

### Step 1 – Normalize dataset
```bash
python latent_space_exploration/00_normalize_dataset_rms.py

---

## Version

This repository corresponds to the frozen baseline used in the thesis defense.
No methodological changes should be introduced after this point.
