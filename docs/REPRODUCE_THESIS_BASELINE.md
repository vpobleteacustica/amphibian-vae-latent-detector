# Reproduce Thesis Baseline (v1.0-thesis-baseline)

This document explains how to reproduce the **official frozen thesis baseline (v1.0-thesis-baseline)**:

Chunk normalization → latent encoding → radial detector fitting → q_in/q_out calibration → benchmark + plots.

> **⚠️ IMPORTANT**
> This repository does not include:
> - Audio datasets
> - Pretrained VAE weights
> - Chunk generation scripts
>
> These assets must be provided locally before running the pipeline.

---

## 0) Requirements

- Python 3.10+
- A working environment with the dependencies used in the thesis experiments
- Pretrained VAE encoder/decoder weights (see next section)
- Pre-generated audio chunks (see Section 2)

---

## 1) VAE Encoder Weights (required)

This thesis baseline relies on a **pretrained VAE encoder/decoder**.

The VAE training procedure is **not part of this repository**.
This repository assumes that a pretrained VAE model is already available locally.

Before running the pipeline, ensure the following files exist locally:

```text
modelos_VAE/models/
bird_net_vae_audio_splitted_encoder_v0/model.pt
bird_net_vae_audio_splitted_decoder_v0/model.pt
bird_net_vae_audio_splitted_encoder_v0/bird_net_vae_audio_splitted.yaml   (if required)
```

These files are **not included** in this repository and must be provided separately.

---

## 2) Local Data – Pre-generated Chunks (required)

This baseline operates on **audio chunks that have already been generated**.

> **⚠️ IMPORTANT**
> Raw WAV-to-chunk generation is **not included** in this repository.

Expected local structure (example):

```text
data/
└── chunks/
    ├── train/
    ├── val/
    └── test/
```

The `train/`, `val/`, and `test/` folders must contain the audio chunks used for:

- Latent encoding
- Radial centroid fitting
- q_in/q_out calibration
- Detection benchmarking

### What Step 1 does

Step 1 of the pipeline (`01_normalize_chunks.sh`) performs RMS peak normalization on the existing `train/`, `val/`, and `test/` chunk folders.

It generates:

```text
train_norm/
val_norm/
test_norm/
```

These normalized folders are then used for latent encoding.
---

## 3) Run the full pipeline

Make sure the environment variables and paths are correctly set in `scripts/_env.sh` before running.

From the repository root:

```bash
bash scripts/run_full_pipeline.sh
```

---

## Version

This document describes the frozen thesis baseline corresponding to release:

v1.0-thesis-baseline

No methodological changes should be introduced when reproducing results for thesis comparison.
