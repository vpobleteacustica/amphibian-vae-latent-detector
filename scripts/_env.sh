#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LSE_DIR="${REPO_ROOT}/latent_space_exploration"

DATA_DIR="${REPO_ROOT}/data"
CHUNKS_DIR="${DATA_DIR}/chunks"

TRAIN_CHUNKS="${CHUNKS_DIR}/train"
VAL_CHUNKS="${CHUNKS_DIR}/val"
TEST_CHUNKS="${CHUNKS_DIR}/test"

TRAIN_NORM="${CHUNKS_DIR}/train_norm"
VAL_NORM="${CHUNKS_DIR}/val_norm"
TEST_NORM="${CHUNKS_DIR}/test_norm"

MODELS_DIR="${REPO_ROOT}/models"
OUTPUTS_DIR="${REPO_ROOT}/outputs"

QIN_DEFAULT="0.95"
MAX_PER_CLASS_DEFAULT="400"
DEVICE_DEFAULT="cpu"
