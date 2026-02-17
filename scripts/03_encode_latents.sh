#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "==[03] Encode latents=="
cd "${LSE_DIR}"
python 07_encode_wav_to_latent.py --root "${TRAIN_NORM}"
python 07_encode_wav_to_latent.py --root "${VAL_NORM}"
python 07_encode_wav_to_latent.py --root "${TEST_NORM}"
echo "Done."
