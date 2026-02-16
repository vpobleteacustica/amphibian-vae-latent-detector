#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_normalize_dataset_rms.py

Crea versiones normalizadas (RMS) de train/val/test chunks.

No modifica el dataset original.
Genera:
  train_chunks_norm/
  val_chunks_norm/
  test_chunks_norm/

Normalización:
  - RMS normalization a target_rms
  - silence gate
  - clipping [-1, 1]
"""

import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm


def rms_normalize(y, target_rms=0.05, rms_min=1e-4, eps=1e-8):
    rms = np.sqrt(np.mean(y**2))

    if rms < rms_min:
        # silencio → no escalar (o podrías devolver None)
        return y, False

    y_norm = y * (target_rms / (rms + eps))
    y_norm = np.clip(y_norm, -1.0, 1.0)
    return y_norm, True


def process_folder(src_root: Path, dst_root: Path, sr=48000):
    species_dirs = [d for d in src_root.iterdir() if d.is_dir()]

    for sp_dir in species_dirs:
        dst_sp = dst_root / sp_dir.name
        dst_sp.mkdir(parents=True, exist_ok=True)

        wavs = list(sp_dir.glob("*.wav"))

        for wav in tqdm(wavs, desc=f"{src_root.name}/{sp_dir.name}"):
            y, _ = librosa.load(wav, sr=sr, mono=True)

            y_norm, ok = rms_normalize(y)

            # Guardar siempre (aunque sea silencio)
            out_path = dst_sp / wav.name
            sf.write(out_path, y_norm, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="latent_space_exploration")
    parser.add_argument("--sr", type=int, default=48000)
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()

    sets = ["train_chunks", "val_chunks", "test_chunks"]

    for s in sets:
        src = base / s
        dst = base / f"{s}_norm"

        if not src.exists():
            print(f"⚠ No existe {src}")
            continue

        print(f"\nProcesando {s} → {s}_norm")
        process_folder(src, dst, sr=args.sr)

    print("\n✅ Dataset normalizado listo.")


if __name__ == "__main__":
    main()