# latent_space_exploration/09n_evaluate_wav_detection.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09n_evaluate_wav_detection.py

Evaluación/detección basada en MAP Gaussiano en el espacio latente z.

Lee desde config.json (escrito por 08b):
  map_detector = {
    "model": "gaussian_map",
    "means": {sp: [..]},
    "precision": {sp: [[..]]},
    "logdet_cov": {sp: float},
    "tau": float | null,
    "meta_fit": { "chunk_seconds": float, "per_species": { sp: {"prior": float, ...}, ... } }
  }

Regla:
  score_k(z) = log N(z | mu_k, cov_k) + log prior_k
  pred = argmax_k score_k
  si tau != None y best_score < tau -> NO_DETECT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from latent_space_exploration.map_detector_core import (
    encode_wav_to_latent,
    find_project_root,
    gaussian_logpdf_from_precision,
    get_chunk_seconds_for_map,
    get_priors_from_map_meta,
    load_encoder,
    load_json,
    read_map_detector_params,
    resolve_default_config,
    resolve_default_encoder_pt,
    resolve_default_encoder_yaml,
)


def detect_species_map(
    wav_path: str | Path,
    *,
    config_path: str | Path | None = None,
    encoder_pt: str | Path | None = None,
    encoder_yaml: str | Path | None = None,
    device: str = "cpu",
    # mel defaults del proyecto
    sr: int = 48000,
    n_mels: int = 64,
    target_frames: int = 192,
    fmin: float = 150.0,
    fmax: float = 15000.0,
    hop_length: int = 384,
    n_fft: int = 2048,
) -> Tuple[bool, Optional[str], float]:
    """
    Retorna:
      detected: bool
      species: str | None
      best_score: float (score máximo incluso si NO_DETECT)
    """
    here = Path(__file__).resolve().parent
    project_root = find_project_root(here)

    wav_p = Path(wav_path).expanduser()
    if not wav_p.is_absolute():
        wav_p = (Path.cwd() / wav_p).resolve()
    if not wav_p.exists():
        raise FileNotFoundError(f"No existe WAV: {wav_p}")

    cfg_p = Path(config_path).expanduser().resolve() if config_path else resolve_default_config(project_root)
    cfg = load_json(cfg_p)

    means, precisions, logdets, tau = read_map_detector_params(cfg)
    species = sorted(set(means.keys()).intersection(precisions.keys()).intersection(logdets.keys()))
    if not species:
        raise RuntimeError("map_detector inconsistente: no hay intersección entre means/precision/logdet_cov.")

    priors = get_priors_from_map_meta(cfg, species)

    duration = get_chunk_seconds_for_map(cfg)

    dev = torch.device(device)
    enc_pt = Path(encoder_pt).expanduser().resolve() if encoder_pt else resolve_default_encoder_pt(project_root)
    enc_yaml = Path(encoder_yaml).expanduser().resolve() if encoder_yaml else resolve_default_encoder_yaml(project_root)

    encoder = load_encoder(enc_pt, enc_yaml, project_root, dev)

    z = encode_wav_to_latent(
        encoder=encoder,
        wav_path=wav_p,
        device=dev,
        sr=sr,
        duration=duration,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        n_fft=n_fft,
        target_frames=target_frames,
    )

    best_sp: Optional[str] = None
    best_score = -float("inf")

    for sp in species:
        mu = means[sp]
        prec = precisions[sp]
        ld = logdets[sp]

        if mu.shape[0] != z.shape[0]:
            continue
        if prec.shape[0] != z.shape[0] or prec.shape[1] != z.shape[0]:
            continue

        lp = float(np.log(float(priors.get(sp, 1e-12)) + 1e-12))
        s = gaussian_logpdf_from_precision(z, mu, prec, ld) + lp

        if s > best_score:
            best_score = s
            best_sp = sp

    if best_sp is None:
        return False, None, best_score

    if tau is not None and best_score < float(tau):
        return False, None, best_score

    return True, best_sp, best_score


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, type=str, help="Ruta al archivo .wav a evaluar")
    p.add_argument("--config", type=str, default=None, help="Ruta a config.json (opcional)")
    p.add_argument("--encoder-pt", type=str, default=None, help="Ruta a model.pt (opcional)")
    p.add_argument("--encoder-yaml", type=str, default=None, help="Ruta a .yaml del encoder (opcional)")
    p.add_argument("--device", type=str, default="cpu")

    # mel params (por si override)
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--target-frames", type=int, default=192)
    p.add_argument("--fmin", type=float, default=150.0)
    p.add_argument("--fmax", type=float, default=15000.0)
    p.add_argument("--hop-length", type=int, default=384)
    p.add_argument("--n-fft", type=int, default=2048)
    return p.parse_args()


def main():
    args = _parse_args()
    detected, sp, best_score = detect_species_map(
        args.wav,
        config_path=args.config,
        encoder_pt=args.encoder_pt,
        encoder_yaml=args.encoder_yaml,
        device=args.device,
        sr=args.sr,
        n_mels=args.n_mels,
        target_frames=args.target_frames,
        fmin=args.fmin,
        fmax=args.fmax,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
    )

    if detected:
        print(f"✅ DETECTADO (MAP): {sp} | best_score={best_score:.6f}")
        sys.exit(0)
    else:
        print(f"❌ NO_DETECT (MAP) | best_score={best_score:.6f}")
        sys.exit(2)


if __name__ == "__main__":
    main()