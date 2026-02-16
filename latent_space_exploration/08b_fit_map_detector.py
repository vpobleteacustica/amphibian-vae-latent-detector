# latent_space_exploration/08b_fit_map_detector.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08b_fit_map_detector.py

Ajusta un detector MAP (Maximum A Posteriori) Gaussiano en el espacio latente z.

Modelo:
  p(z | k) = N(mu_k, Sigma_k)
  score_k(z) = log p(z|k) + log pi_k
  pred(z) = argmax_k score_k(z)

Este script SOLO fitea y guarda parámetros del modelo en config.json.
El rechazo (NO_DETECT) se implementa vía tau:
  si max_k score_k(z) < tau  -> NO_DETECT

Se guarda en config.json:
  map_detector = {
    "model": "gaussian_map",
    "cov_type": "lda" | "qda",
    "cov_structure": "full" | "diag",
    "priors": "empirical" | "uniform",
    "means": {sp: [..]},
    "cov": {sp: [[..]]},
    "precision": {sp: [[..]]},
    "logdet_cov": {sp: float},
    "tau": float | null,
    "meta_fit": {...}
  }
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from latent_space_exploration.map_detector_core import (
    encode_wav_to_latent,
    find_project_root,
    get_chunk_seconds_for_map,
    inv_and_logdet,
    load_encoder,
    load_json,
    resolve_default_config,
    resolve_default_encoder_pt,
    resolve_default_encoder_yaml,
    save_json,
    summarize_1d,
)


def estimate_cov(Z: np.ndarray, eps: float, shrink: float, cov_structure: str) -> np.ndarray:
    """
    Estima covarianza y la regulariza.
    - eps: agrega eps*I (siempre)
    - shrink: cov <- (1-shrink)*cov + shrink*I*mean(diag(cov))
    - cov_structure: "full" o "diag"
    """
    n, d = Z.shape
    if n < 2:
        cov = np.eye(d, dtype=np.float32)
    else:
        cov = np.cov(Z, rowvar=False, bias=False).astype(np.float32)

    if cov_structure == "diag":
        cov = np.diag(np.diag(cov)).astype(np.float32)

    if shrink > 0:
        avg_var = float(np.mean(np.diag(cov))) if d > 0 else 1.0
        cov = (1.0 - shrink) * cov + shrink * (avg_var * np.eye(d, dtype=np.float32))

    cov = cov + (eps * np.eye(d, dtype=np.float32))
    return cov.astype(np.float32)


def gaussian_logpdf(z: np.ndarray, mu: np.ndarray, prec: np.ndarray, logdet_cov: float) -> float:
    d = z.shape[0]
    diff = (z - mu).astype(np.float32)
    quad = float(diff.T @ prec @ diff)
    return -0.5 * (quad + logdet_cov + d * np.log(2.0 * np.pi))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.json")
    p.add_argument("--root", type=str, required=True, help="Carpeta con subcarpetas por especie (train_chunks/...)")
    p.add_argument("--device", type=str, default="cpu")

    # mel params
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--target-frames", type=int, default=192)
    p.add_argument("--fmin", type=float, default=150.0)
    p.add_argument("--fmax", type=float, default=15000.0)
    p.add_argument("--hop-length", type=int, default=384)
    p.add_argument("--n-fft", type=int, default=2048)

    p.add_argument("--encoder-pt", type=str, default=None)
    p.add_argument("--encoder-yaml", type=str, default=None)

    # dataset control
    p.add_argument("--max-per-class", type=int, default=0, help="0 = usar todos; si >0, samplea hasta este N por especie")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--cache", action="store_true", help="Guardar/cargar latentes Z por especie en latent_space_exploration/cache_npz/")

    # MAP params
    p.add_argument("--cov-type", type=str, default="lda", choices=["lda", "qda"])
    p.add_argument("--cov-structure", type=str, default="full", choices=["full", "diag"])
    p.add_argument("--priors", type=str, default="empirical", choices=["empirical", "uniform"])
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--shrink", type=float, default=0.0)

    # tau desde cuantíl de score_true_class en train
    p.add_argument("--set-tau-q", type=float, default=None, help="Ej: 0.01 => tau=quantile(scores_true,0.01)")
    return p.parse_args()


def main():
    args = parse_args()

    if not (0.0 <= args.shrink <= 1.0):
        raise SystemExit("❌ --shrink debe estar en [0,1].")
    if args.set_tau_q is not None and not (0.0 < float(args.set_tau_q) < 1.0):
        raise SystemExit("❌ --set-tau-q debe estar en (0,1).")

    random.seed(args.seed)
    np.random.seed(args.seed)

    here = Path(__file__).resolve()
    project_root = find_project_root(here.parent)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (project_root / cfg_path).resolve()

    cfg = load_json(cfg_path)

    species_list = cfg.get("species", None)
    if not isinstance(species_list, list) or not all(isinstance(s, str) for s in species_list):
        raise SystemExit("❌ config.json debe tener un campo 'species' (lista de strings).")

    # chunks root robusto
    root_in = Path(args.root).expanduser()
    candidates = [root_in] if root_in.is_absolute() else [
        (Path.cwd() / root_in),
        (project_root / root_in),
        (project_root / "latent_space_exploration" / root_in),
    ]
    chunks_dir = None
    for cand in candidates:
        cand = cand.resolve()
        if cand.exists() and cand.is_dir():
            chunks_dir = cand
            break
    if chunks_dir is None:
        msg = "❌ No existe chunks_dir. Probé:\n" + "\n".join(f"   - {c.resolve()}" for c in candidates)
        raise SystemExit(msg)

    # duración: la del cfg global (chunk_seconds) para codificar train
    try:
        chunk_seconds = float(cfg.get("chunk_seconds", 5.0))
    except Exception:
        chunk_seconds = 5.0

    device = torch.device(args.device)
    encoder_pt = Path(args.encoder_pt).expanduser().resolve() if args.encoder_pt else resolve_default_encoder_pt(project_root)
    encoder_yaml = Path(args.encoder_yaml).expanduser().resolve() if args.encoder_yaml else resolve_default_encoder_yaml(project_root)

    cache_dir = (project_root / "latent_space_exploration" / "cache_npz").resolve()
    if args.cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"📌 Project root: {project_root}")
    print(f"🧾 Config: {cfg_path}")
    print(f"📁 Chunks dir: {chunks_dir}")
    print(f"🖥️ Device: {device}")
    print(f"🎯 cov_type={args.cov_type} | cov_structure={args.cov_structure} | priors={args.priors} | eps={args.eps} | shrink={args.shrink}")
    print(f"🎯 max_per_class={args.max_per_class} | cache={args.cache}\n")

    encoder = load_encoder(encoder_pt, encoder_yaml, project_root, device)

    # encode por especie (con cache opcional)
    Z_by_species: Dict[str, np.ndarray] = {}
    n_failed_by_species: Dict[str, int] = {}
    used_by_species: Dict[str, int] = {}

    for sp in species_list:
        sp_dir = (chunks_dir / sp).resolve()
        if not sp_dir.exists():
            print(f"⚠️ {sp}: carpeta no existe (se omite): {sp_dir}")
            continue

        cache_path = cache_dir / f"Z_{chunks_dir.name}_{sp}.npz"
        if args.cache and cache_path.exists():
            data = np.load(cache_path)
            Zm = data["Z"].astype(np.float32)
            Z_by_species[sp] = Zm
            n_failed_by_species[sp] = int(data.get("failed", 0)) if "failed" in data else 0
            used_by_species[sp] = int(Zm.shape[0])
            print(f"🧊 {sp}: cargado cache {cache_path.name} -> N={Zm.shape[0]}")
            continue

        wavs = sorted(sp_dir.glob("*.wav"))
        if len(wavs) == 0:
            print(f"⚠️ {sp}: sin wavs (se omite).")
            continue

        if args.max_per_class and len(wavs) > args.max_per_class:
            wavs = random.sample(wavs, args.max_per_class)

        Z: List[np.ndarray] = []
        n_fail = 0
        for wav in wavs:
            try:
                z = encode_wav_to_latent(
                    encoder=encoder,
                    wav_path=wav,
                    device=device,
                    sr=args.sr,
                    duration=chunk_seconds,
                    n_mels=args.n_mels,
                    fmin=args.fmin,
                    fmax=args.fmax,
                    hop_length=args.hop_length,
                    n_fft=args.n_fft,
                    target_frames=args.target_frames,
                )
                Z.append(z)
            except Exception as e:
                n_fail += 1
                print(f"⚠️ {sp}: fallo {wav.name}: {e}")

        if len(Z) == 0:
            print(f"❌ {sp}: no se pudo codificar nada (se omite).")
            continue

        Zm = np.stack(Z, axis=0).astype(np.float32)
        Z_by_species[sp] = Zm
        n_failed_by_species[sp] = n_fail
        used_by_species[sp] = int(Zm.shape[0])
        print(f"🧪 {sp}: encoded N={Zm.shape[0]} (failed={n_fail})")

        if args.cache:
            np.savez_compressed(cache_path, Z=Zm, failed=n_fail, root=str(chunks_dir))
            print(f"   ↳ guardado cache: {cache_path.name}")

    if not Z_by_species:
        raise SystemExit("❌ No se codificó ninguna especie. Revisa root y/o pipeline.")

    species_present = sorted(Z_by_species.keys())
    K = len(species_present)

    # priors
    if args.priors == "uniform":
        priors = {sp: 1.0 / K for sp in species_present}
    else:
        total = float(sum(Z_by_species[sp].shape[0] for sp in species_present))
        priors = {sp: float(Z_by_species[sp].shape[0]) / total for sp in species_present}

    # means
    means: Dict[str, np.ndarray] = {sp: np.mean(Z_by_species[sp], axis=0).astype(np.float32) for sp in species_present}

    # covariances / precisions / logdets
    covs: Dict[str, np.ndarray] = {}
    precs: Dict[str, np.ndarray] = {}
    logdets: Dict[str, float] = {}

    if args.cov_type == "lda":
        all_centered = [Z_by_species[sp] - means[sp][None, :] for sp in species_present]
        Zc = np.concatenate(all_centered, axis=0)
        cov_shared = estimate_cov(Zc, eps=float(args.eps), shrink=float(args.shrink), cov_structure=args.cov_structure)
        prec_shared, logdet_shared = inv_and_logdet(cov_shared)
        for sp in species_present:
            covs[sp] = cov_shared
            precs[sp] = prec_shared
            logdets[sp] = logdet_shared
    else:
        for sp in species_present:
            Zc = Z_by_species[sp] - means[sp][None, :]
            cov_k = estimate_cov(Zc, eps=float(args.eps), shrink=float(args.shrink), cov_structure=args.cov_structure)
            prec_k, logdet_k = inv_and_logdet(cov_k)
            covs[sp] = cov_k
            precs[sp] = prec_k
            logdets[sp] = logdet_k

    # score summaries + tau opcional (sobre score de la clase verdadera)
    scores_true: List[float] = []
    per_species_meta: Dict[str, Any] = {}

    for sp in species_present:
        Z = Z_by_species[sp]
        mu = means[sp]
        prec = precs[sp]
        ld = logdets[sp]
        lp = float(np.log(priors[sp] + 1e-12))
        s = np.array([gaussian_logpdf(z, mu, prec, ld) + lp for z in Z], dtype=np.float64)
        scores_true.extend(list(s))

        per_species_meta[sp] = {
            "N": int(Z.shape[0]),
            "failed": int(n_failed_by_species.get(sp, 0)),
            "used": int(used_by_species.get(sp, Z.shape[0])),
            "prior": float(priors[sp]),
            "score_true_summary": summarize_1d(s.astype(np.float32)),
        }

    scores_true_arr = np.array(scores_true, dtype=np.float64)
    tau = None
    if args.set_tau_q is not None:
        tau = float(np.quantile(scores_true_arr, float(args.set_tau_q)))
        print(f"\n✅ tau fijado desde train: tau = quantile(score_true_class, q={float(args.set_tau_q)}) = {tau:.6f}")

    # write config
    cfg["map_detector"] = {
        "model": "gaussian_map",
        "cov_type": str(args.cov_type),
        "cov_structure": str(args.cov_structure),
        "priors": str(args.priors),
        "means": {sp: means[sp].astype(float).tolist() for sp in species_present},
        "cov": {sp: covs[sp].astype(float).tolist() for sp in species_present},
        "precision": {sp: precs[sp].astype(float).tolist() for sp in species_present},
        "logdet_cov": {sp: float(logdets[sp]) for sp in species_present},
        "tau": tau,
        "meta_fit": {
            "chunks_dir": str(chunks_dir),
            "chunks_name": chunks_dir.name,
            "chunk_seconds": float(chunk_seconds),
            "sr": int(args.sr),
            "n_mels": int(args.n_mels),
            "target_frames": int(args.target_frames),
            "fmin": float(args.fmin),
            "fmax": float(args.fmax),
            "hop_length": int(args.hop_length),
            "n_fft": int(args.n_fft),
            "max_per_class": int(args.max_per_class),
            "seed": int(args.seed),
            "eps": float(args.eps),
            "shrink": float(args.shrink),
            "tau_from_train_quantile": float(args.set_tau_q) if args.set_tau_q is not None else None,
            "score_true_global_summary": summarize_1d(scores_true_arr.astype(np.float32)),
            "per_species": per_species_meta,
        },
    }

    backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    shutil.copy2(cfg_path, backup)
    save_json(cfg_path, cfg)

    print(f"\n💾 Guardado en: {cfg_path}")
    print(f"🗂️ Backup: {backup}")
    print("\n✅ MAP detector fit listo. (NO_DETECT se decide con tau en 09n/10b.)")


if __name__ == "__main__":
    main()