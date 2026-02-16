#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_wav_detection.py

Exporta:
  - detect_species(wav_path, ...) -> (detected: bool, species: Optional[str])

Qué hace:
  1) Lee config.json (por defecto en el project root) y toma:
       radial_detector.centroids
       radial_detector.thresholds
       chunk_seconds
  2) Convierte el WAV a mel-espectrogram (pipeline validado)
  3) Obtiene vector latente con el encoder del VAE (igual que 07)
  4) Evalúa pertenencia: ||z - mu_k|| <= r_k
  5) Si hay empate (2+ especies aceptadas), decide por prioridad:
       Batrachyla_leptopus > Batrachyla_taeniata > Calyptocephalella_gayi > Pleurodema_thaul

Requisitos (por defecto, desde el project root):
  - ./config.json
  - ./downloaded_models/bird_net_vae_audio_splitted_encoder_v0/model.pt
  - ./downloaded_models/bird_net_vae_audio_splitted_encoder_v0/bird_net_vae_audio_splitted.yaml

Uso como módulo:
  from latent_space_exploration.evaluate_wav_detection import detect_species
  detected, sp = detect_species("mi_audio.wav")

Uso como script:
  python3 latent_space_exploration/evaluate_wav_detection.py --wav path.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch

try:
    import librosa
except Exception as e:
    raise SystemExit(f"❌ Falta librosa. Instala: pip install librosa\nDetalle: {e}")

try:
    from omegaconf import OmegaConf  # type: ignore
    from hydra.utils import instantiate  # type: ignore
except Exception:
    OmegaConf = None  # type: ignore
    instantiate = None  # type: ignore


# ---------------------------------------------------------------------
# Prioridad (desempate)
# ---------------------------------------------------------------------
PRIORITY_ORDER: List[str] = [
    "Batrachyla_leptopus",
    "Batrachyla_taeniata",
    "Calyptocephalella_gayi",
    "Pleurodema_thaul",
]


# ---------------------------------------------------------------------
# Project root / default paths
# ---------------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "downloaded_models").exists() and (cur / "latent_space_exploration").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def resolve_default_config(project_root: Path) -> Path:
    p = project_root / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"No encontré config.json en: {p}")
    return p


def resolve_default_encoder_pt(project_root: Path) -> Path:
    p = project_root / "downloaded_models" / "bird_net_vae_audio_splitted_encoder_v0" / "model.pt"
    if not p.exists():
        raise FileNotFoundError(f"No encontré encoder .pt en: {p}")
    return p


def resolve_default_encoder_yaml(project_root: Path) -> Path:
    p = project_root / "downloaded_models" / "bird_net_vae_audio_splitted_encoder_v0" / "bird_net_vae_audio_splitted.yaml"
    if not p.exists():
        raise FileNotFoundError(f"No encontré encoder YAML en: {p}")
    return p


# ---------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------
def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("config.json no es un objeto JSON (dict).")
    return obj


def get_detector_from_config(cfg: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], float]:
    """
    Retorna:
      centroids: dict especie -> np.array shape [D]
      thresholds: dict especie -> float
      chunk_seconds: float
    """
    rd = cfg.get("radial_detector", None)
    if not isinstance(rd, dict):
        raise ValueError("config.json no contiene radial_detector (dict). Ejecuta antes 08_fit_radial_detector.py")

    cent = rd.get("centroids", None)
    thr = rd.get("thresholds", None)

    if not isinstance(cent, dict) or not isinstance(thr, dict):
        raise ValueError("radial_detector debe contener 'centroids' y 'thresholds' como dicts.")

    centroids: Dict[str, np.ndarray] = {}
    thresholds: Dict[str, float] = {}

    for sp, vec in cent.items():
        if isinstance(sp, str) and isinstance(vec, list) and len(vec) > 0:
            centroids[sp] = np.array(vec, dtype=np.float32)

    for sp, v in thr.items():
        if isinstance(sp, str):
            thresholds[sp] = float(v)

    if not centroids or not thresholds:
        raise ValueError("centroids/thresholds vacíos o mal formateados en config.json.")

    try:
        chunk_seconds = float(cfg.get("chunk_seconds", 5.0))
    except Exception:
        chunk_seconds = 5.0

    return centroids, thresholds, chunk_seconds


# ---------------------------------------------------------------------
# Encoder loading (Hydra)
# ---------------------------------------------------------------------
def load_yaml_cfg(cfg_path: Path) -> Dict[str, Any]:
    if OmegaConf is None:
        raise RuntimeError("Falta omegaconf/hydra-core. Instala: pip install omegaconf hydra-core")
    cfg_obj = OmegaConf.load(str(cfg_path))  # type: ignore
    cfg = OmegaConf.to_container(cfg_obj, resolve=False)  # type: ignore
    if not isinstance(cfg, dict):
        raise RuntimeError("No pude convertir YAML a dict.")
    return cfg


def pick_encoder_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    enc = cfg.get("encoder", None)
    if isinstance(enc, dict) and "_target_" in enc:
        return enc
    raise RuntimeError("El YAML no contiene un bloque 'encoder:' con _target_.")


def split_model_and_state(ckpt: Any):
    if isinstance(ckpt, torch.nn.Module):
        return ckpt, ckpt.state_dict()
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return None, ckpt["state_dict"]
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return None, ckpt
    return None, None


def build_nn_module(obj: Any) -> torch.nn.Module:
    """
    En soundscape_vae, instantiate(enc_cfg) suele devolver un factory callable (_BirdNet).
    Hay que llamar factory() para obtener nn.Module real.
    """
    if isinstance(obj, torch.nn.Module):
        return obj
    if callable(obj):
        m = obj()
        if isinstance(m, torch.nn.Module):
            return m
        raise RuntimeError(f"Llamé factory() pero no devolvió nn.Module: {type(m)}")
    raise RuntimeError(f"No pude construir nn.Module desde: {type(obj)}")


def load_encoder(encoder_pt: Path, encoder_yaml: Path, project_root: Path, device: torch.device) -> torch.nn.Module:
    # asegurar importabilidad del proyecto/repo si hace falta
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    ckpt = torch.load(str(encoder_pt), map_location=str(device))
    model_or_none, state = split_model_and_state(ckpt)

    if model_or_none is not None:
        module = build_nn_module(model_or_none)
        return module.to(device).eval()

    if instantiate is None:
        raise RuntimeError("hydra-core no disponible (instantiate).")

    if state is None:
        raise RuntimeError("No encontré state_dict en el checkpoint.")

    cfg = load_yaml_cfg(encoder_yaml)
    enc_cfg = pick_encoder_cfg(cfg)

    factory = instantiate(enc_cfg)  # type: ignore
    module = build_nn_module(factory)
    module.load_state_dict(state, strict=False)

    return module.to(device).eval()


# ---------------------------------------------------------------------
# Audio -> Mel (pipeline validado)
# ---------------------------------------------------------------------
def crop_or_pad_time(mel: np.ndarray, target_frames: int) -> np.ndarray:
    # mel: [M, T]
    _, T = mel.shape
    if T == target_frames:
        return mel
    if T > target_frames:
        start = (T - target_frames) // 2
        return mel[:, start:start + target_frames]
    pad_total = target_frames - T
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(mel, ((0, 0), (pad_left, pad_right)), mode="constant")


def wav_to_mel(
    wav_path: Path,
    sr: int,
    duration: float,
    n_mels: int,
    fmin: float,
    fmax: float,
    hop_length: int,
    n_fft: int,
    target_frames: int,
) -> torch.Tensor:
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

    # forzar duración fija
    if duration > 0:
        target_len = int(sr * duration)
        if y.shape[0] < target_len:
            y = np.pad(y, (0, target_len - y.shape[0]), mode="constant")
        else:
            y = y[:target_len]

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # standardize global
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)

    # forzar frames
    S_db = crop_or_pad_time(S_db, target_frames=target_frames)

    return torch.tensor(S_db, dtype=torch.float32)  # [M,T]


@torch.no_grad()
def encode_wav_to_latent(
    encoder: torch.nn.Module,
    wav_path: Path,
    device: torch.device,
    *,
    sr: int,
    duration: float,
    n_mels: int,
    fmin: float,
    fmax: float,
    hop_length: int,
    n_fft: int,
    target_frames: int,
) -> np.ndarray:
    mel = wav_to_mel(
        wav_path=wav_path,
        sr=sr,
        duration=duration,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        n_fft=n_fft,
        target_frames=target_frames,
    )

    # CLAVE: encoder espera [B,1,T,M]
    x = mel.T.unsqueeze(0).unsqueeze(0).to(device)

    out = encoder(x)

    # interpretar salida
    if isinstance(out, torch.Tensor):
        t = out
    elif isinstance(out, (list, tuple)):
        t = next((z for z in out if isinstance(z, torch.Tensor)), None)
        if t is None:
            raise RuntimeError("Salida del encoder es tuple/list sin tensor.")
    elif isinstance(out, dict):
        t = None
        for k in ("z", "latent", "mu", "mean", "embedding"):
            if k in out and isinstance(out[k], torch.Tensor):
                t = out[k]
                break
        if t is None:
            t = next((v for v in out.values() if isinstance(v, torch.Tensor)), None)
        if t is None:
            raise RuntimeError("Salida del encoder es dict sin tensor.")
    else:
        raise RuntimeError(f"No sé interpretar salida del encoder: {type(out)}")

    # si viene [B,T,C] promediar en tiempo
    if t.ndim == 3:
        t = t.mean(dim=1)

    # flatten si hace falta
    if t.ndim > 2:
        t = t.view(t.shape[0], -1)

    z = t.detach().cpu().numpy()
    if z.shape[0] != 1:
        raise RuntimeError(f"Esperaba batch=1, obtuve: {z.shape}")
    return z[0].astype(np.float32)


# ---------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------
def l2(a: np.ndarray) -> float:
    return float(np.sqrt(np.sum(a * a)))


def detect_species(
    wav_path: str | Path,
    *,
    config_path: str | Path | None = None,
    encoder_pt: str | Path | None = None,
    encoder_yaml: str | Path | None = None,
    device: str = "cpu",
    sr: int = 48000,
    n_mels: int = 64,
    target_frames: int = 192,
    fmin: float = 150.0,
    fmax: float = 15000.0,
    hop_length: int = 384,
    n_fft: int = 2048,
) -> Tuple[bool, Optional[str]]:
    """
    Retorna:
      (True, "<species>") si detecta
      (False, None) si no detecta

    Regla:
      acepta especie k si ||z - centroid_k|| <= threshold_k
      si acepta múltiples, desempata por PRIORITY_ORDER.
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
    centroids, thresholds, duration = get_detector_from_config(cfg)

    dev = torch.device(device)

    enc_pt = Path(encoder_pt).expanduser().resolve() if encoder_pt else resolve_default_encoder_pt(project_root)
    enc_yaml = Path(encoder_yaml).expanduser().resolve() if encoder_yaml else resolve_default_encoder_yaml(project_root)

    encoder = load_encoder(enc_pt, enc_yaml, project_root, dev)

    z = encode_wav_to_latent(
        encoder=encoder,
        wav_path=wav_p,
        device=dev,
        sr=sr,
        duration=duration,  # usar la misma duración con que calibraste
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        n_fft=n_fft,
        target_frames=target_frames,
    )

    accepted: List[str] = []
    for sp, mu in centroids.items():
        if sp not in thresholds:
            continue
        rk = thresholds[sp]
        if mu.shape[0] != z.shape[0]:
            continue
        d = l2(z - mu)
        if d <= rk:
            accepted.append(sp)

    if not accepted:
        return False, None

    # desempate por prioridad
    for sp in PRIORITY_ORDER:
        if sp in accepted:
            return True, sp

    # fallback: si hay especies fuera del orden, elegir determinísticamente
    return True, sorted(accepted)[0]


# ---------------------------------------------------------------------
# CLI opcional
# ---------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, type=str, help="Ruta al archivo .wav a evaluar")
    p.add_argument("--config", type=str, default=None, help="Ruta a config.json (opcional)")
    p.add_argument("--encoder-pt", type=str, default=None, help="Ruta a model.pt (opcional)")
    p.add_argument("--encoder-yaml", type=str, default=None, help="Ruta a .yaml del encoder (opcional)")
    p.add_argument("--device", type=str, default="cpu")

    # params mel (defaults validados)
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
    detected, sp = detect_species(
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
        print(f"✅ DETECTADO: {sp}")
        sys.exit(0)
    else:
        print("❌ NO DETECTADO")
        sys.exit(2)


if __name__ == "__main__":
    main()


# python3 latent_space_exploration/09_evaluate_wav_detection.py --wav ./latent_space_exploration/test_chunks/Batrachyla_leptopus/audio696_label13_chunk1.wav
