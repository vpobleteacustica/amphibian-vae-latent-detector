# latent_space_exploration/map_detector_core.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_detector_core.py

Core reutilizable para:
- cargar config.json
- cargar encoder (Hydra/omegaconf)
- WAV -> mel (pipeline validado)
- mel -> latent z
- scoring MAP Gaussiano (con precision + logdet_cov)
- utilidades de paths / robustez

Pensado para ser importado por:
  - 08b_fit_map_detector.py
  - 09n_evaluate_wav_detection.py
  - 10b_benchmark_folder_detection_map.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# Paths / IO
# ---------------------------------------------------------------------
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(15):
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
    #p = project_root / "downloaded_models" / "bird_net_vae_audio_splitted_encoder_v0" / "model.pt"
    p = project_root / "models" / "bird_net_vae_audio_splitted_encoder_v0" / "model.pt"
    if not p.exists():
        raise FileNotFoundError(f"No encontré encoder .pt en: {p}")
    return p


def resolve_default_encoder_yaml(project_root: Path) -> Path:
    p = project_root / "models" / "bird_net_vae_audio_splitted_encoder_v0" / "bird_net_vae_audio_splitted.yaml"
    if not p.exists():
        raise FileNotFoundError(f"No encontré encoder YAML en: {p}")
    return p


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("JSON no es un objeto dict.")
    return obj


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def summarize_1d(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"min": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


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
    En soundscape_vae, instantiate(enc_cfg) suele devolver un factory callable.
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


def load_encoder(
    encoder_pt: Path,
    encoder_yaml: Path,
    project_root: Path,
    device: torch.device,
) -> torch.nn.Module:
    # asegurar importabilidad del repo
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
# Audio -> Mel -> Latent (pipeline validado)
# ---------------------------------------------------------------------
def crop_or_pad_time(mel: np.ndarray, target_frames: int) -> np.ndarray:
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
    *,
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

    # frames fijos
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

    # encoder espera [B,1,T,M]
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

    if t.ndim == 3:
        t = t.mean(dim=1)
    if t.ndim > 2:
        t = t.view(t.shape[0], -1)

    z = t.detach().cpu().numpy()
    if z.shape[0] != 1:
        raise RuntimeError(f"Esperaba batch=1, obtuve: {z.shape}")
    return z[0].astype(np.float32)


# ---------------------------------------------------------------------
# MAP helpers
# ---------------------------------------------------------------------
def inv_and_logdet(cov: np.ndarray) -> Tuple[np.ndarray, float]:
    sign, ld = np.linalg.slogdet(cov)
    if sign <= 0:
        d = cov.shape[0]
        cov2 = cov + (1e-3 * np.eye(d, dtype=cov.dtype))
        sign, ld = np.linalg.slogdet(cov2)
        if sign <= 0:
            raise RuntimeError("Covarianza no PD incluso tras regularización.")
        cov = cov2
    prec = np.linalg.inv(cov).astype(np.float32)
    return prec, float(ld)


def gaussian_logpdf_from_precision(z: np.ndarray, mu: np.ndarray, prec: np.ndarray, logdet_cov: float) -> float:
    d = int(z.shape[0])
    diff = (z - mu).astype(np.float32)
    quad = float(diff.T @ prec @ diff)
    return -0.5 * (quad + float(logdet_cov) + d * float(np.log(2.0 * np.pi)))


def get_priors_from_map_meta(cfg: Dict[str, Any], species: List[str]) -> Dict[str, float]:
    """
    Priors preferidos:
      cfg["map_detector"]["meta_fit"]["per_species"][sp]["prior"]
    Fallback: uniforme
    """
    priors: Dict[str, float] = {}
    md = cfg.get("map_detector", {})
    meta = md.get("meta_fit", {}) if isinstance(md, dict) else {}
    per = meta.get("per_species", {}) if isinstance(meta, dict) else {}

    ok = True
    for sp in species:
        try:
            pri = float(per.get(sp, {}).get("prior"))
            priors[sp] = pri
        except Exception:
            ok = False
            break

    if ok and priors:
        s = sum(max(0.0, v) for v in priors.values())
        if s > 0:
            priors = {k: max(0.0, v) / s for k, v in priors.items()}
        return priors

    K = len(species)
    if K == 0:
        return {}
    return {sp: 1.0 / K for sp in species}


def get_chunk_seconds_for_map(cfg: Dict[str, Any]) -> float:
    md = cfg.get("map_detector", {})
    if isinstance(md, dict):
        meta = md.get("meta_fit", {})
        if isinstance(meta, dict) and "chunk_seconds" in meta:
            try:
                return float(meta["chunk_seconds"])
            except Exception:
                pass
    try:
        return float(cfg.get("chunk_seconds", 5.0))
    except Exception:
        return 5.0


def read_map_detector_params(cfg: Dict[str, Any]) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], Optional[float]
]:
    """
    Lee map_detector tal como lo guarda 08b:
      means: {sp: [D]}
      precision: {sp: [[D,D]]}
      logdet_cov: {sp: float}
      tau: float|None
    """
    md = cfg.get("map_detector", None)
    if not isinstance(md, dict):
        raise ValueError("config.json no contiene map_detector (dict). Ejecuta antes 08b_fit_map_detector.py")
    if md.get("model", "") != "gaussian_map":
        raise ValueError(f"map_detector.model inesperado: {md.get('model')}")

    means_raw = md.get("means", None)
    prec_raw = md.get("precision", None)
    logdet_raw = md.get("logdet_cov", None)

    if not isinstance(means_raw, dict) or not isinstance(prec_raw, dict) or not isinstance(logdet_raw, dict):
        raise ValueError("map_detector debe contener 'means', 'precision' y 'logdet_cov' como dicts.")

    means: Dict[str, np.ndarray] = {}
    precisions: Dict[str, np.ndarray] = {}
    logdets: Dict[str, float] = {}

    for sp, vec in means_raw.items():
        if isinstance(sp, str) and isinstance(vec, list) and len(vec) > 0:
            means[sp] = np.array(vec, dtype=np.float32)

    for sp, mat in prec_raw.items():
        if isinstance(sp, str) and isinstance(mat, list) and len(mat) > 0:
            P = np.array(mat, dtype=np.float32)
            if P.ndim != 2 or P.shape[0] != P.shape[1]:
                raise ValueError(f"precision[{sp}] debe ser matriz cuadrada, obtuve shape={P.shape}")
            precisions[sp] = P

    for sp, v in logdet_raw.items():
        if isinstance(sp, str):
            logdets[sp] = float(v)

    tau = md.get("tau", None)
    tau_f = float(tau) if tau is not None else None

    if not means or not precisions or not logdets:
        raise ValueError("means/precision/logdet_cov vacíos o mal formateados en config.json.")

    return means, precisions, logdets, tau_f