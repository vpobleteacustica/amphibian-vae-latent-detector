#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_fit_radial_detector.py

Fittea detector radial por especie:
- Centroid mu_k = mean(z) usando audios de la especie k
- Umbral intra-clase rk_in = quantile(q_in) de ||z - mu_k|| para z de la clase k
- Umbral inter-clase rk_out = quantile(q_out) de ||z_other - mu_k|| para z de otras clases
- Umbral final rk = min(rk_in, rk_out)

Notas:
- rk_out con q_out pequeño (p.ej. 0.01) fuerza baja tasa de falsos positivos hacia esa clase.
- Si una clase tiene pocos ejemplos, rk puede quedar inestable -> usar --max-per-class y/o ajustar q_in/q_out.

Guarda en config.json:
  radial_detector.centroids / thresholds / meta_fit
y crea backup config.json.bak

Uso recomendado (desde repo_jose/modelos_VAE/latent_space_exploration):
  python 08_fit_radial_detector.py \
      --root train_chunks \
      --q-in 0.95 \
      --q-out 0.01 \
      --device cpu \
      --max-per-class 400 \
      --cache

python 08_fit_radial_detector.py \
  --root train_chunks \
  --q-in 0.95 \
  --q-out 0.01 \
  --device cpu \
  --max-per-class 400 \
  --cache
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

try:
    import librosa
except Exception as e:
    raise SystemExit(f"❌ Falta librosa. Instala con: pip install librosa\nDetalle: {e}")

try:
    from omegaconf import OmegaConf  # type: ignore
    from hydra.utils import instantiate  # type: ignore
except Exception:
    OmegaConf = None  # type: ignore
    instantiate = None  # type: ignore


# ----------------------------
# Helpers
# ----------------------------

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(15):
        if (cur / "downloaded_models").exists() and (cur / "latent_space_exploration").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise SystemExit("❌ config.json no es un objeto JSON dict.")
    return obj


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_default_encoder_pt(project_root: Path) -> Path:
    p = project_root / "downloaded_models" / "bird_net_vae_audio_splitted_encoder_v0" / "model.pt"
    if not p.exists():
        raise SystemExit(f"❌ No encontré encoder .pt en: {p}")
    return p


def resolve_default_encoder_yaml(project_root: Path) -> Path:
    p = project_root / "downloaded_models" / "bird_net_vae_audio_splitted_encoder_v0" / "bird_net_vae_audio_splitted.yaml"
    if not p.exists():
        raise SystemExit(f"❌ No encontré encoder YAML en: {p}")
    return p


def l2_norm_rows(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=1))


def quantile_safe(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, q))


def summarize_dist(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"min": float("nan"), "p50": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }


# ----------------------------
# Encoder loading (Hydra)
# ----------------------------

def load_yaml_cfg(cfg_path: Path) -> Dict[str, Any]:
    if OmegaConf is None:
        raise SystemExit("❌ Falta omegaconf/hydra-core. Instala: pip install omegaconf hydra-core")
    cfg_obj = OmegaConf.load(str(cfg_path))  # type: ignore
    cfg = OmegaConf.to_container(cfg_obj, resolve=False)  # type: ignore
    if not isinstance(cfg, dict):
        raise SystemExit("❌ No pude convertir YAML a dict.")
    return cfg


def pick_encoder_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "encoder" in cfg and isinstance(cfg["encoder"], dict):
        enc = cfg["encoder"]
        if "_target_" not in enc:
            raise SystemExit("❌ El bloque encoder del YAML no tiene _target_.")
        return enc
    raise SystemExit("❌ El YAML no contiene un bloque 'encoder:'.")


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
    if isinstance(obj, torch.nn.Module):
        return obj
    if callable(obj):
        m = obj()
        if isinstance(m, torch.nn.Module):
            return m
        raise SystemExit(f"❌ Llamé factory() pero no devolvió nn.Module: {type(m)}")
    raise SystemExit(f"❌ No pude construir nn.Module desde: {type(obj)}")


def load_encoder(encoder_pt: Path, encoder_yaml: Path, project_root: Path, device: torch.device) -> torch.nn.Module:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    ckpt = torch.load(str(encoder_pt), map_location=str(device))
    model_or_none, state = split_model_and_state(ckpt)

    if model_or_none is not None:
        module = build_nn_module(model_or_none)
        return module.to(device).eval()

    if instantiate is None:
        raise SystemExit("❌ hydra-core no disponible (instantiate).")
    if state is None:
        raise SystemExit("❌ No encontré state_dict en el checkpoint.")

    cfg = load_yaml_cfg(encoder_yaml)
    enc_cfg = pick_encoder_cfg(cfg)

    factory = instantiate(enc_cfg)  # type: ignore
    module = build_nn_module(factory)

    module.load_state_dict(state, strict=False)
    return module.to(device).eval()


# ----------------------------
# Audio -> mel -> latent
# ----------------------------

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
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
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

    x = mel.T.unsqueeze(0).unsqueeze(0).to(device)  # [B,1,T,M]
    out = encoder(x)

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


def fit_species_with_fp_control(
    Z_in: np.ndarray,
    Z_out: Optional[np.ndarray],
    q_in: float,
    q_out: float,
) -> Tuple[np.ndarray, float, float, float, Dict[str, Any]]:
    mu = np.mean(Z_in, axis=0).astype(np.float32)

    rho_in = l2_norm_rows(Z_in - mu[None, :])
    rk_in = quantile_safe(rho_in, q_in)

    if Z_out is None or Z_out.size == 0:
        rho_out = np.array([], dtype=np.float32)
        rk_out = float("inf")
    else:
        rho_out = l2_norm_rows(Z_out - mu[None, :])
        rk_out = quantile_safe(rho_out, q_out)

    rk = float(min(rk_in, rk_out))
    extra = {
        "rho_in_summary": summarize_dist(rho_in),
        "rho_out_summary": summarize_dist(rho_out),
    }
    return mu, rk, rk_in, rk_out, extra


# ----------------------------
# CLI / Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.json")
    p.add_argument("--root", type=str, required=True, help="Carpeta con subcarpetas por especie (train_chunks/test_chunks/val_chunks)")
    p.add_argument("--q-in", type=float, default=0.95)
    p.add_argument("--q-out", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--target-frames", type=int, default=192)
    p.add_argument("--fmin", type=float, default=150.0)
    p.add_argument("--fmax", type=float, default=15000.0)
    p.add_argument("--hop-length", type=int, default=384)
    p.add_argument("--n-fft", type=int, default=2048)

    p.add_argument("--encoder-pt", type=str, default=None)
    p.add_argument("--encoder-yaml", type=str, default=None)

    p.add_argument("--max-per-class", type=int, default=0, help="0 = usar todos; si >0, samplea hasta este N por especie")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--cache", action="store_true", help="Guardar/cargar latentes Z por especie en cache_npz/")
    return p.parse_args()


def main():
    args = parse_args()

    # validate quantiles
    if not (0.0 < args.q_in < 1.0):
        raise SystemExit("❌ --q-in debe estar en (0,1).")
    if not (0.0 < args.q_out < 1.0):
        raise SystemExit("❌ --q-out debe estar en (0,1).")

    random.seed(args.seed)
    np.random.seed(args.seed)

    here = Path(__file__).resolve()
    project_root = find_project_root(here.parent)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (project_root / cfg_path).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"❌ No existe config.json en: {cfg_path}")

    cfg = load_json(cfg_path)

    species_list = cfg.get("species", None)
    if not isinstance(species_list, list) or not all(isinstance(s, str) for s in species_list):
        raise SystemExit("❌ config.json debe tener un campo 'species' (lista de strings).")

    chunk_seconds = cfg.get("chunk_seconds", 5.0)
    try:
        chunk_seconds = float(chunk_seconds)
    except Exception:
        chunk_seconds = 5.0

    # ------------------------------------------------------------
    # ✅ AJUSTE CLAVE: resolver --root de forma robusta
    #   Si --root es relativo, probamos:
    #   1) relativo al CWD (ideal si estás en latent_space_exploration/)
    #   2) relativo al project_root
    #   3) relativo a project_root/latent_space_exploration
    # ------------------------------------------------------------
    root_in = Path(args.root).expanduser()
    candidates: List[Path] = []

    if root_in.is_absolute():
        candidates = [root_in]
    else:
        candidates = [
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

    device = torch.device(args.device)
    encoder_pt = Path(args.encoder_pt).resolve() if args.encoder_pt else resolve_default_encoder_pt(project_root)
    encoder_yaml = Path(args.encoder_yaml).resolve() if args.encoder_yaml else resolve_default_encoder_yaml(project_root)

    cache_dir = (project_root / "latent_space_exploration" / "cache_npz").resolve()
    if args.cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"📌 Project root: {project_root}")
    print(f"🧾 Config: {cfg_path}")
    print(f"📁 Chunks dir: {chunks_dir}")
    print(f"🖥️ Device: {device}")
    print(f"🎯 q_in={args.q_in} | q_out={args.q_out} | max_per_class={args.max_per_class} | cache={args.cache}")
    print("")

    # pre-check wav counts
    counts = {}
    for sp in species_list:
        sp_dir = chunks_dir / sp
        wavs = list(sp_dir.glob("*.wav")) if sp_dir.exists() else []
        counts[sp] = len(wavs)
    print("📦 WAVs por especie en root:")
    for sp in species_list:
        print(f"   - {sp}: {counts[sp]}")
    print("")

    encoder = load_encoder(encoder_pt, encoder_yaml, project_root, device)

    # 1) encode Z per species (with optional cache + sampling)
    Z_by_species: Dict[str, np.ndarray] = {}
    n_failed_by_species: Dict[str, int] = {}
    used_by_species: Dict[str, int] = {}

    for sp in species_list:
        sp_dir = (chunks_dir / sp).resolve()
        if not sp_dir.exists():
            print(f"⚠️ {sp}: carpeta no existe: {sp_dir} (se omite).")
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
            print(f"⚠️ {sp}: sin wavs en {sp_dir} (se omite).")
            continue

        # sampling to reduce domination
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

    # 2) compute centroids + thresholds
    centroids: Dict[str, List[float]] = {}
    thresholds: Dict[str, float] = {}
    meta: Dict[str, Any] = {}

    for sp, Z_in in Z_by_species.items():
        Z_out_list = [Z for other, Z in Z_by_species.items() if other != sp]
        Z_out = np.concatenate(Z_out_list, axis=0) if Z_out_list else None

        mu, rk, rk_in, rk_out, extra = fit_species_with_fp_control(
            Z_in=Z_in,
            Z_out=Z_out,
            q_in=args.q_in,
            q_out=args.q_out,
        )

        centroids[sp] = mu.tolist()
        thresholds[sp] = float(rk)

        meta[sp] = {
            "N_in": int(Z_in.shape[0]),
            "N_out": int(Z_out.shape[0]) if Z_out is not None else 0,
            "rk_in": float(rk_in),
            "rk_out": float(rk_out) if np.isfinite(rk_out) else None,
            "rk_final": float(rk),
            "failed": int(n_failed_by_species.get(sp, 0)),
            "used": int(used_by_species.get(sp, Z_in.shape[0])),
            **extra,
        }

        rk_out_print = rk_out if np.isfinite(rk_out) else float("nan")
        print(f"✅ {sp}: rk_in={rk_in:.6f} | rk_out={rk_out_print:.6f} | rk={rk:.6f}")
        print(f"   rho_in:  {extra['rho_in_summary']}")
        print(f"   rho_out: {extra['rho_out_summary']}")

    # 3) write config
    cfg.setdefault("radial_detector", {})
    if not isinstance(cfg["radial_detector"], dict):
        cfg["radial_detector"] = {}

    cfg["radial_detector"]["centroids"] = centroids
    cfg["radial_detector"]["thresholds"] = thresholds
    cfg["radial_detector"]["meta_fit"] = {
        "chunks_dir": str(chunks_dir),
        "chunks_name": chunks_dir.name,
        "q_in": float(args.q_in),
        "q_out": float(args.q_out),
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
        "per_species": meta,
    }

    backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    shutil.copy2(cfg_path, backup)
    save_json(cfg_path, cfg)

    print(f"\n💾 Guardado en: {cfg_path}")
    print(f"🗂️ Backup: {backup}")


if __name__ == "__main__":
    main()