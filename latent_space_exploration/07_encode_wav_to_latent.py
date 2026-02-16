#!/usr/bin/env python3
"""
07_encode_wav_to_latent.py

Qué hace:
- Carga el encoder entrenado (model.pt) + arquitectura desde YAML (Hydra)
- Lee un WAV
- Genera un mel-spectrogram
- IMPORTANTE (descubrimiento): el encoder espera el input como [B, 1, T, M]
  (o sea, mel TRANSPOSED). Además, n_mels correcto = 64.
- Encuentra automáticamente un target_frames válido (opcional) usando un hook al primer Linear
  y luego imprime el vector latente en consola.

Uso recomendado (rápido, sin auto):
python3 latent_space_exploration/07_encode_wav_to_latent.py \
  --wav latent_space_exploration/test_chunks/.../audio.wav \
  --encoder-config downloaded_models/bird_net_vae_audio_splitted_encoder_v0/bird_net_vae_audio_splitted.yaml \
  --n-mels 64 --target-frames 192

Uso con auto-frames (si quieres que busque):
python3 latent_space_exploration/07_encode_wav_to_latent.py \
  --wav latent_space_exploration/test_chunks/.../audio.wav \
  --encoder-config downloaded_models/bird_net_vae_audio_splitted_encoder_v0/bird_net_vae_audio_splitted.yaml \
  --n-mels 64 --auto-frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    import librosa
except Exception as e:
    raise SystemExit(f"❌ Falta librosa. Instala con: pip install librosa\nDetalle: {e}")

try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:
    OmegaConf = None  # type: ignore

try:
    from hydra.utils import instantiate  # type: ignore
except Exception:
    instantiate = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "downloaded_models").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def resolve_default_encoder(project_root: Path) -> Optional[Path]:
    p = project_root / "downloaded_models" / "bird_net_vae_audio_splitted_encoder_v0" / "model.pt"
    return p if p.exists() else None


# --------------------------------------------------------------------------------------
# YAML / Hydra
# --------------------------------------------------------------------------------------

def load_yaml_cfg(cfg_path: Path) -> Dict[str, Any]:
    # Carga YAML sin resolver interpolaciones Hydra (${...})
    if OmegaConf is not None:
        cfg_obj = OmegaConf.load(str(cfg_path))  # type: ignore
        cfg = OmegaConf.to_container(cfg_obj, resolve=False)  # type: ignore
        if not isinstance(cfg, dict):
            raise SystemExit("❌ No pude convertir YAML a dict (OmegaConf).")
        return cfg

    if yaml is None:
        raise SystemExit("❌ Falta omegaconf/hydra-core o pyyaml. Instala: pip install omegaconf hydra-core pyyaml")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)  # type: ignore
    if not isinstance(cfg, dict):
        raise SystemExit("❌ No pude convertir YAML a dict (PyYAML).")
    return cfg


def pick_encoder_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "encoder" in cfg and isinstance(cfg["encoder"], dict):
        enc = cfg["encoder"]
        if "_target_" not in enc:
            for k in ("target", "class_path"):
                if k in enc:
                    enc["_target_"] = enc[k]
                    break
        return enc
    if "_target_" in cfg:
        return cfg
    raise SystemExit("❌ El YAML no contiene bloque 'encoder:' ni '_target_'.")


# --------------------------------------------------------------------------------------
# Checkpoint / encoder
# --------------------------------------------------------------------------------------

def split_model_and_state(ckpt: Any) -> Tuple[Optional[Any], Optional[Dict[str, torch.Tensor]]]:
    if isinstance(ckpt, torch.nn.Module):
        return ckpt, ckpt.state_dict()

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return None, ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return None, ckpt["model_state_dict"]
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return None, ckpt  # type: ignore

    return None, None


def build_torch_module_from_factory(obj: Any) -> torch.nn.Module:
    """
    En tu repo, _BirdNet es un factory callable: hay que hacer obj() para obtener nn.Module real.
    """
    if isinstance(obj, torch.nn.Module):
        return obj

    for attr in ("module", "model", "net", "encoder", "backbone"):
        try:
            inner = getattr(obj, attr)
        except Exception:
            continue
        if isinstance(inner, torch.nn.Module):
            return inner

    if callable(obj):
        built = obj()
        if isinstance(built, torch.nn.Module):
            return built
        raise SystemExit(f"❌ Llamé obj() pero NO devolvió nn.Module: {type(built)}")

    raise SystemExit("❌ No pude obtener nn.Module desde el encoder instanciado.")


def load_encoder(
    encoder_path: Path,
    project_root: Path,
    device: torch.device,
    encoder_config_path: Optional[Path],
) -> torch.nn.Module:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    ckpt = torch.load(str(encoder_path), map_location=str(device))
    model_or_none, state = split_model_and_state(ckpt)

    if model_or_none is not None:
        module = build_torch_module_from_factory(model_or_none)
        return module.to(device).eval()

    if encoder_config_path is None:
        raise SystemExit("❌ model.pt es state_dict: necesitas --encoder-config (YAML).")
    if state is None:
        raise SystemExit("❌ No encontré state_dict dentro del checkpoint.")
    if instantiate is None:
        raise SystemExit("❌ Falta hydra-core. Instala: pip install hydra-core")

    cfg = load_yaml_cfg(encoder_config_path)
    enc_cfg = pick_encoder_cfg(cfg)

    factory = instantiate(enc_cfg)  # _BirdNet (callable)
    module = build_torch_module_from_factory(factory)

    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        print(f"⚠️ Missing keys: {len(missing)} (primeras 10): {missing[:10]}")
    if unexpected:
        print(f"⚠️ Unexpected keys: {len(unexpected)} (primeras 10): {unexpected[:10]}")

    return module.to(device).eval()


def find_first_linear(module: torch.nn.Module) -> torch.nn.Linear:
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            return m
    raise RuntimeError("No encontré ningún nn.Linear dentro del encoder.")


# --------------------------------------------------------------------------------------
# Audio -> mel
# --------------------------------------------------------------------------------------

def crop_or_pad_time(mel: np.ndarray, target_frames: int) -> np.ndarray:
    """
    mel: [n_mels, T]
    Fuerza T == target_frames con crop centrado o pad con ceros.
    """
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


# --------------------------------------------------------------------------------------
# Forward + extracción vector
# --------------------------------------------------------------------------------------

def extract_vector(out: Any) -> torch.Tensor:
    """
    Convierte salida a [B, D].
    - [B,T,C] -> mean(T) -> [B,C]
    - [B, ...] con >2 dims -> flatten -> [B,-1]
    """
    if isinstance(out, torch.Tensor):
        t = out
    elif isinstance(out, (list, tuple)):
        t = next((x for x in out if isinstance(x, torch.Tensor)), None)
        if t is None:
            raise ValueError("Salida list/tuple sin tensors.")
    elif isinstance(out, dict):
        t = None
        for k in ("z", "latent", "mu", "mean", "embedding", "enc"):
            if k in out and isinstance(out[k], torch.Tensor):
                t = out[k]
                break
        if t is None:
            t = next((v for v in out.values() if isinstance(v, torch.Tensor)), None)
        if t is None:
            raise ValueError("Salida dict sin tensors.")
    else:
        raise ValueError(f"No sé interpretar salida: {type(out)}")

    if t.ndim == 3:
        if t.shape[1] == 0:
            return t[:, :0]
        t = t.mean(dim=1)

    if t.ndim > 2:
        t = t.view(t.shape[0], -1)

    return t


def try_forward(encoder: torch.nn.Module, x: torch.Tensor) -> Tuple[bool, Optional[torch.Tensor], Optional[str]]:
    try:
        with torch.no_grad():
            out = encoder(x)
        vec = extract_vector(out)
        if vec.numel() == 0 or vec.shape[1] == 0:
            return False, None, "Salida vacía (numel=0)."
        return True, vec, None
    except Exception as e:
        return False, None, str(e)


# --------------------------------------------------------------------------------------
# Hook probe (para auto-frames)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def probe_linear_input_shape(
    encoder: torch.nn.Module,
    linear: torch.nn.Linear,
    x: torch.Tensor,
) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
    """
    Captura (N, F) antes del primer Linear.
    - F = inp.shape[-1]
    - N = inp.numel() // F  (colapsa B*T*... en una sola dimensión)
    """
    captured: Dict[str, Any] = {"shape": None}

    def pre_hook(_mod, inputs):
        try:
            inp = inputs[0]
            if isinstance(inp, torch.Tensor) and inp.ndim >= 2:
                F = int(inp.shape[-1])
                N = int(inp.numel() // F)
                captured["shape"] = (N, F)
        except Exception:
            pass

    handle = linear.register_forward_pre_hook(pre_hook)
    try:
        _ = encoder(x)
        shp = captured["shape"]
        handle.remove()
        if shp is None:
            return False, None, "No pude capturar la entrada al Linear (hook no disparó)."
        return True, shp, None
    except Exception as e:
        shp = captured["shape"]
        handle.remove()
        if shp is None:
            return False, None, str(e)
        return True, shp, str(e)


def auto_find_frames_with_hook(
    encoder: torch.nn.Module,
    wav_path: Path,
    device: torch.device,
    sr: int,
    duration: float,
    n_mels: int,
    fmin: float,
    fmax: float,
    hop_length: int,
    n_fft: int,
    start_frames: int,
    max_frames: int,
    step: int,
) -> int:
    """
    Busca un target_frames tal que:
      - la entrada al primer Linear tenga F == linear.in_features
      - N > 0 (no vacío)

    IMPORTANTE:
      aquí usamos SIEMPRE la orientación correcta: x = mel.T -> [1,1,T,M]
    """
    linear = find_first_linear(encoder)
    expected_F = int(linear.in_features)

    start = max(8, int(start_frames))
    step = max(1, int(step))
    max_frames = max(start, int(max_frames))

    for frames in range(start, max_frames + 1, step):
        mel = wav_to_mel(
            wav_path=wav_path,
            sr=sr,
            duration=duration,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            hop_length=hop_length,
            n_fft=n_fft,
            target_frames=frames,
        )
        x = mel.T.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T,M] ✅
        hook_ok, shp, _err = probe_linear_input_shape(encoder, linear, x)
        if not hook_ok or shp is None:
            continue
        N, F = shp
        if F == expected_F and N > 0:
            return frames

    raise SystemExit(
        "❌ No encontré un target_frames válido usando el hook.\n"
        f"Probé frames desde {start} hasta {max_frames} step={step}.\n"
        "Tip: sube --auto-max-frames (ej. 4096) o baja --auto-step (ej. 1)."
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, type=str)
    p.add_argument("--encoder", type=str, default=None)
    p.add_argument("--encoder-config", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")

    # Defaults que ahora sabemos que sí calzan con tu encoder:
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--n-mels", type=int, default=64)          # ✅ clave
    p.add_argument("--fmin", type=float, default=150.0)
    p.add_argument("--fmax", type=float, default=15000.0)
    p.add_argument("--hop-length", type=int, default=384)
    p.add_argument("--n-fft", type=int, default=2048)

    p.add_argument("--target-frames", type=int, default=192)  # ✅ clave (en tu prueba N>0 aparece aquí)
    p.add_argument("--auto-frames", action="store_true")

    # Parámetros de búsqueda (si te preocupa el tiempo, sube step)
    p.add_argument("--auto-max-frames", type=int, default=512)
    p.add_argument("--auto-step", type=int, default=8)

    p.add_argument("--jsonl", action="store_true")
    p.add_argument("--precision", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    here = Path(__file__).resolve().parent
    project_root = find_project_root(here)

    wav_path = Path(args.wav)
    if not wav_path.is_absolute():
        wav_path = (Path.cwd() / wav_path).resolve()

    device = torch.device(args.device)

    encoder_path = Path(args.encoder) if args.encoder else resolve_default_encoder(project_root)
    if encoder_path is None or not encoder_path.exists():
        raise SystemExit("❌ No encontré encoder por defecto. Pasa la ruta con --encoder ...")

    encoder_config_path = Path(args.encoder_config) if args.encoder_config else None

    print(f"📌 Project root: {project_root}")
    print(f"🎧 WAV: {wav_path}")
    print(f"🧠 Encoder: {encoder_path}")
    if encoder_config_path:
        print(f"🧾 Encoder config: {encoder_config_path}")
    print(f"🖥️ Device: {device}\n")

    encoder = load_encoder(
        encoder_path=encoder_path,
        project_root=project_root,
        device=device,
        encoder_config_path=encoder_config_path,
    )

    if args.auto_frames:
        chosen_frames = auto_find_frames_with_hook(
            encoder=encoder,
            wav_path=wav_path,
            device=device,
            sr=args.sr,
            duration=args.duration,
            n_mels=args.n_mels,
            fmin=args.fmin,
            fmax=args.fmax,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            start_frames=args.target_frames,
            max_frames=args.auto_max_frames,
            step=args.auto_step,
        )
    else:
        chosen_frames = args.target_frames

    print(f"✅ target_frames usado: {chosen_frames}")

    mel = wav_to_mel(
        wav_path=wav_path,
        sr=args.sr,
        duration=args.duration,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        target_frames=chosen_frames,
    )

    # ✅ orientación correcta: [B,1,T,M]
    x = mel.T.unsqueeze(0).unsqueeze(0).to(device)

    ok, vec, err = try_forward(encoder, x)
    if not ok or vec is None:
        raise SystemExit(f"❌ Forward falló incluso con target_frames={chosen_frames}. Error: {err}")

    vec_np = vec.detach().cpu().numpy()[0]
    np.set_printoptions(precision=int(args.precision), suppress=True, linewidth=180)

    if args.jsonl:
        print(json.dumps(
            {"wav": str(wav_path), "latent_dim": int(vec_np.size), "vector": vec_np.astype(float).tolist()},
            ensure_ascii=False
        ))
    else:
        print(f"✅ Latent dim: {vec_np.size}")
        print(vec_np)


if __name__ == "__main__":
    main()




# python3 latent_space_exploration/07_encode_wav_to_latent.py --wav latent_space_exploration/test_chunks/Batrachyla_leptopus/audio696_label13_chunk1.wav --encoder-config downloaded_models/bird_net_vae_audio_splitted_encoder_v0/bird_net_vae_audio_splitted.yaml 
# python3 07_encode_wav_to_latent.py --wav ./test_chunks/Batrachyla_leptopus/audio696_label13_chunk1.wav
# latent_space_exploration/test_chunks/Batrachyla_leptopus/audio696_label13_chunk1.wav
# python3 latent_space_exploration/07_encode_wav_to_latent.py --wav test_chunks/Batrachyla_leptopus/audio696_label13_chunk1.wav --encoder-config downloaded_models/bird_net_vae_audio_splitted_encoder_v0/bird_net_vae_audio_splitted.yaml
