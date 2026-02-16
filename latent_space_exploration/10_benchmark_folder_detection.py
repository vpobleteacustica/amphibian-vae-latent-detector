#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_benchmark_folder_detection.py

Escanea un directorio con estructura:
  test_chunks/
    EspecieA/*.wav
    EspecieB/*.wav
    ...

Para cada wav:
  - Ejecuta detección con el pipeline del VAE + radial detector (09_evaluate_wav_detection.py)
  - Compara especie predicha vs especie esperada (nombre de carpeta)
  - Genera métricas + diagramas:
      - Matriz de confusión (incluye NO_DETECT)
      - Accuracy por especie
      - Rate de NO_DETECT por especie
      - Resumen global

Outputs:
  outputs/detection_benchmark/
    results.csv
    confusion_matrix.png
    accuracy_by_class.png
    no_detect_rate_by_class.png
    global_counts.png
    summary.txt

Uso:
  python3 latent_space_exploration/10_benchmark_folder_detection.py \
      --root latent_space_exploration/test_chunks

Notas:
  - Importa 09_evaluate_wav_detection.py vía importlib (porque empieza con número y no es importable normal).
  - Carga encoder/config una sola vez para evitar lentitud y crashes.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise SystemExit(f"❌ Falta pandas. Instala: pip install pandas\nDetalle: {e}")

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(f"❌ Falta matplotlib. Instala: pip install matplotlib\nDetalle: {e}")

# seaborn es opcional (solo mejora heatmap)
try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None  # type: ignore


# -------------------------
# Utils
# -------------------------
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "downloaded_models").exists() and (cur / "latent_space_exploration").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def load_eval_module(project_root: Path):
    """
    Carga latent_space_exploration/09_evaluate_wav_detection.py como módulo dinámico.
    """
    mod_path = project_root / "latent_space_exploration" / "09_evaluate_wav_detection.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"No encontré: {mod_path}")

    spec = importlib.util.spec_from_file_location("evaluate_wav_detection", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("No pude crear spec para 09_evaluate_wav_detection.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_wav_detection"] = module
    spec.loader.exec_module(module)
    return module


def list_audio_files(root: Path, exts: Tuple[str, ...] = (".wav", ".WAV")) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            files.append(p)
    return sorted(files)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Detector session (carga una vez)
# -------------------------
@dataclass
class DetectorSession:
    module: object
    project_root: Path
    config_path: Path
    encoder_pt: Path
    encoder_yaml: Path
    device: str

    # mel params (deben coincidir con 09)
    sr: int = 48000
    n_mels: int = 64
    target_frames: int = 192
    fmin: float = 150.0
    fmax: float = 15000.0
    hop_length: int = 384
    n_fft: int = 2048

    # loaded
    centroids: Dict[str, np.ndarray] = None  # type: ignore
    thresholds: Dict[str, float] = None  # type: ignore
    duration: float = 5.0
    encoder: object = None  # torch.nn.Module

    def load(self) -> None:
        # config
        cfg = self.module.load_json(self.config_path)
        centroids, thresholds, duration = self.module.get_detector_from_config(cfg)
        self.centroids = centroids
        self.thresholds = thresholds
        self.duration = duration

        # encoder (una sola vez)
        import torch  # local import
        dev = torch.device(self.device)
        self.encoder = self.module.load_encoder(
            self.encoder_pt, self.encoder_yaml, self.project_root, dev
        )

    def predict_one(self, wav_path: Path) -> Tuple[bool, Optional[str], float]:
        """
        Retorna:
          detected, predicted_species, best_distance
        best_distance = min ||z - mu|| encontrado (aunque no detecte)
        """
        import torch  # local import
        dev = torch.device(self.device)

        z = self.module.encode_wav_to_latent(
            encoder=self.encoder,
            wav_path=wav_path,
            device=dev,
            sr=self.sr,
            duration=self.duration,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            target_frames=self.target_frames,
        )

        accepted: List[str] = []
        best_d = float("inf")

        for sp, mu in self.centroids.items():
            if sp not in self.thresholds:
                continue
            rk = float(self.thresholds[sp])
            if mu.shape[0] != z.shape[0]:
                continue
            d = float(self.module.l2(z - mu))
            best_d = min(best_d, d)
            if d <= rk:
                accepted.append(sp)

        if not accepted:
            return False, None, best_d

        # desempate por prioridad (usa la lista del módulo si existe, si no fallback)
        priority = getattr(self.module, "PRIORITY_ORDER", [])
        if priority:
            for sp in priority:
                if sp in accepted:
                    return True, sp, best_d

        return True, sorted(accepted)[0], best_d


# -------------------------
# Plots
# -------------------------
def plot_confusion_matrix(df: "pd.DataFrame", out_png: Path) -> None:
    labels = sorted(set(df["true_species"].unique()).union(set(df["pred_species"].unique())))
    # asegurar que NO_DETECT aparezca al final si existe
    if "NO_DETECT" in labels:
        labels = [l for l in labels if l != "NO_DETECT"] + ["NO_DETECT"]

    cm = pd.crosstab(df["true_species"], df["pred_species"], rownames=["true"], colnames=["pred"], dropna=False)
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)

    plt.figure(figsize=(1 + 0.6 * len(labels), 1 + 0.6 * len(labels)))

    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cbar=True)
    else:
        plt.imshow(cm.values, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        # anotar valores
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm.iat[i, j]), ha="center", va="center", fontsize=8)

    plt.title("Confusion Matrix (incluye NO_DETECT)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_class(df: "pd.DataFrame", out_png: Path) -> None:
    g = df.groupby("true_species")["correct"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, max(4, 0.35 * len(g))))
    plt.barh(g.index.tolist(), (g.values * 100.0))
    plt.xlabel("Accuracy (%)")
    plt.title("Accuracy por especie")
    plt.xlim(0, 100)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def plot_no_detect_rate_by_class(df: "pd.DataFrame", out_png: Path) -> None:
    g = df.groupby("true_species")["pred_species"].apply(lambda s: (s == "NO_DETECT").mean()).sort_values(ascending=False)
    plt.figure(figsize=(10, max(4, 0.35 * len(g))))
    plt.barh(g.index.tolist(), (g.values * 100.0))
    plt.xlabel("NO_DETECT rate (%)")
    plt.title("Tasa de NO_DETECT por especie")
    plt.xlim(0, 100)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def plot_global_counts(df: "pd.DataFrame", out_png: Path) -> None:
    total = len(df)
    correct = int(df["correct"].sum())
    wrong = int((~df["correct"]).sum())
    no_det = int((df["pred_species"] == "NO_DETECT").sum())

    labels = ["Correct", "Wrong", "NO_DETECT"]
    values = [correct, wrong, no_det]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.title(f"Resumen global (N={total})")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def write_summary(df: "pd.DataFrame", out_txt: Path) -> None:
    total = len(df)
    correct = int(df["correct"].sum())
    acc = (correct / total) if total else 0.0
    no_det = int((df["pred_species"] == "NO_DETECT").sum())
    no_det_rate = (no_det / total) if total else 0.0

    per_class = df.groupby("true_species").agg(
        n=("file", "count"),
        acc=("correct", "mean"),
        no_detect=("pred_species", lambda s: (s == "NO_DETECT").mean()),
    ).sort_values("acc", ascending=False)

    lines = []
    lines.append("=== Detection Benchmark Summary ===")
    lines.append(f"Total files: {total}")
    lines.append(f"Correct: {correct}  | Accuracy: {acc*100:.2f}%")
    lines.append(f"NO_DETECT: {no_det} | Rate: {no_det_rate*100:.2f}%")
    lines.append("")
    lines.append("=== Per-class ===")
    for sp, row in per_class.iterrows():
        lines.append(f"- {sp:30s}  n={int(row['n']):4d}  acc={row['acc']*100:6.2f}%  no_detect={row['no_detect']*100:6.2f}%")

    out_txt.write_text("\n".join(lines), encoding="utf-8")


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="Carpeta raíz a escanear (ej: latent_space_exploration/test_chunks)")
    p.add_argument("--config", type=str, default=None, help="Ruta a config.json (opcional)")
    p.add_argument("--encoder-pt", type=str, default=None, help="Ruta a model.pt del encoder (opcional)")
    p.add_argument("--encoder-yaml", type=str, default=None, help="Ruta a YAML del encoder (opcional)")
    p.add_argument("--device", type=str, default="cpu", help="cpu o cuda")

    # mel params (por si quieres override)
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--target-frames", type=int, default=192)
    p.add_argument("--fmin", type=float, default=150.0)
    p.add_argument("--fmax", type=float, default=15000.0)
    p.add_argument("--hop-length", type=int, default=384)
    p.add_argument("--n-fft", type=int, default=2048)
    return p.parse_args()


def main():
    here = Path(__file__).resolve().parent
    project_root = find_project_root(here)

    args = parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else (project_root / "latent_space_exploration" / "test_chunks")
    if not root.exists():
        raise FileNotFoundError(f"No existe root: {root}")

    # cargar módulo 09
    mod = load_eval_module(project_root)

    # defaults coherentes con tu proyecto (los mismos de 09)
    config_path = Path(args.config).expanduser().resolve() if args.config else (project_root / "config.json")
    encoder_pt = Path(args.encoder_pt).expanduser().resolve() if args.encoder_pt else (
        project_root / "downloaded_models" / "bird_net_vae_audio_splitted_encoder_v0" / "model.pt"
    )
    encoder_yaml = Path(args.encoder_yaml).expanduser().resolve() if args.encoder_yaml else (
        project_root / "downloaded_models" / "bird_net_vae_audio_splitted_encoder_v0" / "bird_net_vae_audio_splitted.yaml"
    )

    # output dir
    out_dir = project_root / "outputs" / "detection_benchmark"
    safe_mkdir(out_dir)

    # preparar sesión
    session = DetectorSession(
        module=mod,
        project_root=project_root,
        config_path=config_path,
        encoder_pt=encoder_pt,
        encoder_yaml=encoder_yaml,
        device=args.device,
        sr=args.sr,
        n_mels=args.n_mels,
        target_frames=args.target_frames,
        fmin=args.fmin,
        fmax=args.fmax,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
    )

    print("=" * 70)
    print("🔎 BENCHMARK DETECTION ON FOLDER")
    print(f"Root: {root}")
    print(f"Outputs: {out_dir}")
    print("=" * 70)

    print("⏳ Cargando detector (config + encoder) una sola vez...")
    session.load()
    print("✅ Listo.")

    # recorrer subcarpetas (cada carpeta = ground truth)
    class_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not class_dirs:
        raise RuntimeError(f"No encontré subcarpetas de especies en: {root}")

    rows = []
    total_files = 0

    for class_dir in sorted(class_dirs):
        true_species = class_dir.name
        wavs = list_audio_files(class_dir)
        if not wavs:
            print(f"⚠️ Sin wavs en {class_dir}")
            continue

        print(f"\n📁 {true_species}: {len(wavs)} archivos")
        for wav in wavs:
            total_files += 1
            try:
                detected, pred, best_d = session.predict_one(wav)
                pred_species = pred if detected and pred is not None else "NO_DETECT"
                correct = (pred_species == true_species)
                rows.append({
                    "file": str(wav),
                    "true_species": true_species,
                    "pred_species": pred_species,
                    "detected": bool(detected),
                    "correct": bool(correct),
                    "best_distance": float(best_d),
                })
            except Exception as e:
                rows.append({
                    "file": str(wav),
                    "true_species": true_species,
                    "pred_species": "ERROR",
                    "detected": False,
                    "correct": False,
                    "best_distance": np.nan,
                    "error": str(e),
                })

    if not rows:
        raise RuntimeError("No se procesó ningún archivo (rows vacío).")

    df = pd.DataFrame(rows)

    # guardar CSV
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n✅ CSV guardado: {csv_path}")

    # resumen
    summary_path = out_dir / "summary.txt"
    write_summary(df[df["pred_species"] != "ERROR"].copy(), summary_path)
    print(f"✅ Resumen guardado: {summary_path}")

    # plots (ignorar ERROR en métricas visuales)
    df_ok = df[df["pred_species"] != "ERROR"].copy()

    plot_confusion_matrix(df_ok, out_dir / "confusion_matrix.png")
    plot_accuracy_by_class(df_ok, out_dir / "accuracy_by_class.png")
    plot_no_detect_rate_by_class(df_ok, out_dir / "no_detect_rate_by_class.png")
    plot_global_counts(df_ok, out_dir / "global_counts.png")

    print("\n📈 Diagramas generados:")
    print(f" - {out_dir / 'confusion_matrix.png'}")
    print(f" - {out_dir / 'accuracy_by_class.png'}")
    print(f" - {out_dir / 'no_detect_rate_by_class.png'}")
    print(f" - {out_dir / 'global_counts.png'}")

    # métricas rápidas en consola
    total = len(df_ok)
    acc = float(df_ok["correct"].mean()) if total else 0.0
    no_det_rate = float((df_ok["pred_species"] == "NO_DETECT").mean()) if total else 0.0

    print("\n" + "=" * 70)
    print(f"✅ DONE  | N={total} | Acc={acc*100:.2f}% | NO_DETECT={no_det_rate*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()


## python3 latent_space_exploration/10_benchmark_folder_detection.py   --root latent_space_exploration/val_chunks   --device cpu 