# latent_space_exploration/10b_benchmark_folder_detection_map.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10b_benchmark_folder_detection_map.py

Benchmark de detección MAP (paralelo, NO interfiere con lo de José).

Escanea un directorio con estructura:
  val_chunks/  (o test_chunks/)
    EspecieA/*.wav
    EspecieB/*.wav
    ...

Para cada wav:
  - Codifica z con el encoder del VAE (una sola carga)
  - Score MAP usando parámetros en config.json (08b)
  - Rechazo por tau (si existe): best_score < tau -> NO_DETECT
  - Compara predicha vs verdadera (nombre de carpeta)
  - Exporta CSV + plots + summary
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise SystemExit(f"❌ Falta pandas. Instala: pip install pandas\nDetalle: {e}")

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(f"❌ Falta matplotlib. Instala: pip install matplotlib\nDetalle: {e}")

try:
    import torch
except Exception as e:
    raise SystemExit(f"❌ Falta torch. Detalle: {e}")

# seaborn es opcional
try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None  # type: ignore

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


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_audio_files(root: Path, exts: Tuple[str, ...] = (".wav", ".WAV")) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            files.append(p)
    return sorted(files)


@dataclass
class MapDetectorSession:
    project_root: Path
    config_path: Path
    encoder_pt: Path
    encoder_yaml: Path
    device: str

    # mel params
    sr: int = 48000
    n_mels: int = 64
    target_frames: int = 192
    fmin: float = 150.0
    fmax: float = 15000.0
    hop_length: int = 384
    n_fft: int = 2048

    # loaded
    means: Dict[str, np.ndarray] = None  # type: ignore
    precisions: Dict[str, np.ndarray] = None  # type: ignore
    logdets: Dict[str, float] = None  # type: ignore
    priors: Dict[str, float] = None  # type: ignore
    tau: Optional[float] = None
    duration: float = 5.0
    encoder: torch.nn.Module = None  # type: ignore
    species: List[str] = None  # type: ignore

    def load(self) -> None:
        cfg = load_json(self.config_path)

        means, precisions, logdets, tau = read_map_detector_params(cfg)
        species = sorted(set(means.keys()).intersection(precisions.keys()).intersection(logdets.keys()))
        if not species:
            raise RuntimeError("map_detector inconsistente: no hay intersección entre means/precision/logdet_cov.")

        self.means = means
        self.precisions = precisions
        self.logdets = logdets
        self.tau = tau
        self.species = species

        self.priors = get_priors_from_map_meta(cfg, species)
        self.duration = float(get_chunk_seconds_for_map(cfg))

        dev = torch.device(self.device)
        self.encoder = load_encoder(self.encoder_pt, self.encoder_yaml, self.project_root, dev)

    def predict_one(self, wav_path: Path) -> Tuple[bool, Optional[str], float]:
        dev = torch.device(self.device)

        z = encode_wav_to_latent(
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

        best_sp: Optional[str] = None
        best_score = -float("inf")

        for sp in self.species:
            mu = self.means[sp]
            prec = self.precisions[sp]
            ld = self.logdets[sp]
            if mu.shape[0] != z.shape[0]:
                continue

            lp = float(np.log(float(self.priors.get(sp, 1e-12)) + 1e-12))
            s = gaussian_logpdf_from_precision(z, mu, prec, ld) + lp

            if s > best_score:
                best_score = s
                best_sp = sp

        if best_sp is None:
            return False, None, best_score

        if self.tau is not None and best_score < float(self.tau):
            return False, None, best_score

        return True, best_sp, best_score


# -------------------------
# Plots + Summary
# -------------------------
def plot_confusion_matrix(df: "pd.DataFrame", out_png: Path) -> None:
    labels = sorted(set(df["true_species"].unique()).union(set(df["pred_species"].unique())))
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
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm.iat[i, j]), ha="center", va="center", fontsize=8)
    plt.title("Confusion Matrix (incluye NO_DETECT) — MAP")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_class(df: "pd.DataFrame", out_png: Path) -> None:
    g = df.groupby("true_species")["correct"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, max(4, 0.35 * len(g))))
    plt.barh(g.index.tolist(), (g.values * 100.0))
    plt.xlabel("Accuracy (%)")
    plt.title("Accuracy por especie — MAP")
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
    plt.title("Tasa de NO_DETECT por especie — MAP")
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
    plt.title(f"Resumen global (N={total}) — MAP")
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
    lines.append("=== Detection Benchmark Summary (MAP) ===")
    lines.append(f"Total files: {total}")
    lines.append(f"Correct: {correct}  | Accuracy: {acc*100:.2f}%")
    lines.append(f"NO_DETECT: {no_det} | Rate: {no_det_rate*100:.2f}%")
    lines.append("")
    lines.append("=== Per-class ===")
    for sp, row in per_class.iterrows():
        lines.append(
            f"- {sp:30s}  n={int(row['n']):4d}  acc={row['acc']*100:6.2f}%  no_detect={row['no_detect']*100:6.2f}%"
        )

    out_txt.write_text("\n".join(lines), encoding="utf-8")


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="Carpeta raíz a escanear (ej: latent_space_exploration/val_chunks)")
    p.add_argument("--config", type=str, default=None, help="Ruta a config.json (opcional)")
    p.add_argument("--encoder-pt", type=str, default=None, help="Ruta a model.pt (opcional)")
    p.add_argument("--encoder-yaml", type=str, default=None, help="Ruta a YAML del encoder (opcional)")
    p.add_argument("--device", type=str, default="cpu", help="cpu o cuda")

    # mel params (override)
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

    root = Path(args.root).expanduser().resolve() if args.root else (project_root / "latent_space_exploration" / "val_chunks")
    if not root.exists():
        raise FileNotFoundError(f"No existe root: {root}")

    config_path = Path(args.config).expanduser().resolve() if args.config else resolve_default_config(project_root)
    encoder_pt = Path(args.encoder_pt).expanduser().resolve() if args.encoder_pt else resolve_default_encoder_pt(project_root)
    encoder_yaml = Path(args.encoder_yaml).expanduser().resolve() if args.encoder_yaml else resolve_default_encoder_yaml(project_root)

    out_dir = project_root / "outputs" / "detection_benchmark_map"
    safe_mkdir(out_dir)

    session = MapDetectorSession(
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
    print("🔎 BENCHMARK DETECTION ON FOLDER — MAP")
    print(f"Root: {root}")
    print(f"Config: {config_path}")
    print(f"Outputs: {out_dir}")
    print("=" * 70)

    print("⏳ Cargando detector MAP (config + encoder) una sola vez...")
    session.load()
    print("✅ Listo.")
    print(f"⏱️ chunk_seconds usados: {session.duration}")
    print(f"🎯 tau (rechazo): {session.tau}\n")

    class_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not class_dirs:
        raise RuntimeError(f"No encontré subcarpetas de especies en: {root}")

    rows: List[Dict[str, Any]] = []

    for class_dir in sorted(class_dirs):
        true_species = class_dir.name
        wavs = list_audio_files(class_dir)
        if not wavs:
            print(f"⚠️ Sin wavs en {class_dir}")
            continue

        print(f"\n📁 {true_species}: {len(wavs)} archivos")
        for wav in wavs:
            try:
                detected, pred, best_score = session.predict_one(wav)
                pred_species = pred if detected and pred is not None else "NO_DETECT"
                correct = (pred_species == true_species)
                rows.append({
                    "file": str(wav),
                    "true_species": true_species,
                    "pred_species": pred_species,
                    "detected": bool(detected),
                    "correct": bool(correct),
                    "best_score": float(best_score),
                })
            except Exception as e:
                rows.append({
                    "file": str(wav),
                    "true_species": true_species,
                    "pred_species": "ERROR",
                    "detected": False,
                    "correct": False,
                    "best_score": np.nan,
                    "error": str(e),
                })

    if not rows:
        raise RuntimeError("No se procesó ningún archivo (rows vacío).")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n✅ CSV guardado: {csv_path}")

    df_ok = df[df["pred_species"] != "ERROR"].copy()

    summary_path = out_dir / "summary.txt"
    write_summary(df_ok, summary_path)
    print(f"✅ Resumen guardado: {summary_path}")

    plot_confusion_matrix(df_ok, out_dir / "confusion_matrix.png")
    plot_accuracy_by_class(df_ok, out_dir / "accuracy_by_class.png")
    plot_no_detect_rate_by_class(df_ok, out_dir / "no_detect_rate_by_class.png")
    plot_global_counts(df_ok, out_dir / "global_counts.png")

    print("\n📈 Diagramas generados:")
    print(f" - {out_dir / 'confusion_matrix.png'}")
    print(f" - {out_dir / 'accuracy_by_class.png'}")
    print(f" - {out_dir / 'no_detect_rate_by_class.png'}")
    print(f" - {out_dir / 'global_counts.png'}")

    total = len(df_ok)
    acc = float(df_ok["correct"].mean()) if total else 0.0
    no_det_rate = float((df_ok["pred_species"] == "NO_DETECT").mean()) if total else 0.0

    print("\n" + "=" * 70)
    print(f"✅ DONE (MAP) | N={total} | Acc={acc*100:.2f}% | NO_DETECT={no_det_rate*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()