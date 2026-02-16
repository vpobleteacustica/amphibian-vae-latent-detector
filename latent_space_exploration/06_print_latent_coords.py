#!/usr/bin/env python3
"""
Imprime coordenadas (vectores) del espacio latente SIN reducción de dimensionalidad.

- Lee un .parquet que contiene embeddings/latentes (columnas numéricas).
- Detecta automáticamente una columna "label/id" si existe para imprimirla como referencia.
- Imprime solo los primeros N puntos para no saturar la terminal (configurable).

Ejemplos:
  python 06_print_latent_coords.py
  python 06_print_latent_coords.py --n 10
  python 06_print_latent_coords.py --all
  python 06_print_latent_coords.py --parquet downloaded_models/bird_net_vae_audio_splitted_features_v0/bird_net_vae_audio_splitted_features.parquet --n 5
  python 06_print_latent_coords.py --jsonl --n 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


LABEL_CANDIDATES = ["label", "class", "category", "group", "filename", "file", "path", "id", "uid", "name"]


def find_project_root(start_dir: Path) -> Path:
    """Sube por los padres hasta encontrar una carpeta que parezca el root del proyecto.

    - Prioriza encontrar `downloaded_models/` (tu estructura actual).
    - Si no existe, prueba con `embeddings/` (estructura anterior).
    - Si no encuentra nada, usa el padre inmediato (caso común: script dentro de subcarpeta).
    """

    start_dir = start_dir.resolve()
    for p in [start_dir, *start_dir.parents]:
        if (p / "downloaded_models").exists():
            return p
        if (p / "embeddings").exists():
            return p

    # Fallback razonable: el padre del script (si existe)
    return start_dir.parent if start_dir.parent.exists() else start_dir


def resolve_default_parquet(project_root: Path) -> Optional[Path]:
    """Intenta encontrar un parquet por defecto en ubicaciones típicas del proyecto."""
    candidates: List[Path] = []

    # 1) Ruta del parquet que estás usando ahora (downloaded_models)
    candidates.append(
        project_root / "downloaded_models" / "bird_net_vae_audio_splitted_features_v0" / "bird_net_vae_audio_splitted_features.parquet"
    )

    # 2) Ruta usada en scripts existentes
    candidates.append(project_root / "embeddings" / "bird_net_vae_audio_splitted" / "features.parquet")

    # 3) Parquets en el root del proyecto
    candidates.extend(sorted(project_root.glob("*.parquet")))

    # 4) Parquets en embeddings/**/features.parquet
    candidates.extend(sorted(project_root.glob("embeddings/**/features.parquet")))

    # 5) Cualquier parquet dentro del proyecto (último recurso)
    candidates.extend(sorted(project_root.glob("**/*.parquet")))

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def extract_features_and_optional_label(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], str, List[str]]:
    """Extrae matriz X (todas las columnas numéricas) y un label/id opcional si existe."""
    label_col = None
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            label_col = col
            break

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No se encontraron columnas numéricas en el parquet (no hay embeddings/latentes).")

    X = df[numeric_cols].to_numpy()

    labels = None
    if label_col is not None:
        labels = df[label_col].to_numpy()

    return X, labels, (label_col or ""), numeric_cols


def print_rows(
    X: np.ndarray,
    labels: Optional[np.ndarray],
    label_col: str,
    numeric_cols: List[str],
    n: int,
    jsonl: bool,
    precision: int,
    max_width: int,
) -> None:
    n_total, dim = X.shape

    print("=" * 70)
    print("🧠 LATENT SPACE (SIN REDUCCIÓN)")
    print("=" * 70)
    print(f"📌 Total puntos: {n_total}")
    print(f"📐 Dimensión latente: {dim}")
    print(f"🔢 Columnas numéricas: {len(numeric_cols)}")
    if label_col:
        print(f"🏷️ Columna identificadora detectada: '{label_col}'")
    else:
        print("🏷️ Columna identificadora: (no detectada)")
    print("-" * 70)

    # Config de impresión
    np.set_printoptions(
        precision=precision,
        suppress=True,
        linewidth=max_width,
        threshold=max(1000, dim * 2),
        edgeitems=dim,
    )

    n_to_print = min(n_total, n)
    if n_to_print <= 0:
        print("⚠️ n_to_print es 0 (no hay nada que imprimir).")
        return

    for i in range(n_to_print):
        vec = X[i]
        label_val = None if labels is None else labels[i]

        if jsonl:
            rec = {
                "index": int(i),
                "label_col": label_col if label_col else None,
                "label": None if label_val is None else str(label_val),
                "vector": vec.astype(float).tolist(),
            }
            print(json.dumps(rec, ensure_ascii=False))
        else:
            header = f"[{i}]"
            if label_col and label_val is not None:
                header += f" {label_col}={label_val}"
            print(header)
            print(vec)
            print()

    if n_total > n_to_print:
        print(f"… (mostrados {n_to_print}/{n_total}). Usa --all o --n {n_total} para imprimir todo.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Imprime vectores latentes sin reducir dimensionalidad.")
    p.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Ruta al archivo .parquet con embeddings/latentes. Si se omite, se intenta autodetectar.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=2,
        help="Cantidad de puntos a imprimir (por defecto: 2).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Imprime todos los puntos (equivalente a --n N_TOTAL).",
    )
    p.add_argument(
        "--jsonl",
        action="store_true",
        help="Salida en formato JSON Lines (una fila por punto). Útil para parseo.",
    )
    p.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Precisión decimal para imprimir floats (modo texto).",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=160,
        help="Ancho máximo (caracteres) para impresión del vector (modo texto).",
    )
    p.add_argument(
        "--show-cols",
        action="store_true",
        help="Imprime también los nombres de columnas numéricas (útil para debugging).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Este script puede vivir en una subcarpeta (ej: latent_space_exploration/).
    # Por eso detectamos el root subiendo carpetas hasta encontrar `downloaded_models/`.
    scripts_dir = Path(__file__).resolve().parent
    project_root = find_project_root(scripts_dir)

    parquet_path = Path(args.parquet) if args.parquet else resolve_default_parquet(project_root)
    if parquet_path is None:
        expected = (
            project_root
            / "downloaded_models"
            / "bird_net_vae_audio_splitted_features_v0"
            / "bird_net_vae_audio_splitted_features.parquet"
        )
        raise FileNotFoundError(
            "No pude autodetectar un .parquet.\n"
            f"- Root detectado: {project_root}\n"
            f"- Ruta esperada: {expected}\n"
            "Pásalo explícitamente con --parquet /ruta/al/archivo.parquet"
        )

    if not parquet_path.exists():
        raise FileNotFoundError(f"No existe el parquet: {parquet_path}")

    print(f"📂 Cargando parquet: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
    except ImportError:
        msg = (
            "\n❌ Pandas no puede leer Parquet porque falta un engine (pyarrow/fastparquet).\n"
            "Instala uno de estos paquetes en *el mismo entorno* donde corres el script y vuelve a intentar:\n\n"
            "  pip install pyarrow\n"
            "\n(Alternativa: pip install fastparquet)\n"
        )
        raise SystemExit(msg)

    X, labels, label_col, numeric_cols = extract_features_and_optional_label(df)

    if args.show_cols:
        print("\n🧾 Nombres de columnas numéricas:")
        for c in numeric_cols:
            print(f"  - {c}")
        print()

    n = X.shape[0] if args.all else args.n
    print_rows(
        X=X,
        labels=labels,
        label_col=label_col,
        numeric_cols=numeric_cols,
        n=int(n),
        jsonl=bool(args.jsonl),
        precision=int(args.precision),
        max_width=int(args.max_width),
    )


if __name__ == "__main__":
    main()