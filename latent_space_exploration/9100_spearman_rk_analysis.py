#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
9100_spearman_rk_analysis.py

Lee un grid de experimentos tipo:
  outputs/qout_grid_YYYYMMDD/qout_0.10/
      summary.txt
      config_snapshot.json
      run.log
  outputs/qout_grid_YYYYMMDD/qout_0.15/
      ...

Extrae:
- Global: Accuracy y NO_DETECT rate desde summary.txt
- Per-class: acc y no_detect por especie desde summary.txt
- rk per especie desde config_snapshot.json (rk_per_species, rk_in_per_species, rk_out_per_species)

Luego calcula correlaciones Spearman:
- GLOBAL: corr(q_out, ACC_global), corr(q_out, NO_DETECT_global)
- PER SPECIES: corr(rk, acc), corr(rk, no_detect) y opcional corr(q_out, rk)

Guarda:
- spearman_table.csv en el grid dir
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise SystemExit(f"❌ Falta pandas. Instala: pip install pandas\nDetalle: {e}")

try:
    from scipy.stats import spearmanr
except Exception as e:
    raise SystemExit(f"❌ Falta scipy (spearmanr). Instala: pip install scipy\nDetalle: {e}")


# -----------------------------
# Parsers
# -----------------------------
_SUMMARY_GLOBAL_RE = re.compile(r"Correct:\s*\d+\s*\|\s*Accuracy:\s*([0-9.]+)%")
_SUMMARY_NODET_RE = re.compile(r"NO_DETECT:\s*\d+\s*\|\s*Rate:\s*([0-9.]+)%")
_SUMMARY_CLASS_RE = re.compile(
    r"^\-\s*(?P<sp>.+?)\s+n=\s*(?P<n>\d+)\s+acc=\s*(?P<acc>[0-9.]+)%\s+no_detect=\s*(?P<nd>[0-9.]+)%",
    re.IGNORECASE,
)

@dataclass
class RunRecord:
    q_out: float
    # global
    acc_global: float
    no_detect_global: float
    # per species metrics
    acc_per_species: Dict[str, float]
    no_detect_per_species: Dict[str, float]
    # rk per species
    rk_per_species: Dict[str, float]
    rk_in_per_species: Dict[str, float]
    rk_out_per_species: Dict[str, float]
    # paths
    run_dir: Path


def parse_summary_txt(path: Path) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    txt = path.read_text(encoding="utf-8", errors="replace").splitlines()

    acc_global = None
    no_det_global = None

    acc_sp: Dict[str, float] = {}
    nd_sp: Dict[str, float] = {}

    for line in txt:
        m1 = _SUMMARY_GLOBAL_RE.search(line)
        if m1:
            acc_global = float(m1.group(1)) / 100.0

        m2 = _SUMMARY_NODET_RE.search(line)
        if m2:
            no_det_global = float(m2.group(1)) / 100.0

        m3 = _SUMMARY_CLASS_RE.match(line.strip())
        if m3:
            sp = m3.group("sp").strip()
            acc = float(m3.group("acc")) / 100.0
            nd = float(m3.group("nd")) / 100.0
            acc_sp[sp] = acc
            nd_sp[sp] = nd

    if acc_global is None or no_det_global is None:
        raise RuntimeError(f"No pude parsear global ACC/NO_DETECT en: {path}")

    return acc_global, no_det_global, acc_sp, nd_sp


def parse_config_snapshot(path: Path) -> Tuple[float, Dict[str, float], Dict[str, float], Dict[str, float]]:
    snap = json.loads(path.read_text(encoding="utf-8"))
    q_out = float(snap.get("q_out"))
    rk = {k: float(v) for k, v in (snap.get("rk_per_species") or {}).items()}
    rk_in = {k: float(v) for k, v in (snap.get("rk_in_per_species") or {}).items()}
    rk_out = {k: float(v) for k, v in (snap.get("rk_out_per_species") or {}).items()}
    return q_out, rk, rk_in, rk_out


def spearman_safe(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Devuelve (rho, p). Si hay pocos puntos o series constantes, retorna NaN con p=NaN.
    """
    if len(x) < 3 or len(y) < 3:
        return (float("nan"), float("nan"))
    try:
        rho, p = spearmanr(x, y)
        rho = float(rho) if rho is not None else float("nan")
        p = float(p) if p is not None else float("nan")
        return rho, p
    except Exception:
        return (float("nan"), float("nan"))


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help="Directorio del grid (ej: ../outputs/qout_grid_20260213)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent

    grid_dir = Path(args.grid_dir).expanduser().resolve() if args.grid_dir else (here / "../outputs/qout_grid_20260213").resolve()
    if not grid_dir.exists():
        raise FileNotFoundError(f"No existe grid dir: {grid_dir}")

    print(f"🗂️ GRID DIR: {grid_dir}")

    run_dirs = sorted([d for d in grid_dir.iterdir() if d.is_dir() and d.name.startswith("qout_")])
    if not run_dirs:
        raise RuntimeError(f"No encontré carpetas qout_* en: {grid_dir}")

    records: List[RunRecord] = []

    for d in run_dirs:
        summary = d / "summary.txt"
        snap = d / "config_snapshot.json"

        if not summary.exists():
            print(f"⚠️ Falta summary.txt en {d} (skip)")
            continue
        if not snap.exists():
            print(f"⚠️ Falta config_snapshot.json en {d} (skip)")
            continue

        acc_g, nd_g, acc_sp, nd_sp = parse_summary_txt(summary)
        q_out, rk_sp, rk_in_sp, rk_out_sp = parse_config_snapshot(snap)

        records.append(
            RunRecord(
                q_out=q_out,
                acc_global=acc_g,
                no_detect_global=nd_g,
                acc_per_species=acc_sp,
                no_detect_per_species=nd_sp,
                rk_per_species=rk_sp,
                rk_in_per_species=rk_in_sp,
                rk_out_per_species=rk_out_sp,
                run_dir=d,
            )
        )

    if not records:
        raise RuntimeError("No pude cargar ningún run (records vacío).")

    # ordenar por q_out
    records = sorted(records, key=lambda r: r.q_out)

    # ---- GLOBAL correlations
    qouts = [r.q_out for r in records]
    accg = [r.acc_global for r in records]
    ndg = [r.no_detect_global for r in records]

    rho1, p1 = spearman_safe(qouts, ndg)
    rho2, p2 = spearman_safe(qouts, accg)

    print("\n" + "=" * 62)
    print("📊 SPEARMAN CORRELATIONS")
    print("=" * 62)
    print("\n◆ GLOBAL")
    print(f"corr(q_out, NO_DETECT_global) = {rho1: .3f}  (p={p1: .4f})")
    print(f"corr(q_out, ACC_global)       = {rho2: .3f}  (p={p2: .4f})")

    # ---- PER CLASS correlations (rk vs metrics)
    # reunir universo de especies (desde snapshots + summaries)
    species = sorted(set().union(*[set(r.rk_per_species.keys()) for r in records]).union(*[set(r.acc_per_species.keys()) for r in records]))

    rows_out = []

    # guardar global en tabla también
    rows_out.append({
        "scope": "GLOBAL",
        "species": "",
        "x": "q_out",
        "y": "NO_DETECT_global",
        "rho": rho1,
        "p": p1,
        "n_points": len(qouts),
    })
    rows_out.append({
        "scope": "GLOBAL",
        "species": "",
        "x": "q_out",
        "y": "ACC_global",
        "rho": rho2,
        "p": p2,
        "n_points": len(qouts),
    })

    print("\n◆ PER CLASS (rk vs metrics)")
    for sp in species:
        # x: rk (preferimos rk_per_species)
        x_rk = []
        y_acc = []
        y_nd = []
        x_q = []

        for r in records:
            if sp in r.rk_per_species and sp in r.acc_per_species and sp in r.no_detect_per_species:
                x_rk.append(r.rk_per_species[sp])
                y_acc.append(r.acc_per_species[sp])
                y_nd.append(r.no_detect_per_species[sp])
                x_q.append(r.q_out)

        if len(x_rk) < 3:
            print(f"- {sp}: rk/metrics insuficientes (n={len(x_rk)}).")
            continue

        rho_acc, p_acc = spearman_safe(x_rk, y_acc)
        rho_nd, p_nd = spearman_safe(x_rk, y_nd)
        rho_qrk, p_qrk = spearman_safe(x_q, x_rk)

        print(f"- {sp}:")
        print(f"    corr(rk, ACC)       = {rho_acc: .3f} (p={p_acc: .4f})")
        print(f"    corr(rk, NO_DETECT) = {rho_nd: .3f} (p={p_nd: .4f})")
        print(f"    corr(q_out, rk)     = {rho_qrk: .3f} (p={p_qrk: .4f})")

        rows_out.append({
            "scope": "PER_CLASS",
            "species": sp,
            "x": "rk",
            "y": "ACC",
            "rho": rho_acc,
            "p": p_acc,
            "n_points": len(x_rk),
        })
        rows_out.append({
            "scope": "PER_CLASS",
            "species": sp,
            "x": "rk",
            "y": "NO_DETECT",
            "rho": rho_nd,
            "p": p_nd,
            "n_points": len(x_rk),
        })
        rows_out.append({
            "scope": "PER_CLASS",
            "species": sp,
            "x": "q_out",
            "y": "rk",
            "rho": rho_qrk,
            "p": p_qrk,
            "n_points": len(x_rk),
        })

    df = pd.DataFrame(rows_out)
    out_csv = grid_dir / "spearman_table.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("\n✅ Tabla guardada en:")
    print(out_csv)


if __name__ == "__main__":
    main()