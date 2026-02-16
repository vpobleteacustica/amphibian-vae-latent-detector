#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
9200_run_qout_grid_with_snapshot.py

Corre el grid de q_out y guarda:
- summary.txt
- results.csv
- config_snapshot.json (incluye rk por especie)
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# === CONFIG ===
GRID_VALUES = [0.10, 0.15, 0.20, 0.25]
PROJECT_ROOT = Path("../").resolve()
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "qout_grid_20260213_with_rk"
CONFIG_PATH = PROJECT_ROOT / "config.json"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("📌 Project root:", PROJECT_ROOT)
print("📁 Output root:", OUTPUT_ROOT)
print()

for q_out in GRID_VALUES:

    print("="*60)
    print(f"Running q_out = {q_out}")
    print("="*60)

    # 1️⃣ Ejecutar benchmark (tu script original)
    subprocess.run([
        "bash",
        "run_qout_grid.sh",
        str(q_out)
    ])

    # 2️⃣ Leer config actual (ya contiene rk por especie)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # 3️⃣ Extraer rk por especie
    rk_per_species = {}
    for sp in config.get("species", {}):
        rk_per_species[sp] = config["species"][sp].get("rk", None)

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "q_out": q_out,
        "rk_per_species": rk_per_species,
        "full_config": config
    }

    # 4️⃣ Guardar snapshot
    run_dir = OUTPUT_ROOT / f"qout_{q_out:.2f}"
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_dir / "config_snapshot.json"
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=4)

    print("💾 Snapshot guardado en:", snapshot_path)
    print()

print("✅ GRID COMPLETADO CON SNAPSHOTS")