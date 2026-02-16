#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


RK_LINE = re.compile(
    r"✅\s+(?P<sp>[\w_]+):\s+rk_in=(?P<rk_in>[0-9.]+)\s+\|\s+rk_out=(?P<rk_out>[0-9.]+)\s+\|\s+rk=(?P<rk>[0-9.]+)"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=str, help="Path al run.log")
    p.add_argument("--q-in", required=True, type=float, dest="q_in")
    p.add_argument("--q-out", required=True, type=float, dest="q_out")
    p.add_argument("--out", required=True, type=str, help="Path de salida config_snapshot.json")
    return p.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not log_path.exists():
        raise FileNotFoundError(f"No existe log: {log_path}")

    txt = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    rk_per_species = {}
    rk_in_per_species = {}
    rk_out_per_species = {}

    for line in txt:
        m = RK_LINE.search(line)
        if m:
            sp = m.group("sp")
            rk_in = float(m.group("rk_in"))
            rk_out = float(m.group("rk_out"))
            rk = float(m.group("rk"))
            rk_in_per_species[sp] = rk_in
            rk_out_per_species[sp] = rk_out
            rk_per_species[sp] = rk

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "q_in": float(args.q_in),
        "q_out": float(args.q_out),
        "rk_in_per_species": rk_in_per_species,
        "rk_out_per_species": rk_out_per_species,
        "rk_per_species": rk_per_species,
        "source_log": str(log_path),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    if not rk_per_species:
        print("⚠️ WARNING: No se encontraron líneas rk en el log. Revisa el formato del print en 08_fit_radial_detector.py")

    print(f"✅ Snapshot escrito: {out_path}")


if __name__ == "__main__":
    main()