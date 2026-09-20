#!/usr/bin/env python
"""Enrola a alguien a partir de fotos sueltas.

    ./.venv/bin/python scripts/enrolar_fotos.py m1ke ~/Pictures/yo/*.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.identity import Reconocedor  # noqa: E402

CARAS = Path(__file__).resolve().parent.parent / "data" / "faces"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    nombre, rutas = sys.argv[1], sys.argv[2:]
    rec = Reconocedor(CARAS)
    vectores = []
    for r in rutas:
        img = cv2.imread(r)
        if img is None:
            print(f"  ✗ {r}: no se pudo leer")
            continue
        caras = rec.embeddings(img)
        if not caras:
            print(f"  ✗ {r}: sin cara")
            continue
        vectores.append(caras[0][0])
        print(f"  ✓ {r}")
    if not vectores:
        print("Ninguna cara utilizable.")
        return 1
    rec.base.guardar(nombre, np.array(vectores, dtype=np.float32))
    print(f"\n«{nombre}» guardado con {len(rec.base.gente[nombre])} muestras en {CARAS}")
    rec.cerrar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
