#!/usr/bin/env python
"""Lista o borra personas de la base local de caras."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.identity import BaseCaras  # noqa: E402

base = BaseCaras(Path(__file__).resolve().parent.parent / "data" / "faces")
if len(sys.argv) > 2 and sys.argv[1] == "borrar":
    print("borrado" if base.olvidar(sys.argv[2]) else "no existe", sys.argv[2])
else:
    for nombre, vecs in base.gente.items():
        print(f"{nombre:20s} {len(vecs)} muestras")
    if not base.gente:
        print("(vacío) — enrola con la tecla 'e' mientras corre la app")
