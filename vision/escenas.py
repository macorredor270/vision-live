"""Escenas: qué se abre cuando haces el gesto. Estilo «modo Stark»."""
from __future__ import annotations

import shlex
import subprocess
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paso:
    cmd: str
    espera: float = 0.0          # segundos a esperar DESPUÉS de lanzarlo
    shell: bool = False          # si el comando necesita sh -c (pipes, ||, &&)


@dataclass
class Escena:
    clave: str
    nombre: str
    disparador: str              # "palmada", o el nombre de un gesto sostenido
    pasos: list[Paso] = field(default_factory=list)
    sostener: float = 1.2        # segundos que hay que aguantar un gesto
    enfriamiento: float = 6.0


def cargar(ruta: Path) -> list[Escena]:
    with open(ruta, "rb") as f:
        datos = tomllib.load(f)
    escenas = []
    for clave, e in datos.items():
        if not isinstance(e, dict) or "disparador" not in e:
            continue
        escenas.append(Escena(
            clave=clave,
            nombre=e.get("nombre", clave),
            disparador=e["disparador"],
            sostener=float(e.get("sostener", 1.2)),
            enfriamiento=float(e.get("enfriamiento", 6.0)),
            pasos=[Paso(cmd=p["cmd"], espera=float(p.get("espera", 0.0)),
                        shell=bool(p.get("shell", False)))
                   for p in e.get("pasos", [])],
        ))
    return escenas


class Lanzador:
    """Ejecuta una escena en segundo plano para no congelar el vídeo."""

    def __init__(self, seco: bool = False):
        self.seco = seco             # modo prueba: enseña lo que haría, sin hacerlo
        self.registro: list[str] = []

    def lanzar(self, escena: Escena) -> None:
        threading.Thread(target=self._ejecutar, args=(escena,), daemon=True).start()

    def _ejecutar(self, escena: Escena) -> None:
        for paso in escena.pasos:
            self._paso(paso)
            if paso.espera:
                time.sleep(paso.espera)

    def _paso(self, paso: Paso) -> None:
        etiqueta = paso.cmd if len(paso.cmd) < 90 else paso.cmd[:87] + "…"
        self.registro.append(etiqueta)
        print(f"   ▸ {etiqueta}", flush=True)
        if self.seco:
            return
        try:
            # Se lanza directamente, sin pasar por el compositor: la API de
            # dispatchers de Hyprland cambió en la 0.56 y lanzar por ahí es
            # frágil. Dónde aparece cada ventana se decide con reglas en
            # hyprland.conf, que es donde corresponde.
            if paso.shell:
                subprocess.Popen(["sh", "-c", paso.cmd], start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(shlex.split(paso.cmd), start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"   ✗ falló: {e}", flush=True)
