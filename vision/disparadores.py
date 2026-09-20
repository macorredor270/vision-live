"""Detecta gestos "de disparo": palmada, gestos sostenidos…"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .hands import EstadoManos


class DetectorPalmada:
    """Una palmada = dos manos que se acercan rápido y acaban juntas.

    El detalle que lo hace funcionar: en el momento del choque MediaPipe suele
    perder una de las dos manos (se tapan entre ellas). Por eso el detector se
    "arma" cuando ve el acercamiento rápido y sigue disparando durante un
    instante aunque deje de ver las dos manos.
    """

    def __init__(self, ventana: float = 0.45, dist_choque: float = 1.8,
                 velocidad_min: float = 0.9, recorrido_min: float = 0.7,
                 dist_armado: float = 2.6, gracia: float = 0.5,
                 enfriamiento: float = 2.5):
        self.hist: deque[tuple[float, float]] = deque(maxlen=16)
        self.ventana = ventana            # cuánto pasado se mira, en segundos
        self.dist_choque = dist_choque    # distancia (en palmas) que cuenta como juntas
        self.velocidad_min = velocidad_min
        # Lo que importa no es de dónde parten las manos, sino cuánto terreno
        # se comen: así también vale la palmada corta, con las manos ya cerca.
        self.recorrido_min = recorrido_min
        self.dist_armado = dist_armado
        self.gracia = gracia              # margen tras armarse para aceptar el choque
        self.enfriamiento = enfriamiento
        self.ultimo = 0.0
        self.armado_hasta = 0.0
        self.distancia: float | None = None   # para enseñarlo en pantalla

    @property
    def armado(self) -> bool:
        return time.time() < self.armado_hasta

    def _distancia(self, manos: EstadoManos) -> float | None:
        if len(manos.manos) < 2:
            return None
        a, b = manos.manos[0], manos.manos[1]
        escala = (np.linalg.norm(a.puntos[0][:2] - a.puntos[9][:2]) +
                  np.linalg.norm(b.puntos[0][:2] - b.puntos[9][:2])) / 2
        if escala < 1e-6:
            return None
        return float(np.linalg.norm(a.puntos[9][:2] - b.puntos[9][:2]) / escala)

    def actualizar(self, manos: EstadoManos, ahora: float | None = None) -> bool:
        ahora = ahora if ahora is not None else time.time()
        d = self._distancia(manos)
        self.distancia = d
        enfriando = ahora - self.ultimo < self.enfriamiento

        if d is None:
            # Se perdió una mano. Si venían lanzadas la una contra la otra,
            # esa desaparición ES el choque.
            if not enfriando and ahora < self.armado_hasta:
                ultimo_d = self.hist[-1][1] if self.hist else 99.0
                if ultimo_d < self.dist_armado:
                    return self._disparar(ahora)
            return False

        self.hist.append((ahora, d))
        recientes = [(t, x) for t, x in self.hist if ahora - t <= self.ventana]
        if len(recientes) >= 2:
            (t0, d0), (t1, d1) = recientes[0], recientes[-1]
            velocidad = (d0 - d1) / max(t1 - t0, 1e-3)
            if velocidad > self.velocidad_min and (d0 - d1) > self.recorrido_min:
                self.armado_hasta = ahora + self.gracia

        if enfriando:
            return False
        if d < self.dist_choque and ahora < self.armado_hasta:
            return self._disparar(ahora)
        return False

    def _disparar(self, ahora: float) -> bool:
        self.ultimo = ahora
        self.armado_hasta = 0.0
        self.hist.clear()
        return True


class DetectorGestoSostenido:
    """Dispara cuando un gesto concreto se mantiene unos segundos.

    Sirve para gestos con los que no quieres falsos positivos: hay que
    sostenerlos a propósito, no basta con pasar por ahí.
    """

    def __init__(self, gesto: str, segundos: float = 1.2, enfriamiento: float = 4.0):
        self.gesto = gesto.lower()
        self.segundos = segundos
        self.enfriamiento = enfriamiento
        self.desde: float | None = None
        self.ultimo = 0.0
        self.progreso = 0.0

    def actualizar(self, gestos: list[str], ahora: float | None = None) -> bool:
        ahora = ahora if ahora is not None else time.time()
        presente = any(self.gesto in g.lower() for g in gestos)
        if not presente:
            self.desde = None
            self.progreso = 0.0
            return False
        if self.desde is None:
            self.desde = ahora
            return False
        self.progreso = min((ahora - self.desde) / self.segundos, 1.0)
        if self.progreso >= 1.0 and ahora - self.ultimo > self.enfriamiento:
            self.ultimo = ahora
            self.desde = None
            self.progreso = 0.0
            return True
        return False
