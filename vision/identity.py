"""Identidad: ¿de quién es esta cara? Embeddings ArcFace, comparación local."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Identidad:
    nombre: str = "desconocido"
    similitud: float = 0.0
    bbox: tuple[int, int, int, int] | None = None


class BaseCaras:
    """Las caras guardadas viven en data/faces/<nombre>.npy. Nada sale del PC."""

    def __init__(self, carpeta: Path):
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.gente: dict[str, np.ndarray] = {}
        self.cargar()

    def cargar(self) -> None:
        self.gente = {}
        for f in sorted(self.carpeta.glob("*.npy")):
            vecs = np.load(f)
            if vecs.ndim == 1:
                vecs = vecs[None, :]
            self.gente[f.stem] = vecs

    def guardar(self, nombre: str, embeddings: np.ndarray) -> None:
        ruta = self.carpeta / f"{nombre}.npy"
        if ruta.exists():
            embeddings = np.vstack([np.load(ruta), embeddings])
        np.save(ruta, embeddings.astype(np.float32))
        self.cargar()

    def olvidar(self, nombre: str) -> bool:
        ruta = self.carpeta / f"{nombre}.npy"
        if ruta.exists():
            ruta.unlink()
            self.cargar()
            return True
        return False

    def buscar(self, emb: np.ndarray, umbral: float) -> tuple[str, float]:
        """Coseno contra cada muestra guardada; gana la más parecida."""
        mejor, puntos = "desconocido", 0.0
        e = emb / (np.linalg.norm(emb) + 1e-9)
        for nombre, vecs in self.gente.items():
            v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
            s = float(np.max(v @ e))
            if s > puntos:
                mejor, puntos = nombre, s
        return (mejor, puntos) if puntos >= umbral else ("desconocido", puntos)


class Reconocedor:
    """InsightFace (ArcFace) en un hilo: identifica sin bajar los FPS del vídeo."""

    def __init__(self, carpeta_caras: Path, umbral: float = 0.42,
                 det_size: int = 480, proveedores: list[str] | None = None):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        self.base = BaseCaras(carpeta_caras)
        self.umbral = umbral
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            # Pedir un provider que no esté compilado hace fallar a onnxruntime,
            # así que sólo se piden los que realmente existen.
            providers=[p for p in (proveedores or ["CUDAExecutionProvider",
                                                   "CPUExecutionProvider"])
                       if p in ort.get_available_providers()] or ["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

        self._entrada: np.ndarray | None = None
        self._resultado: list[Identidad] = []
        self._lock = threading.Lock()
        self._hay_trabajo = threading.Event()
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def embeddings(self, frame_bgr: np.ndarray) -> list[tuple[np.ndarray, tuple]]:
        caras = self.app.get(frame_bgr)
        return [
            (c.normed_embedding, tuple(int(v) for v in c.bbox))
            for c in sorted(caras, key=lambda c: -(c.bbox[2] - c.bbox[0]))
        ]

    def enviar(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._entrada = frame_bgr
        self._hay_trabajo.set()

    @property
    def resultado(self) -> list[Identidad]:
        with self._lock:
            return list(self._resultado)

    def recargar(self) -> None:
        self.base.cargar()

    def _bucle(self) -> None:
        while not self._parar.is_set():
            if not self._hay_trabajo.wait(timeout=0.2):
                continue
            self._hay_trabajo.clear()
            with self._lock:
                frame, self._entrada = self._entrada, None
            if frame is None:
                continue
            try:
                caras = self.embeddings(frame)
            except Exception:
                continue
            ids = []
            for emb, bbox in caras:
                nombre, sim = self.base.buscar(emb, self.umbral)
                ids.append(Identidad(nombre, sim, bbox))
            with self._lock:
                self._resultado = ids

    def cerrar(self) -> None:
        self._parar.set()
        self._hay_trabajo.set()
        self._hilo.join(timeout=2.0)
