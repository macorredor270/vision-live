"""Postura corporal: 33 puntos y una lectura simple de lo que está haciendo."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

CONEXIONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32),
]


@dataclass
class EstadoPose:
    detectada: bool = False
    puntos: np.ndarray | None = None
    acciones: list[str] = field(default_factory=list)


class AnalizadorPose:
    def __init__(self, modelo: str):
        opciones = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=modelo),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
        )
        self._pl = PoseLandmarker.create_from_options(opciones)

    def procesar(self, rgb: np.ndarray, ts_ms: int) -> EstadoPose:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._pl.detect_for_video(img, ts_ms)
        est = EstadoPose()
        if not res.pose_landmarks:
            return est
        p = np.array([[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks[0]],
                     dtype=np.float32)
        est.detectada = True
        est.puntos = p

        hombro_y = (p[11][1] + p[12][1]) / 2
        if p[15][1] < hombro_y and p[15][3] > 0.5:
            est.acciones.append("mano izq levantada")
        if p[16][1] < hombro_y and p[16][3] > 0.5:
            est.acciones.append("mano dcha levantada")
        if abs(p[11][1] - p[12][1]) > 0.09:
            est.acciones.append("hombros torcidos")
        cadera_y = (p[23][1] + p[24][1]) / 2
        if cadera_y - hombro_y < 0.22 and p[23][3] > 0.5:
            est.acciones.append("inclinado hacia delante")
        return est

    def cerrar(self) -> None:
        self._pl.close()
