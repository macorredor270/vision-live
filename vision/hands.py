"""Manos: gestos del reconocedor de MediaPipe + gestos propios por geometría."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    RunningMode,
)

TRADUCCION = {
    "Closed_Fist": "puño",
    "Open_Palm": "palma abierta",
    "Pointing_Up": "señalando arriba",
    "Thumb_Down": "pulgar abajo",
    "Thumb_Up": "pulgar arriba",
    "Victory": "victoria",
    "ILoveYou": "te quiero",
    "None": "",
}
PUNTAS = (4, 8, 12, 16, 20)
NUDILLOS = (3, 6, 10, 14, 18)


@dataclass
class Mano:
    lado: str                      # "izquierda" / "derecha"
    puntos: np.ndarray             # (21, 3) normalizado
    gesto: str = ""
    confianza: float = 0.0
    dedos: int = 0
    extendidos: list[bool] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # normalizado


def _dedos_extendidos(p: np.ndarray, lado: str) -> list[bool]:
    """Un dedo está extendido si su punta queda más lejos de la muñeca que el nudillo."""
    muñeca = p[0][:2]
    fuera = []
    for punta, nudillo in zip(PUNTAS[1:], NUDILLOS[1:]):
        fuera.append(
            np.linalg.norm(p[punta][:2] - muñeca) > np.linalg.norm(p[nudillo][:2] - muñeca)
        )
    # El pulgar se abre de lado, no hacia arriba: se mira el eje X.
    pulgar = p[4][0] < p[3][0] if lado == "derecha" else p[4][0] > p[3][0]
    return [bool(pulgar), *(bool(x) for x in fuera)]


def _gestos_propios(p: np.ndarray, ext: list[bool]) -> list[str]:
    escala = np.linalg.norm(p[0][:2] - p[9][:2]) + 1e-6
    juntos = lambda a, b: np.linalg.norm(p[a][:2] - p[b][:2]) / escala
    g = []
    if juntos(4, 8) < 0.35 and all(ext[2:]):
        g.append("OK")
    if ext[1] and ext[4] and not ext[2] and not ext[3]:
        g.append("cuernos")
    if ext[1] and not any(ext[2:]) and not ext[0]:
        g.append("señalando")
    if juntos(4, 8) < 0.25 and not any(ext[1:]):
        g.append("pellizco")
    if all(ext):
        g.append("mano abierta")
    return g


@dataclass
class EstadoManos:
    manos: list[Mano] = field(default_factory=list)
    gestos: list[str] = field(default_factory=list)


class AnalizadorManos:
    def __init__(self, modelo: str, max_manos: int = 2):
        opciones = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=modelo),
            running_mode=RunningMode.VIDEO,
            num_hands=max_manos,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._gr = GestureRecognizer.create_from_options(opciones)

    def procesar(self, rgb: np.ndarray, ts_ms: int = 0) -> EstadoManos:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._gr.recognize_for_video(img, ts_ms)
        estado = EstadoManos()
        if not res.hand_landmarks:
            return estado

        h, w = rgb.shape[:2]
        for i, lm in enumerate(res.hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            etiqueta = res.handedness[i][0].category_name if res.handedness else "Right"
            lado = "izquierda" if etiqueta == "Left" else "derecha"
            ext = _dedos_extendidos(pts, lado)
            mano = Mano(lado=lado, puntos=pts, dedos=sum(ext), extendidos=ext)

            if res.gestures and res.gestures[i]:
                top = res.gestures[i][0]
                mano.gesto = TRADUCCION.get(top.category_name, top.category_name)
                mano.confianza = top.score

            mano.bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()),
                         float(pts[:, 0].max()), float(pts[:, 1].max()))

            propios = _gestos_propios(pts, ext)
            if not mano.gesto and propios:
                mano.gesto = propios[0]
            for g in filter(None, [mano.gesto, *propios]):
                etiq = f"{g} ({lado})"
                if etiq not in estado.gestos:
                    estado.gestos.append(etiq)
            estado.manos.append(mano)
        return estado

    def cerrar(self) -> None:
        self._gr.close()
