"""Cara: 478 puntos, 52 blendshapes, mirada, parpadeo, orientación y expresión."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# Contornos de cada ojo para el EAR (Eye Aspect Ratio).
OJO_DER = (33, 160, 158, 133, 153, 144)   # ojo derecho del sujeto
OJO_IZQ = (362, 385, 387, 263, 373, 380)
IRIS_DER = (469, 470, 471, 472)
IRIS_IZQ = (474, 475, 476, 477)


@dataclass
class EstadoCara:
    detectada: bool = False
    puntos: np.ndarray | None = None          # (478, 3) normalizado 0-1
    blendshapes: dict[str, float] = field(default_factory=dict)
    yaw: float = 0.0                          # + mira a su izquierda
    pitch: float = 0.0                        # + mira arriba
    roll: float = 0.0                         # + cabeza inclinada
    ear_izq: float = 0.0
    ear_der: float = 0.0
    ojo_izq_cerrado: bool = False
    ojo_der_cerrado: bool = False
    parpadeos: int = 0
    mirada: tuple[float, float] = (0.0, 0.0)  # (x, y) en -1..1
    direccion_mirada: str = "centro"
    expresion: str = "neutral"
    gestos: list[str] = field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None  # normalizado 0-1


def _ear(p: np.ndarray, idx) -> float:
    """Relación alto/ancho del ojo: ~0.3 abierto, ~0.1 cerrado."""
    a, b, c, d, e, f = (p[i][:2] for i in idx)
    ancho = np.linalg.norm(a - d)
    if ancho < 1e-6:
        return 0.0
    return (np.linalg.norm(b - f) + np.linalg.norm(c - e)) / (2 * ancho)


def _euler(m: np.ndarray) -> tuple[float, float, float]:
    """Ángulos (yaw, pitch, roll) en grados desde la matriz de transformación."""
    r = m[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x, y, z = math.atan2(-r[1, 2], r[1, 1]), math.atan2(-r[2, 0], sy), 0.0
    return math.degrees(y), -math.degrees(x), math.degrees(z)


# (nombre mostrado, blendshapes que lo activan, umbral, blendshapes que lo vetan)
# El veto hace falta porque el modelo sube browDown al sonreír fuerte: sin él,
# toda sonrisa amplia se reportaría además como ceño fruncido.
REGLAS_GESTO = [
    ("sonriendo",        ("mouthSmileLeft", "mouthSmileRight"), 0.35, ()),
    ("boca abierta",     ("jawOpen",),                          0.40, ()),
    ("beso",             ("mouthPucker",),                      0.55, ("jawOpen",)),
    ("cejas arriba",     ("browOuterUpLeft", "browOuterUpRight"), 0.45, ()),
    ("ceño fruncido",    ("browDownLeft", "browDownRight"),     0.62,
     ("mouthSmileLeft", "mouthSmileRight")),
    ("mejillas infladas", ("cheekPuff",),                       0.40, ()),
    ("lengua fuera",     ("tongueOut",),                        0.35, ()),
    ("mandíbula a un lado", ("jawLeft", "jawRight"),            0.45, ()),
    ("nariz arrugada",   ("noseSneerLeft", "noseSneerRight"),   0.55, ()),
]
VETO = 0.45  # a partir de aquí, un blendshape de veto anula la regla


def _expresion(b: dict[str, float]) -> str:
    """Emoción aproximada a partir de las unidades de acción facial."""
    g = lambda *k: max((b.get(x, 0.0) for x in k), default=0.0)
    sonrisa = g("mouthSmileLeft", "mouthSmileRight")
    ceja_ab = g("browDownLeft", "browDownRight")
    ceja_ar = g("browInnerUp", "browOuterUpLeft", "browOuterUpRight")
    boca = g("jawOpen")
    triste = g("mouthFrownLeft", "mouthFrownRight")

    if sonrisa > 0.45:
        return "feliz"
    if ceja_ar > 0.55 and boca > 0.35:
        return "sorprendido"
    if ceja_ab > 0.62 and sonrisa < 0.2 and g("mouthPressLeft", "mouthPressRight") > 0.3:
        return "enfadado"
    if triste > 0.35 and ceja_ar > 0.3:
        return "triste"
    if g("eyeSquintLeft", "eyeSquintRight") > 0.5:
        return "entrecerrando"
    return "neutral"


class AnalizadorCara:
    """Envuelve FaceLandmarker y traduce sus salidas a algo legible."""

    def __init__(self, modelo: str, max_caras: int = 1, umbral_ear: float = 0.15):
        opciones = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=modelo),
            running_mode=RunningMode.VIDEO,
            num_faces=max_caras,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = FaceLandmarker.create_from_options(opciones)
        self.umbral_ear = umbral_ear
        self._parpadeos = 0
        self._cerrados_prev = False

    def procesar(self, rgb: np.ndarray, ts_ms: int) -> EstadoCara:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(img, ts_ms)
        est = EstadoCara(parpadeos=self._parpadeos)
        if not res.face_landmarks:
            self._cerrados_prev = False
            return est

        h, w = rgb.shape[:2]
        lm = res.face_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        est.detectada = True
        est.puntos = pts

        est.bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()),
                    float(pts[:, 0].max()), float(pts[:, 1].max()))

        if res.face_blendshapes:
            est.blendshapes = {c.category_name: c.score for c in res.face_blendshapes[0]}
        if res.facial_transformation_matrixes:
            est.yaw, est.pitch, est.roll = _euler(
                np.array(res.facial_transformation_matrixes[0])
            )

        px = pts[:, :2] * np.array([w, h], dtype=np.float32)
        est.ear_der, est.ear_izq = _ear(px, OJO_DER), _ear(px, OJO_IZQ)
        b = est.blendshapes
        # El EAR es geométrico y el blendshape es del modelo: con los dos juntos
        # el guiño se detecta bien aunque la cara esté girada.
        est.ojo_der_cerrado = est.ear_der < self.umbral_ear or b.get("eyeBlinkRight", 0) > 0.55
        est.ojo_izq_cerrado = est.ear_izq < self.umbral_ear or b.get("eyeBlinkLeft", 0) > 0.55

        ambos = est.ojo_izq_cerrado and est.ojo_der_cerrado
        if ambos and not self._cerrados_prev:
            self._parpadeos += 1
        self._cerrados_prev = ambos
        est.parpadeos = self._parpadeos

        gx = (b.get("eyeLookOutLeft", 0) + b.get("eyeLookInRight", 0)
              - b.get("eyeLookInLeft", 0) - b.get("eyeLookOutRight", 0)) / 2
        gy = (b.get("eyeLookUpLeft", 0) + b.get("eyeLookUpRight", 0)
              - b.get("eyeLookDownLeft", 0) - b.get("eyeLookDownRight", 0)) / 2
        est.mirada = (float(np.clip(gx * 2, -1, 1)), float(np.clip(gy * 2, -1, 1)))

        if ambos:
            est.direccion_mirada = "ojos cerrados"
        else:
            hor = "izquierda" if gx > 0.22 else "derecha" if gx < -0.22 else ""
            ver = "arriba" if gy > 0.22 else "abajo" if gy < -0.22 else ""
            est.direccion_mirada = " ".join(x for x in (ver, hor) if x) or "al frente"

        est.gestos = [
            nombre for nombre, claves, umbral, vetos in REGLAS_GESTO
            if max((b.get(k, 0.0) for k in claves), default=0.0) > umbral
            and max((b.get(k, 0.0) for k in vetos), default=0.0) < VETO
        ]
        if est.ojo_izq_cerrado ^ est.ojo_der_cerrado:
            est.gestos.append("guiño ojo " + ("izq" if est.ojo_izq_cerrado else "dcho"))
        if abs(est.roll) > 18:
            est.gestos.append("cabeza inclinada")
        if abs(est.yaw) > 22:
            est.gestos.append("mirando de lado")
        if est.pitch > 15:
            est.gestos.append("cabeza arriba")
        elif est.pitch < -15:
            est.gestos.append("cabeza abajo")

        est.expresion = _expresion(b)
        return est

    def cerrar(self) -> None:
        self._lm.close()
