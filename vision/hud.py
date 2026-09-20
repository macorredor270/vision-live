"""HUD: tarjetas de cristal sobre el vídeo, escaladas al tamaño de la imagen."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pose import CONEXIONES as CONEXIONES_POSE

# Paleta (RGB). Un acento por tipo de información y nada más.
TINTA        = (236, 240, 244)
TINTA_SUAVE  = (150, 158, 168)
MENTA        = (110, 231, 168)
CIELO        = (125, 211, 252)
AMBAR        = (251, 191, 114)
ORQUIDEA     = (240, 171, 252)
CARMIN       = (248, 113, 113)
CRISTAL      = (14, 17, 22)

CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
OJO_DER = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
OJO_IZQ = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
CEJA_DER = [70, 63, 105, 66, 107]
CEJA_IZQ = [300, 293, 334, 296, 336]
LABIO_EXT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
             37, 39, 40, 185]
NARIZ = [168, 6, 197, 195, 5, 4]
IRIS_DER, IRIS_IZQ = (469, 470, 471, 472), (474, 475, 476, 477)


def _ruta_fuente(peso: str) -> str:
    consulta = {"bold": "sans-serif:bold", "medium": "sans-serif:medium"}.get(peso, "sans-serif")
    try:
        salida = subprocess.run(["fc-match", "-f", "%{file}", consulta],
                                capture_output=True, text=True, timeout=3).stdout.strip()
        if salida:
            return salida
    except Exception:
        pass
    return "/usr/share/fonts/gnu-free/FreeSans.otf"


@lru_cache(maxsize=32)
def _fuente(tam: int, peso: str = "regular") -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_ruta_fuente(peso), tam)
    except Exception:
        return ImageFont.load_default()


def _fusionar(cajas: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Une las regiones que se solapan para no componer dos veces lo mismo."""
    restantes = list(cajas)
    salida: list[list[int]] = []
    for caja in restantes:
        for actual in salida:
            if (caja[0] < actual[2] and caja[2] > actual[0]
                    and caja[1] < actual[3] and caja[3] > actual[1]):
                actual[0] = min(actual[0], caja[0])
                actual[1] = min(actual[1], caja[1])
                actual[2] = max(actual[2], caja[2])
                actual[3] = max(actual[3], caja[3])
                break
        else:
            salida.append(list(caja))
    return [tuple(c) for c in salida]


@dataclass
class EstadoHud:
    """Todo lo que el HUD necesita saber para pintarse."""
    fps: float = 0.0
    cara: object | None = None
    identidad: str = ""
    similitud: float = 0.0
    gestos: list[str] = field(default_factory=list)
    objetos: dict[str, int] = field(default_factory=dict)
    escenas: list[tuple[str, str, float]] = field(default_factory=list)  # nombre, gesto, 0-1
    aviso: str = ""
    grabando: bool = False


class Hud:
    def __init__(self, ancho: int, alto: int):
        self.reescalar(ancho, alto)

    def reescalar(self, ancho: int, alto: int) -> None:
        self.w, self.h = ancho, alto
        self._sucio: list[tuple[int, int, int, int]] = []
        # Todo se mide en esta unidad: así el HUD se ve igual en 720p que en
        # vertical de 1920, sin textos diminutos ni paneles que tapan la cara.
        self.u = max(min(ancho, alto) / 720.0, 0.75)
        self.margen = int(22 * self.u)
        self.radio = int(16 * self.u)

    def t(self, n: float) -> int:
        return max(int(n * self.u), 8)

    # ---------- primitivas ----------

    def _marcar(self, caja) -> None:
        """Apunta una región pintada: al final sólo se compone eso, no el frame entero."""
        x0, y0, x1, y1 = (int(v) for v in caja)
        h = int(6 * self.u)                      # margen para bordes y antialias
        self._sucio.append((max(x0 - h, 0), max(y0 - h, 0),
                            min(x1 + h, self.w), min(y1 + h, self.h)))

    def _cristal(self, img: np.ndarray, caja: tuple[int, int, int, int]) -> None:
        """Desenfoca el vídeo bajo la tarjeta y lo oscurece: efecto cristal."""
        x0, y0, x1, y1 = caja
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        self._marcar((x0, y0, x1, y1))
        region = img[y0:y1, x0:x1]
        k = max(int(17 * self.u) | 1, 7)
        borroso = cv2.blur(region, (k, k))
        oscuro = np.full_like(region, CRISTAL[::-1], dtype=np.uint8)
        region[:] = cv2.addWeighted(borroso, 0.38, oscuro, 0.62, 0)

    def _tarjeta(self, d: ImageDraw.ImageDraw, caja, alfa_borde: int = 40) -> None:
        self._marcar(caja)
        d.rounded_rectangle(caja, self.radio, outline=(255, 255, 255, alfa_borde),
                            width=max(int(self.u), 1))

    def _texto(self, d, xy, txt, color=TINTA, tam=14, peso="regular", alfa=255, anclaje="la"):
        d.text(xy, txt, font=_fuente(self.t(tam), peso), fill=(*color, alfa), anchor=anclaje)

    def _ancho(self, txt: str, tam: int, peso: str = "regular") -> int:
        f = _fuente(self.t(tam), peso)
        return int(f.getbbox(txt)[2] - f.getbbox(txt)[0])

    def _chip(self, d, x, y, txt, color, tam=13, relleno=28):
        """Píldora con texto: la unidad de información del HUD."""
        alto = self.t(tam) + int(14 * self.u)
        ancho = self._ancho(txt, tam, "medium") + int(26 * self.u)
        self._marcar((x, y, x + ancho, y + alto))
        d.rounded_rectangle((x, y, x + ancho, y + alto), alto // 2,
                            fill=(*color, relleno), outline=(*color, 110),
                            width=max(int(self.u), 1))
        self._texto(d, (x + ancho / 2, y + alto / 2), txt, color, tam, "medium", anclaje="mm")
        return ancho, alto

    # ---------- capas sobre el vídeo ----------

    def cara(self, img: np.ndarray, est, detalle: str = "normal") -> None:
        pts = est.puntos
        px = (pts[:, :2] * [self.w, self.h]).astype(np.int32)
        grosor = max(int(self.u), 1)

        if detalle == "completo":
            for x, y in px:
                cv2.circle(img, (x, y), grosor, (90, 150, 110), -1, cv2.LINE_AA)

        for trazo, color in ((LABIO_EXT, ORQUIDEA), (NARIZ, (120, 140, 160)),
                             (CEJA_DER, CIELO), (CEJA_IZQ, CIELO)):
            cv2.polylines(img, [px[trazo]], False, color[::-1], grosor, cv2.LINE_AA)

        for contorno, iris in ((OJO_DER, IRIS_DER), (OJO_IZQ, IRIS_IZQ)):
            cv2.polylines(img, [px[contorno]], True, CIELO[::-1], grosor, cv2.LINE_AA)
            if len(px) > 477:
                centro = px[list(iris)].mean(axis=0).astype(int)
                radio = max(int(np.linalg.norm(px[iris[0]] - px[iris[2]]) / 2), 2)
                cv2.circle(img, tuple(centro), radio, AMBAR[::-1], grosor + 1, cv2.LINE_AA)
                gx, gy = est.mirada
                punta = (int(centro[0] - gx * 42 * self.u), int(centro[1] - gy * 34 * self.u))
                cv2.arrowedLine(img, tuple(centro), punta, AMBAR[::-1], grosor + 1,
                                cv2.LINE_AA, tipLength=0.3)

        if est.bbox:
            x1, y1, x2, y2 = est.bbox
            self._marco(img, (int(x1 * self.w), int(y1 * self.h),
                              int(x2 * self.w), int(y2 * self.h)),
                        MENTA, margen=int(18 * self.u))

    def _marco(self, img, bbox, color, margen: int = 0) -> None:
        """Cuatro esquinas en vez de un rectángulo: tapa mucho menos."""
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = x1 - margen, y1 - margen, x2 + margen, y2 + margen
        largo = max(int((x2 - x1) * 0.18), int(12 * self.u))
        grosor = max(int(2 * self.u), 2)
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * largo, cy), color[::-1], grosor, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * largo), color[::-1], grosor, cv2.LINE_AA)

    def manos(self, img: np.ndarray, manos) -> None:
        grosor = max(int(2 * self.u), 2)
        for mano in manos:
            px = (mano.puntos[:, :2] * [self.w, self.h]).astype(np.int32)
            for a, b in CONEXIONES_MANO:
                cv2.line(img, tuple(px[a]), tuple(px[b]), ORQUIDEA[::-1], grosor, cv2.LINE_AA)
            for i in (4, 8, 12, 16, 20):
                cv2.circle(img, tuple(px[i]), grosor + 2, (255, 255, 255), -1, cv2.LINE_AA)

    def pose(self, img: np.ndarray, pts) -> None:
        px = (pts[:, :2] * [self.w, self.h]).astype(np.int32)
        grosor = max(int(2 * self.u), 2)
        for a, b in CONEXIONES_POSE:
            if pts[a][3] > 0.4 and pts[b][3] > 0.4:
                cv2.line(img, tuple(px[a]), tuple(px[b]), (200, 210, 120), grosor, cv2.LINE_AA)

    def objetos(self, img: np.ndarray, lista, d: ImageDraw.ImageDraw) -> None:
        grosor = max(int(1.5 * self.u), 1)
        for o in lista:
            x1, y1, x2, y2 = o.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), AMBAR[::-1], grosor, cv2.LINE_AA)
            txt = f"{o.nombre} {o.confianza:.0%}"
            alto = self.t(12) + int(10 * self.u)
            ancho = self._ancho(txt, 12, "medium") + int(18 * self.u)
            y = max(y1 - alto - int(5 * self.u), 0)
            self._marcar((x1, y, x1 + ancho, y + alto))
            d.rounded_rectangle((x1, y, x1 + ancho, y + alto), alto // 2, fill=(*AMBAR, 200))
            self._texto(d, (x1 + ancho / 2, y + alto / 2), txt, (18, 20, 24), 12,
                        "medium", anclaje="mm")

    # ---------- tarjetas ----------

    def panel(self, img: np.ndarray, d: ImageDraw.ImageDraw, e: EstadoHud) -> None:
        m = self.margen
        ancho = int(330 * self.u)
        cara = e.cara
        filas = 3 if (cara is not None and cara.detectada) else 1
        alto = int((58 + filas * 34) * self.u)
        caja = (m, m, m + ancho, m + alto)

        self._cristal(img, caja)
        self._tarjeta(d, caja)

        x, y = m + int(20 * self.u), m + int(16 * self.u)
        self._texto(d, (x, y), "VISION LIVE", MENTA, 13, "bold")
        self._texto(d, (m + ancho - int(20 * self.u), y), f"{e.fps:.0f} FPS",
                    TINTA, 13, "bold", anclaje="ra")
        y += int(26 * self.u)

        if cara is not None and cara.detectada:
            nombre = e.identidad or "desconocido"
            self._texto(d, (x, y), nombre, AMBAR if e.similitud else TINTA_SUAVE, 21, "bold")
            if e.similitud:
                self._texto(d, (m + ancho - int(20 * self.u), y + int(8 * self.u)),
                            f"{e.similitud:.0%}", TINTA_SUAVE, 12, anclaje="ra")
            y += int(34 * self.u)
            self._dato(d, x, y, "expresión", cara.expresion, ORQUIDEA)
            self._dato(d, x + int(165 * self.u), y, "mirada", cara.direccion_mirada, CIELO)
            y += int(34 * self.u)
            self._dato(d, x, y, "parpadeos", str(cara.parpadeos), TINTA)
            self._dato(d, x + int(165 * self.u), y, "cabeza",
                       f"{cara.yaw:+.0f}° {cara.pitch:+.0f}°", TINTA)
        else:
            self._texto(d, (x, y), "buscando una cara…", TINTA_SUAVE, 15)

    def _dato(self, d, x, y, etiqueta, valor, color) -> None:
        self._texto(d, (x, y), etiqueta.upper(), TINTA_SUAVE, 10, "medium", alfa=190)
        self._texto(d, (x, y + int(13 * self.u)), valor, color, 15, "medium")

    def chips_gestos(self, img: np.ndarray, d: ImageDraw.ImageDraw, e: EstadoHud) -> None:
        if not e.gestos:
            return
        gestos = e.gestos[:5]
        alto = self.t(13) + int(14 * self.u)
        hueco = int(9 * self.u)
        anchos = [self._ancho(g, 13, "medium") + int(26 * self.u) for g in gestos]
        total = sum(anchos) + hueco * (len(gestos) - 1)
        x = (self.w - total) // 2
        y = self.h - self.margen - alto
        self._cristal(img, (x - hueco, y - hueco // 2, x + total + hueco, y + alto + hueco // 2))
        for g, ancho in zip(gestos, anchos):
            self._chip(d, x, y, g, MENTA)
            x += ancho + hueco

    def chips_entorno(self, img: np.ndarray, d: ImageDraw.ImageDraw, e: EstadoHud) -> None:
        if not e.objetos:
            return
        items = sorted(e.objetos.items(), key=lambda kv: -kv[1])[:5]
        textos = [f"{n} ×{c}" if c > 1 else n for n, c in items]
        alto = self.t(12) + int(12 * self.u)
        x, y = self.margen, self.margen + int(0 * self.u)
        # se apila bajo el panel principal
        y = self.margen + int((58 + (3 if (e.cara is not None and e.cara.detectada) else 1) * 34) * self.u) + int(12 * self.u)
        for txt in textos:
            ancho, _ = self._chip(d, x, y, txt, AMBAR, tam=12, relleno=22)
            y += alto + int(8 * self.u)

    def panel_escenas(self, img: np.ndarray, d: ImageDraw.ImageDraw, e: EstadoHud) -> None:
        if not e.escenas:
            return
        ancho = int(250 * self.u)
        alto = int((26 + 30 * len(e.escenas)) * self.u)
        x0 = self.w - self.margen - ancho
        caja = (x0, self.margen, x0 + ancho, self.margen + alto)
        self._cristal(img, caja)
        self._tarjeta(d, caja)
        x, y = x0 + int(18 * self.u), self.margen + int(14 * self.u)
        for nombre, gesto, progreso in e.escenas:
            activo = progreso > 0.02
            self._texto(d, (x, y), nombre, TINTA if activo else TINTA_SUAVE, 13, "medium")
            self._texto(d, (x0 + ancho - int(18 * self.u), y + int(2 * self.u)), gesto,
                        MENTA if activo else TINTA_SUAVE, 11, anclaje="ra")
            barra_y = y + int(19 * self.u)
            largo = ancho - int(36 * self.u)
            d.rounded_rectangle((x, barra_y, x + largo, barra_y + int(3 * self.u)),
                                int(2 * self.u), fill=(255, 255, 255, 30))
            if activo:
                d.rounded_rectangle(
                    (x, barra_y, x + int(largo * min(progreso, 1.0)), barra_y + int(3 * self.u)),
                    int(2 * self.u), fill=(*MENTA, 230))
            y += int(30 * self.u)

    def aviso(self, img: np.ndarray, d: ImageDraw.ImageDraw, texto: str) -> None:
        alto = self.t(17) + int(24 * self.u)
        ancho = self._ancho(texto, 17, "bold") + int(52 * self.u)
        x = (self.w - ancho) // 2
        y = int(self.h * 0.14)
        self._cristal(img, (x, y, x + ancho, y + alto))
        self._tarjeta(d, (x, y, x + ancho, y + alto), alfa_borde=90)
        self._texto(d, (x + ancho / 2, y + alto / 2), texto, MENTA, 17, "bold", anclaje="mm")

    def grabando(self, d: ImageDraw.ImageDraw) -> None:
        r = int(7 * self.u)
        cx, cy = self.w - self.margen - r, self.h - self.margen - r
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*CARMIN, 235))

    # ---------- composición ----------

    def dibujar(self, frame: np.ndarray, e: EstadoHud, objetos_lista=None) -> np.ndarray:
        if (frame.shape[1], frame.shape[0]) != (self.w, self.h):
            self.reescalar(frame.shape[1], frame.shape[0])
        self._sucio = []

        capa = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        d = ImageDraw.Draw(capa)

        if objetos_lista:
            self.objetos(frame, objetos_lista, d)
        self.panel(frame, d, e)
        self.chips_entorno(frame, d, e)
        self.panel_escenas(frame, d, e)
        self.chips_gestos(frame, d, e)
        if e.aviso:
            self.aviso(frame, d, e.aviso)
        if e.grabando:
            self.grabando(d)

        if not self._sucio:
            return frame

        rgba = np.asarray(capa)                      # (h, w, 4) RGB + alfa
        for x0, y0, x1, y1 in _fusionar(self._sucio):
            trozo = rgba[y0:y1, x0:x1]
            alfa = trozo[:, :, 3:4].astype(np.float32) / 255.0
            if not alfa.any():
                continue
            color = trozo[:, :, 2::-1].astype(np.float32)      # RGB → BGR
            destino = frame[y0:y1, x0:x1]
            destino[:] = (destino.astype(np.float32) * (1 - alfa) + color * alfa).astype(np.uint8)
        return frame
