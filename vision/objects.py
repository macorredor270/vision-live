"""Objetos del entorno con YOLO, en un hilo aparte para no frenar el vídeo."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

# Traducción de las 80 clases COCO que trae YOLO de fábrica.
ES = {
    "person": "persona", "bicycle": "bicicleta", "car": "coche", "motorcycle": "moto",
    "airplane": "avión", "bus": "autobús", "train": "tren", "truck": "camión",
    "boat": "barco", "traffic light": "semáforo", "fire hydrant": "boca de incendios",
    "stop sign": "señal de stop", "parking meter": "parquímetro", "bench": "banco",
    "bird": "pájaro", "cat": "gato", "dog": "perro", "horse": "caballo",
    "sheep": "oveja", "cow": "vaca", "elephant": "elefante", "bear": "oso",
    "zebra": "cebra", "giraffe": "jirafa", "backpack": "mochila", "umbrella": "paraguas",
    "handbag": "bolso", "tie": "corbata", "suitcase": "maleta", "frisbee": "frisbee",
    "skis": "esquís", "snowboard": "snowboard", "sports ball": "pelota",
    "kite": "cometa", "baseball bat": "bate", "baseball glove": "guante",
    "skateboard": "monopatín", "surfboard": "tabla de surf", "tennis racket": "raqueta",
    "bottle": "botella", "wine glass": "copa", "cup": "taza", "fork": "tenedor",
    "knife": "cuchillo", "spoon": "cuchara", "bowl": "bol", "banana": "plátano",
    "apple": "manzana", "sandwich": "sándwich", "orange": "naranja",
    "broccoli": "brócoli", "carrot": "zanahoria", "hot dog": "perrito caliente",
    "pizza": "pizza", "donut": "donut", "cake": "tarta", "chair": "silla",
    "couch": "sofá", "potted plant": "planta", "bed": "cama", "dining table": "mesa",
    "toilet": "váter", "tv": "tele", "laptop": "portátil", "mouse": "ratón",
    "remote": "mando", "keyboard": "teclado", "cell phone": "móvil",
    "microwave": "microondas", "oven": "horno", "toaster": "tostadora",
    "sink": "fregadero", "refrigerator": "nevera", "book": "libro", "clock": "reloj",
    "vase": "jarrón", "scissors": "tijeras", "teddy bear": "peluche",
    "hair drier": "secador", "toothbrush": "cepillo de dientes",
}


@dataclass
class Objeto:
    nombre: str
    confianza: float
    bbox: tuple[int, int, int, int]


@dataclass
class EstadoObjetos:
    objetos: list[Objeto] = field(default_factory=list)
    resumen: dict[str, int] = field(default_factory=dict)
    ms: float = 0.0


class DetectorObjetos:
    """Corre YOLO en segundo plano sobre el último frame disponible."""

    def __init__(self, modelo: str = "yolo11n.pt", conf: float = 0.4,
                 dispositivo: str | None = None, imgsz: int = 640):
        from ultralytics import YOLO
        import torch

        self.dispositivo = dispositivo or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.modelo = YOLO(modelo)
        self.modelo.to(self.dispositivo)
        self.conf = conf
        self.imgsz = imgsz
        # Primera pasada en frío: en GPU tarda segundos. Mejor gastarlos aquí
        # que en el primer frame real.
        self.modelo.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                            imgsz=imgsz, verbose=False, device=self.dispositivo)

        self._entrada: np.ndarray | None = None
        self._estado = EstadoObjetos()
        self._lock = threading.Lock()
        self._hay_trabajo = threading.Event()
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def enviar(self, frame_bgr: np.ndarray) -> None:
        """Encola un frame; si el hilo sigue ocupado, el anterior se descarta."""
        with self._lock:
            self._entrada = frame_bgr
        self._hay_trabajo.set()

    @property
    def estado(self) -> EstadoObjetos:
        with self._lock:
            return self._estado

    def _bucle(self) -> None:
        import time

        while not self._parar.is_set():
            if not self._hay_trabajo.wait(timeout=0.2):
                continue
            self._hay_trabajo.clear()
            with self._lock:
                frame, self._entrada = self._entrada, None
            if frame is None:
                continue

            t0 = time.perf_counter()
            try:
                res = self.modelo.predict(
                    frame, conf=self.conf, imgsz=self.imgsz, verbose=False,
                    device=self.dispositivo,
                )[0]
            except Exception:
                continue

            objetos, resumen = [], {}
            nombres = res.names
            for caja in res.boxes:
                cls = nombres[int(caja.cls)]
                nombre = ES.get(cls, cls)
                x1, y1, x2, y2 = (int(v) for v in caja.xyxy[0])
                objetos.append(Objeto(nombre, float(caja.conf), (x1, y1, x2, y2)))
                resumen[nombre] = resumen.get(nombre, 0) + 1

            nuevo = EstadoObjetos(objetos, resumen, (time.perf_counter() - t0) * 1000)
            with self._lock:
                self._estado = nuevo

    def cerrar(self) -> None:
        self._parar.set()
        self._hay_trabajo.set()
        self._hilo.join(timeout=2.0)
