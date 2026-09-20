"""Fuentes de vídeo: webcam USB, cámara del móvil por IP, archivo o pantalla."""
from __future__ import annotations

import glob
import re
import threading
import time

import cv2


def listar_webcams() -> list[int]:
    """Índices de /dev/videoN que realmente entregan imagen."""
    encontradas = []
    for ruta in sorted(glob.glob("/dev/video*")):
        idx = int(re.sub(r"\D", "", ruta))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened() and cap.read()[0]:
            encontradas.append(idx)
        cap.release()
    return encontradas


def urls_candidatas(host: str) -> list[str]:
    """URLs que usan las apps de cámara IP más comunes.

    iPhone (IP Camera Lite): MJPEG en 8081 con admin/admin y RTSP en 8554.
    Android (IP Webcam): MJPEG en 8080.
    """
    if ":" in host:                      # el usuario ya puso puerto: respetarlo
        return [f"http://{host}/video", f"http://{host}/live",
                f"http://admin:admin@{host}/video", f"http://{host}/"]
    return [
        f"http://admin:admin@{host}:8081/video",   # IP Camera Lite (iOS)
        f"http://admin:admin@{host}:8081/live",
        f"http://{host}:8081/video",
        f"rtsp://admin:admin@{host}:8554/live",    # IP Camera Lite, RTSP
        f"rtsp://{host}:8554/live",
        f"rtsp://{host}:554/live",
        f"http://{host}:8080/video",               # IP Webcam (Android)
        f"http://{host}:8080/videofeed",
        f"http://{host}:4747/video",               # DroidCam
        f"http://{host}:8080/",                    # SimpleIPCamera y similares
    ]


def probar_url(url: str, timeout_ms: int = 3500) -> tuple[bool, str]:
    """Intenta abrir una URL y leer un frame. Devuelve (funciona, descripción)."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    try:
        if not cap.isOpened():
            return False, "no responde"
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, "conecta pero no da imagen"
        return True, f"{frame.shape[1]}x{frame.shape[0]}"
    finally:
        cap.release()


def buscar_camara_ip(host: str, verbose: bool = True, espera: float = 0.0) -> str | None:
    """Prueba las URLs típicas del host y devuelve la primera que dé imagen.

    Con `espera` repite el barrido ese número de segundos: las apps de cámara
    apagan el sensor cuando no hay ningún cliente conectado y tardan un momento
    en volver a emitir, así que conviene darles margen en vez de fallar.
    """
    limite = time.time() + espera
    primera_vuelta = True
    while True:
        for url in urls_candidatas(host):
            ok, detalle = probar_url(url)
            if verbose and primera_vuelta:
                print(f"  {'✓' if ok else '·'} {url}  ({detalle})")
            if ok:
                return url
        if time.time() >= limite:
            return None
        if primera_vuelta and verbose:
            print("  … esperando a que la cámara del móvil empiece a emitir "
                  "(abre la app y enciende el servidor)")
        primera_vuelta = False
        time.sleep(1.0)


def _normalizar(source: str) -> int | str:
    """'0' -> 0, 'auto' -> primera webcam, IP suelta -> URL de IP Webcam."""
    if source == "auto":
        cams = listar_webcams()
        if not cams:
            raise RuntimeError(
                "No hay ninguna webcam conectada.\n"
                "  Usa el móvil:  --source http://IP_DEL_MOVIL:8080/video\n"
                "  o un vídeo:    --source ruta/al/video.mp4"
            )
        return cams[0]
    if source.isdigit():
        return int(source)
    # Sólo la IP (o IP:puerto): probar las URLs de las apps de cámara conocidas.
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}(:\d+)?", source):
        print(f"→ buscando la cámara en {source}…")
        url = buscar_camara_ip(source, espera=45.0)
        if url is None:
            raise RuntimeError(
                f"No respondió ninguna cámara en {source}.\n"
                "  Comprueba que la app del móvil está emitiendo y que el móvil\n"
                "  está en el mismo WiFi. Si conoces la URL exacta, pásala entera:\n"
                "  --source http://usuario:clave@IP:PUERTO/ruta"
            )
        print(f"   usando {url}")
        return url
    return source


class VideoSource:
    """Captura en un hilo aparte y guarda sólo el frame más reciente.

    Así el bucle de inferencia nunca se come frames viejos acumulados en el
    buffer, que es lo que hace que una cámara IP se vea "con retraso".
    """

    ROTACIONES = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    def __init__(self, source: str = "auto", ancho: int = 1280, alto: int = 720,
                 fps: int = 30, rotar: int = 0):
        # El móvil emite en la orientación del sensor: se endereza aquí, antes
        # de que nadie mire la imagen, para que los modelos vean la cara derecha.
        self.rotacion = self.ROTACIONES.get(rotar % 360)
        self._host_original = source if re.fullmatch(
            r"\d{1,3}(\.\d{1,3}){3}(:\d+)?", source) else None
        self.spec = _normalizar(source)
        self.es_archivo = isinstance(self.spec, str) and not self.spec.startswith(
            ("http", "rtsp", "udp", "tcp")
        )
        api = cv2.CAP_V4L2 if isinstance(self.spec, int) else cv2.CAP_ANY
        # Una cámara de móvil puede tardar un instante en aceptar la siguiente
        # conexión (sobre todo justo después de buscarla), así que se reintenta.
        for intento in range(3):
            self.cap = cv2.VideoCapture(self.spec, api)
            if self.cap.isOpened():
                break
            self.cap.release()
            time.sleep(1.0)
        if isinstance(self.spec, int):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de vídeo: {self.spec}")

        self._frame = None
        self._n = 0
        self.estado = "conectando"
        self.reconexiones = 0
        self._lock = threading.Lock()
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()
        self._esperar_primer_frame()

    def _esperar_primer_frame(self, timeout: float = 10.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._frame is not None:
                return
            time.sleep(0.02)
        raise RuntimeError(f"La fuente {self.spec} no entrega imagen (¿permisos? ¿URL correcta?)")

    def _bucle(self) -> None:
        fallos = 0
        while not self._parar.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if not ok:
                fallos += 1
                if self.es_archivo:  # el vídeo terminó: volver al principio
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    fallos = 0
                    continue
                # El móvil corta el vídeo en cuanto se va el cliente o cambia de
                # red: en vez de morir, se vuelve a abrir sola las veces que haga
                # falta. La app de arriba ni se entera.
                if fallos >= 12:
                    self.estado = "reconectando"
                    if self._reconectar():
                        fallos = 0
                    continue
                time.sleep(0.05)
                continue
            fallos = 0
            self.estado = "ok"
            if self.rotacion is not None:
                frame = cv2.rotate(frame, self.rotacion)
            with self._lock:
                self._frame = frame
                self._n += 1
            if self.es_archivo:  # respetar la velocidad real del vídeo
                time.sleep(1 / max(self.cap.get(cv2.CAP_PROP_FPS) or 30, 1))

    def _reconectar(self) -> bool:
        """Reabre la fuente. Si era una IP, vuelve a buscar la URL que funcione."""
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(1.0)
        if self._parar.is_set():
            return False
        destino = self.spec
        if self._host_original:
            nueva = buscar_camara_ip(self._host_original, verbose=False)
            if nueva:
                destino = self.spec = nueva
        cap = cv2.VideoCapture(destino, cv2.CAP_V4L2 if isinstance(destino, int) else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            self.cap = cap
            self.reconexiones += 1
            self.estado = "ok"
            return True
        cap.release()
        return False

    def leer(self):
        """Devuelve (frame, numero_de_frame) o (None, n) si la fuente murió."""
        with self._lock:
            if self._frame is None:
                return None, self._n
            return self._frame.copy(), self._n

    @property
    def viva(self) -> bool:
        return not self._parar.is_set()

    def cerrar(self) -> None:
        self._parar.set()
        self._hilo.join(timeout=1.0)
        self.cap.release()
