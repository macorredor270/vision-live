"""Bucle principal: captura, analiza y dibuja todo en tiempo real."""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .hud import EstadoHud, Hud
from .disparadores import DetectorGestoSostenido, DetectorPalmada
from .escenas import Lanzador, cargar as cargar_escenas
from .face import AnalizadorCara
from .hands import AnalizadorManos
from .objects import DetectorObjetos
from .pose import AnalizadorPose
from .sources import VideoSource, buscar_camara_ip, listar_webcams

RAIZ = Path(__file__).resolve().parent.parent
MODELOS = RAIZ / "models"
CARAS = RAIZ / "data" / "faces"
CAPTURAS = RAIZ / "data" / "capturas"


def argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vision-live",
        description="Reconocimiento en tiempo real de cara, ojos, gestos y entorno. Todo local.",
    )
    p.add_argument("--source", default="auto",
                   help="'auto', índice de webcam (0), IP del móvil (192.168.1.40), "
                        "URL http/rtsp o ruta de un vídeo")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-mirror", action="store_true", help="no voltear la imagen como un espejo")
    p.add_argument("--rotar", type=int, default=0, choices=[0, 90, 180, 270],
                   help="girar la imagen (el móvil suele llegar tumbado: prueba 90 o 270)")
    p.add_argument("--no-objects", action="store_true", help="desactivar YOLO")
    p.add_argument("--no-identity", action="store_true", help="desactivar el reconocimiento de quién eres")
    p.add_argument("--pose", action="store_true", help="activar el esqueleto corporal")
    p.add_argument("--faces", type=int, default=1, help="cuántas caras seguir a la vez")
    p.add_argument("--yolo", default="yolo11n.pt", help="yolo11n/s/m/l/x.pt (n = más rápido)")
    p.add_argument("--yolo-conf", type=float, default=0.4)
    p.add_argument("--yolo-cada", type=int, default=3, help="lanzar YOLO cada N frames")
    p.add_argument("--id-cada", type=int, default=15, help="reidentificar la cara cada N frames")
    p.add_argument("--umbral-id", type=float, default=0.42, help="parecido mínimo para dar un nombre")
    p.add_argument("--detalle-malla", choices=["normal", "completo"], default="normal")
    p.add_argument("--analisis", type=int, default=640, metavar="PX",
                   help="lado mayor al que se reduce la imagen para analizarla "
                        "(el dibujo sigue a resolución completa; 0 = sin reducir)")
    p.add_argument("--escenas", nargs="?", const="escenas.toml", default=None,
                   metavar="ARCHIVO",
                   help="activar los gestos que lanzan apps (por defecto escenas.toml)")
    p.add_argument("--escenas-seco", action="store_true",
                   help="con --escenas: enseña qué lanzaría, sin lanzarlo")
    p.add_argument("--list-cams", action="store_true", help="listar webcams y salir")
    p.add_argument("--probe", metavar="IP",
                   help="probar las URLs típicas de cámara IP en esa IP y salir")
    p.add_argument("--benchmark", type=int, metavar="N",
                   help="procesar N frames sin abrir ventana y medir el rendimiento")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = argumentos(argv)

    if args.probe:
        print(f"Probando las URLs conocidas en {args.probe}:")
        url = buscar_camara_ip(args.probe)
        print(f"\n→ Funciona: ./run --source '{url}'" if url else
              "\nNinguna respondió. ¿Está la app emitiendo y el móvil en el mismo WiFi?")
        return 0 if url else 1

    if args.list_cams:
        cams = listar_webcams()
        print("Webcams:", ", ".join(f"/dev/video{i}" for i in cams) if cams else "ninguna")
        if not cams:
            print("Usa el móvil con la app 'IP Webcam':  --source 192.168.1.XX")
        return 0

    CAPTURAS.mkdir(parents=True, exist_ok=True)
    print("→ abriendo fuente de vídeo…")
    fuente = VideoSource(args.source, args.width, args.height, rotar=args.rotar)
    print(f"   {fuente.spec}")

    detector = None
    if not args.no_objects:
        print("→ cargando YOLO…")
        ruta_yolo = args.yolo if Path(args.yolo).exists() else str(MODELOS / args.yolo)
        detector = DetectorObjetos(ruta_yolo, args.yolo_conf)
        print(f"   {args.yolo} en {detector.dispositivo}")

    reconocedor = None
    if not args.no_identity:
        print("→ cargando reconocimiento facial…")
        try:
            from .identity import Reconocedor
            reconocedor = Reconocedor(CARAS, umbral=args.umbral_id)
            print(f"   {len(reconocedor.base.gente)} persona(s) en la base: "
                  f"{', '.join(reconocedor.base.gente) or '—'}")
        except Exception as e:
            print(f"   no disponible ({e}); sigo sin identificar")

    # MediaPipe se crea el último: así los modelos pesados (torch, onnxruntime)
    # ya están en memoria y el primer frame del bucle no se pierde cargándolos.
    cara = AnalizadorCara(str(MODELOS / "face_landmarker.task"), max_caras=args.faces)
    manos = AnalizadorManos(str(MODELOS / "gesture_recognizer.task"))
    poses = AnalizadorPose(str(MODELOS / "pose_landmarker_full.task")) if args.pose else None


    escenas, lanzador, vigilantes = [], None, []
    if args.escenas:
        ruta_escenas = Path(args.escenas)
        if not ruta_escenas.is_absolute():
            ruta_escenas = RAIZ / ruta_escenas
        escenas = cargar_escenas(ruta_escenas)
        lanzador = Lanzador(seco=args.escenas_seco)
        for e in escenas:
            if e.disparador == "palmada":
                vigilantes.append((e, DetectorPalmada(enfriamiento=e.enfriamiento)))
            else:
                vigilantes.append((e, DetectorGestoSostenido(
                    e.disparador, segundos=e.sostener, enfriamiento=e.enfriamiento)))
        print(f"→ escenas activas ({ruta_escenas.name}):")
        for e in escenas:
            print(f"   {e.nombre}  ←  {e.disparador}")

    ver = {"malla": True, "objetos": True, "manos": True, "pose": args.pose}
    pintor = Hud(args.width, args.height)
    estado_hud = EstadoHud()
    fps_hist = deque(maxlen=30)
    grabador = None
    aviso, aviso_hasta = "", 0.0
    ms_cara = 0.0
    t_inicio = time.time()
    n = 0
    vistos = {"cara": 0, "manos": 0, "objetos": 0}

    sin_ventana = bool(args.benchmark)
    if not sin_ventana:
        cv2.namedWindow("vision-live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("vision-live", args.width, args.height)

    try:
        while fuente.viva:
            frame, _ = fuente.leer()
            if frame is None:
                break
            n += 1
            espejo = not args.no_mirror
            t0 = time.perf_counter()

            # Los modelos trabajan sobre una copia reducida: devuelven coordenadas
            # normalizadas (0-1), así que el dibujo a resolución completa encaja
            # igual y se ganan bastantes fps en vertical 1080x1920.
            if args.analisis and max(frame.shape[:2]) > args.analisis:
                escala = args.analisis / max(frame.shape[:2])
                pequeno = cv2.resize(frame, None, fx=escala, fy=escala,
                                     interpolation=cv2.INTER_AREA)
            else:
                pequeno = frame
            rgb = cv2.cvtColor(pequeno, cv2.COLOR_BGR2RGB)
            ts = int((time.time() - t_inicio) * 1000)
            est_cara = cara.procesar(rgb, ts)
            est_manos = manos.procesar(rgb, ts)
            est_pose = poses.procesar(rgb, ts) if poses else None
            ms_cara = ms_cara * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1
            # Los modelos ven la imagen tal cual llega (con la imagen espejada el
            # detector de manos falla); el espejo se aplica sólo a lo que se pinta.
            if espejo:
                _espejar_cara(est_cara)
                _espejar_manos(est_manos)
                if est_pose:
                    _espejar_pose(est_pose)

            if detector and n % args.yolo_cada == 0:
                detector.enviar(frame)
            if reconocedor and n % args.id_cada == 0:
                reconocedor.enviar(frame)

            lienzo = cv2.flip(frame, 1) if espejo else frame
            if est_cara.detectada and ver["malla"]:
                pintor.cara(lienzo, est_cara, args.detalle_malla)
            if ver["manos"] and est_manos.manos:
                pintor.manos(lienzo, est_manos.manos)
            if est_pose and est_pose.detectada and ver["pose"]:
                pintor.pose(lienzo, est_pose.puntos)

            estado_obj = detector.estado if detector else None
            if estado_obj and espejo:
                estado_obj = _espejar_objetos(estado_obj, frame.shape[1])

            nombre_id, similitud_id = ("tú" if not reconocedor else "desconocido"), 0.0
            if reconocedor:
                ids = reconocedor.resultado
                if ids and ids[0].nombre != "desconocido":
                    nombre_id, similitud_id = ids[0].nombre, ids[0].similitud

            vistos["cara"] += est_cara.detectada
            vistos["manos"] += bool(est_manos.manos)
            vistos["objetos"] += bool(estado_obj and estado_obj.objetos)

            gestos = list(est_cara.gestos) + est_manos.gestos + (
                est_pose.acciones if est_pose else [])

            for escena, vigilante in vigilantes:
                disparo = (vigilante.actualizar(est_manos)
                           if isinstance(vigilante, DetectorPalmada)
                           else vigilante.actualizar(gestos))
                if disparo:
                    print(f"\n⚡ {escena.nombre}", flush=True)
                    lanzador.lanzar(escena)
                    aviso, aviso_hasta = f"⚡ {escena.nombre}", time.time() + 3

            estado_hud.fps = float(np.mean(fps_hist)) if fps_hist else 0.0
            estado_hud.cara = est_cara
            estado_hud.identidad = nombre_id
            estado_hud.similitud = similitud_id
            estado_hud.gestos = gestos
            estado_hud.objetos = estado_obj.resumen if (estado_obj and ver["objetos"]) else {}
            estado_hud.escenas = [
                (e.nombre, e.disparador,
                 1.0 if getattr(v, "armado", False) else getattr(v, "progreso", 0.0))
                for e, v in vigilantes
            ]
            if fuente.estado == "reconectando":
                estado_hud.aviso = "reconectando con la cámara…"
            else:
                estado_hud.aviso = aviso if time.time() < aviso_hasta else ""
            estado_hud.grabando = grabador is not None

            lienzo = pintor.dibujar(
                lienzo, estado_hud,
                estado_obj.objetos if (estado_obj and ver["objetos"]) else None)

            if grabador is not None:
                grabador.write(lienzo)
            fps_hist.append(1.0 / max(time.perf_counter() - t0, 1e-6))

            if sin_ventana:
                if n == 1:
                    print(f"   {lienzo.shape[1]}x{lienzo.shape[0]}")
                if n >= args.benchmark:
                    ruta = CAPTURAS / "benchmark.png"
                    cv2.imwrite(str(ruta), lienzo)
                    print(f"\n{n} frames · {np.mean(fps_hist):.1f} FPS · "
                          f"cara+manos {ms_cara:.0f} ms · "
                          f"yolo {estado_obj.ms if estado_obj else 0:.0f} ms")
                    print(f"cara={est_cara.detectada} expresión={est_cara.expresion} "
                          f"mirada={est_cara.direccion_mirada} identidad={nombre_id}")
                    print(f"gestos={gestos}")
                    print(f"frames con cara {vistos['cara']}/{n}, "
                          f"con manos {vistos['manos']}/{n}, "
                          f"con objetos {vistos['objetos']}/{n}")
                    print(f"entorno={estado_obj.resumen if estado_obj else {}}")
                    print(f"reconexiones de la cámara: {fuente.reconexiones}")
                    print(f"último frame -> {ruta}")
                    break
                continue

            cv2.imshow("vision-live", lienzo)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("m"):
                ver["malla"] = not ver["malla"]
            elif k == ord("o"):
                ver["objetos"] = not ver["objetos"]
            elif k == ord("h"):
                ver["manos"] = not ver["manos"]
            elif k == ord("p"):
                if poses is None:
                    poses = AnalizadorPose(str(MODELOS / "pose_landmarker_full.task"))
                ver["pose"] = not ver["pose"]
            elif k == ord("s"):
                ruta = CAPTURAS / f"{time.strftime('%Y%m%d-%H%M%S')}.png"
                cv2.imwrite(str(ruta), lienzo)
                aviso, aviso_hasta = f"foto guardada: {ruta.name}", time.time() + 2
            elif k == ord("r"):
                if grabador is None:
                    ruta = CAPTURAS / f"{time.strftime('%Y%m%d-%H%M%S')}.mp4"
                    grabador = cv2.VideoWriter(
                        str(ruta), cv2.VideoWriter_fourcc(*"mp4v"),
                        20, (lienzo.shape[1], lienzo.shape[0]))
                    aviso, aviso_hasta = f"grabando en {ruta.name}", time.time() + 2
                else:
                    grabador.release()
                    grabador = None
                    aviso, aviso_hasta = "grabación guardada", time.time() + 2
            elif k == ord("e") and reconocedor:
                aviso, aviso_hasta = "escribe el nombre en la terminal…", time.time() + 3
                cv2.imshow("vision-live", lienzo)
                cv2.waitKey(1)
                nombre = input("\nNombre de la persona (enter = cancelar): ").strip()
                if nombre:
                    guardadas = _enrolar_en_vivo(fuente, reconocedor, nombre,
                                                 espejo, muestras=12)
                    aviso = (f"{nombre}: {guardadas} muestras guardadas" if guardadas
                             else "no se detectó ninguna cara")
                    aviso_hasta = time.time() + 3
                    t_inicio = time.time() - ts / 1000  # no romper la línea temporal
    finally:
        if grabador is not None:
            grabador.release()
        fuente.cerrar()
        cara.cerrar()
        manos.cerrar()
        if poses:
            poses.cerrar()
        if detector:
            detector.cerrar()
        if reconocedor:
            reconocedor.cerrar()
        cv2.destroyAllWindows()
    return 0


def _espejar_cara(est) -> None:
    if not est.detectada:
        return
    est.puntos[:, 0] = 1.0 - est.puntos[:, 0]
    x1, y1, x2, y2 = est.bbox
    est.bbox = (1.0 - x2, y1, 1.0 - x1, y2)
    est.yaw, est.roll = -est.yaw, -est.roll
    est.mirada = (-est.mirada[0], est.mirada[1])
    if "izquierda" in est.direccion_mirada:
        est.direccion_mirada = est.direccion_mirada.replace("izquierda", "derecha")
    elif "derecha" in est.direccion_mirada:
        est.direccion_mirada = est.direccion_mirada.replace("derecha", "izquierda")


def _espejar_manos(est) -> None:
    for mano in est.manos:
        mano.puntos[:, 0] = 1.0 - mano.puntos[:, 0]
        x1, y1, x2, y2 = mano.bbox
        mano.bbox = (1.0 - x2, y1, 1.0 - x1, y2)


def _espejar_pose(est) -> None:
    if est.detectada:
        est.puntos[:, 0] = 1.0 - est.puntos[:, 0]


def _espejar_objetos(estado, ancho: int):
    from .objects import EstadoObjetos, Objeto
    return EstadoObjetos(
        [Objeto(o.nombre, o.confianza,
                (ancho - o.bbox[2], o.bbox[1], ancho - o.bbox[0], o.bbox[3]))
         for o in estado.objetos],
        estado.resumen, estado.ms)


def _enrolar_en_vivo(fuente, reconocedor, nombre: str, espejo: bool,
                     muestras: int = 12) -> int:
    """Toma varias fotos seguidas para que el perfil aguante giros y luces."""
    print(f"Capturando {muestras} muestras de «{nombre}». Mueve un poco la cabeza…")
    vectores = []
    intentos = 0
    while len(vectores) < muestras and intentos < muestras * 6:
        intentos += 1
        frame, _ = fuente.leer()
        if frame is None:
            break
        vista = cv2.flip(frame, 1) if espejo else frame.copy()
        cv2.putText(vista, f"{nombre}: {len(vectores)}/{muestras}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (120, 255, 140), 3, cv2.LINE_AA)
        cv2.imshow("vision-live", vista)
        cv2.waitKey(60)
        try:
            caras = reconocedor.embeddings(frame)
        except Exception:
            caras = []
        if caras:
            vectores.append(caras[0][0])
    if vectores:
        reconocedor.base.guardar(nombre, np.array(vectores, dtype=np.float32))
        reconocedor.recargar()
        print(f"✓ Guardado. «{nombre}» tiene ahora "
              f"{len(reconocedor.base.gente[nombre])} muestras.")
    return len(vectores)


if __name__ == "__main__":
    raise SystemExit(main())
