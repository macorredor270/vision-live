<p align="center">
  <img src="docs/banner.svg" alt="vision-live: cara, ojos, manos, postura y objetos en tiempo real, 100% local." width="100%">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-00d4ff">
  <img alt="Local" src="https://img.shields.io/badge/nube-0%25-00ffb2">
  <img alt="GPU" src="https://img.shields.io/badge/YOLO11-CUDA-00ffb2">
  <img alt="FPS" src="https://img.shields.io/badge/1280%C3%97720-~25%20fps-00d4ff">
</p>

<p align="center">
  <a href="#verificar-que-funciona-sin-cámara-ni-gui">Verificar que funciona</a> ·
  <a href="#uso">Uso</a> ·
  <a href="#usar-el-iphone-como-cámara">iPhone como cámara</a> ·
  <a href="#modo-stark-un-gesto-abre-tus-apps">Modo Stark</a> ·
  <a href="#que-te-reconozca-por-tu-nombre">Reconocimiento</a> ·
  <a href="#rendimiento-medido">Rendimiento</a> ·
  <a href="#instalación-desde-cero">Instalación</a>
</p>

---

Percepción visual en tiempo real, **100 % en tu PC**: sin APIs, sin cuentas, sin
límites de uso y sin que ninguna imagen salga del ordenador.

Reconoce a la vez:

| Qué | Cómo |
|---|---|
| **Tu cara** (quién eres, por nombre) | InsightFace / ArcFace, base local en `data/faces/` |
| **478 puntos faciales** + 52 expresiones | MediaPipe Face Landmarker |
| **Ojos**: iris, hacia dónde miras, parpadeos, guiños | iris + EAR geométrico |
| **Orientación de la cabeza** (giro, alzado, ladeo) | matriz de transformación facial |
| **Emoción aproximada** | combinación de unidades de acción facial |
| **Gestos de mano** | MediaPipe Gesture Recognizer + geometría propia (OK, cuernos, pellizco, dedos extendidos…) |
| **Postura del cuerpo** (33 puntos) | MediaPipe Pose (`--pose`) |
| **Objetos del entorno** (80 clases en español) | YOLO11 sobre CUDA |

## Verificar que funciona (sin cámara ni GUI)

No hace falta webcam, iPhone ni pantalla para comprobar que la instalación
funciona de verdad: el repo trae vídeos de prueba en `data/test/` y `--benchmark`
procesa frames sin abrir ninguna ventana — es la forma en que se verificó este
mismo README, ejecutándolo tal cual en una máquina sin cámara conectada:

```bash
./run --source data/test/demo.mp4 --benchmark 30 --pose
```

Salida real de esa ejecución:

```
→ abriendo fuente de vídeo…
   data/test/demo.mp4
→ cargando YOLO…
   yolo11n.pt en cuda:0
→ cargando reconocimiento facial…
   0 persona(s) en la base: —
   1280x720

30 frames · 16.6 FPS · cara+manos 36 ms · yolo 25 ms
cara=True expresión=feliz mirada=al frente identidad=desconocido
gestos=['sonriendo']
frames con cara 30/30, con manos 30/30, con objetos 27/30
entorno={'persona': 1, 'corbata': 1}
reconexiones de la cámara: 0
último frame -> data/capturas/benchmark.png
```

Cara, manos y objetos detectados en el 100 %, 90 % y 90 % de los frames
respectivamente, sobre GPU (`cuda:0`) — instalación y modelos correctos de
punta a punta. Si esto corre, `./run` con una cámara real también correrá.

## Uso

```bash
./run                              # primera webcam que encuentre
./run --source 192.168.1.40        # la cámara del iPhone por WiFi (ver abajo)
./run --probe 192.168.1.40         # ver qué URL de cámara responde en esa IP
./run --source data/test/demo.mp4  # un vídeo
./run --pose                       # añadir el esqueleto corporal
./run --list-cams                  # ver qué cámaras hay
./run --benchmark 100              # medir rendimiento sin abrir ventana
```

Teclas: `q` salir · `e` enrolar tu cara · `m` malla · `o` objetos · `h` manos ·
`p` pose · `s` foto · `r` grabar vídeo.

<details>
<summary>Todos los flags (<code>./run --help</code>)</summary>

| Flag | Qué hace |
|---|---|
| `--source` | `'auto'`, índice de webcam (`0`), IP del móvil, URL http/rtsp o ruta de un vídeo |
| `--width` / `--height` | resolución de captura |
| `--no-mirror` | no voltear la imagen como un espejo |
| `--rotar {0,90,180,270}` | girar la imagen (el móvil suele llegar tumbado) |
| `--no-objects` | desactivar YOLO |
| `--no-identity` | desactivar el reconocimiento de quién eres |
| `--pose` | activar el esqueleto corporal |
| `--faces N` | cuántas caras seguir a la vez |
| `--yolo yolo11{n,s,m,l,x}.pt` | tamaño del modelo (`n` = más rápido) |
| `--yolo-conf` | confianza mínima de YOLO |
| `--yolo-cada N` | lanzar YOLO cada N frames |
| `--id-cada N` | reidentificar la cara cada N frames |
| `--umbral-id` | parecido mínimo para dar un nombre |
| `--detalle-malla {normal,completo}` | densidad de la malla facial dibujada |
| `--analisis PX` | lado mayor al que se reduce la imagen para analizarla (0 = sin reducir) |
| `--escenas [ARCHIVO]` | activar los gestos que lanzan apps |
| `--escenas-seco` | con `--escenas`: enseña qué lanzaría, sin lanzarlo |
| `--list-cams` | listar webcams y salir |
| `--probe IP` | probar las URLs típicas de cámara IP en esa IP y salir |
| `--benchmark N` | procesar N frames sin abrir ventana y medir el rendimiento |

</details>

## Usar el iPhone como cámara

No hay ninguna webcam conectada al PC, así que la cámara es el iPhone.

### Por WiFi (lo más rápido de montar)

1. Instala **IP Camera Lite** en el iPhone (gratis en la App Store).
2. Ábrela y pulsa **«Turn on IP Camera Server»**. Arriba te muestra una
   dirección tipo `http://192.168.1.40:8081`.
3. Con el iPhone en el mismo WiFi que el PC:

```bash
./run --source 192.168.1.40 --rotar 270
```

El iPhone emite en la orientación del sensor, así que llega tumbado: `--rotar 270`
lo endereza cuando lo tienes en vertical (`90` si lo giras al otro lado). La
imagen se endereza antes de analizarla, para que los modelos vean la cara derecha.

Basta con la IP: la app prueba sola las URLs conocidas (IP Camera Lite, IP
Webcam de Android, DroidCam…) y usa la que responda. Para ver cuál funciona sin
arrancar nada:

```bash
./run --probe 192.168.1.40
```

Si cambiaste el usuario o la contraseña en la app, pasa la URL entera:
`--source 'http://usuario:clave@192.168.1.40:8081/video'`.

### Por cable USB-C (menos retardo, sin depender del WiFi)

```bash
./scripts/iphone_usb.sh        # instala lo que falte, monta /dev/video9
./run --source 9
./scripts/iphone_usb.sh --parar
```

El script usa `usbmuxd` + `iproxy` para tunelar el puerto de la app por el cable
y `v4l2loopback` para que el iPhone aparezca como una webcam normal del sistema
— también la verán OBS, Zoom o el navegador. Pide sudo para instalar paquetes y
cargar el módulo del kernel.

## Modo Stark: un gesto abre tus apps

```bash
./run --source 172.20.10.1 --rotar 270 --escenas
```

Con `--escenas` la app vigila gestos de disparo y lanza lo que pongas en
`escenas.toml`. De fábrica vienen tres:

| Gesto | Qué hace |
|---|---|
| **Palmada** (juntar las dos manos de golpe) | Notificación + voz «Bienvenido de vuelta, señor», Spotify, Claude Code en kitty y Firefox, todo en el escritorio 7 de Hyprland |
| **Puño** sostenido 1,5 s | Modo foco: pausa la música y salta al escritorio 2 |
| **Palma abierta** sostenida 1 s | Play / pausa de la música |

Editar `escenas.toml` es todo lo que hace falta para cambiarlo: cada escena tiene
un `disparador` (`palmada` o cualquier gesto que la app reconozca) y una lista de
`pasos` con comandos. Para probar sin que se abra nada:

```bash
./run --source 172.20.10.1 --rotar 270 --escenas --escenas-seco
```

La palmada exige que las dos manos se acerquen **rápido**: acercarlas despacio no
dispara, y hay un tiempo de enfriamiento para que una palmada no cuente dos veces.

## Que te reconozca por tu nombre

Con la app abierta pulsa **`e`**, escribe tu nombre en la terminal y mueve
despacio la cabeza: guarda 12 muestras para aguantar giros y cambios de luz.

Desde fotos que ya tengas:

```bash
./.venv/bin/python scripts/enrolar_fotos.py m1ke ~/Pictures/yo/*.jpg
./.venv/bin/python scripts/gente.py              # ver quién está guardado
./.venv/bin/python scripts/gente.py borrar m1ke  # borrar a alguien
```

Los perfiles son vectores en `data/faces/*.npy`. No se guarda ninguna foto y
nada se sube a ningún sitio.

## Robustez

- **La cámara se cae y la app sigue viva**: si el iPhone corta el vídeo (lo hace
  en cuanto se va el último cliente), la fuente se reabre sola, volviendo a
  buscar la URL que funcione. Probado matando el emisor a mitad de sesión:
  200/200 frames procesados, 27 FPS, sin caerse.
- **Espera a que empieces a emitir**: al arrancar con una IP, busca la cámara
  durante 45 s en vez de fallar al instante.
- **Los modelos analizan una copia reducida** (`--analisis 640`) y el HUD se
  dibuja a resolución completa: mismas detecciones, bastantes fps más en vertical.
- El HUD se escala solo al tamaño de la imagen, así que se ve igual de bien en
  720p apaisado que en 1080×1920 vertical.

## Rendimiento medido

Con `--benchmark N` mide sin abrir ventana y dice cuántos frames detectaron cada cosa.

**RTX 3050, 1280×720, todo activado:**

| Escenario | FPS | Detección |
|---|---|---|
| 1280×720, todo activado | ~25 | cara 30/30, manos 30/30, objetos 27/30 |
| 1080×1920 vertical (el iPhone) | ~20 | cara 30/30, manos 30/30, objetos 27/30 |

Reparto: cara + manos ~26 ms (CPU), YOLO11n ~16 ms (CUDA), HUD ~10 ms,
identidad ~90 ms en su propio hilo cada 15 frames.

Para ir más rápido: `--yolo-cada 5`, `--no-objects` o bajar a `--width 960`.
Para afinar más: `--yolo yolo11s.pt` (o `m`, `l`, `x`), que se descargan solos.

## Estructura

```
vision/sources.py    captura en hilo aparte (webcam / IP / archivo)
vision/face.py       cara, ojos, mirada, expresión, orientación
vision/hands.py      manos y gestos
vision/pose.py       postura corporal
vision/objects.py    YOLO en hilo aparte
vision/identity.py   embeddings y base local de caras
vision/hud.py        dibujo del HUD
vision/app.py        bucle principal y CLI
models/               pesos: face/hand/gesture/pose landmarker + yolo11n.pt
data/test/            vídeos y fotos de prueba (para benchmark sin cámara)
scripts/              enrolar_fotos.py · gente.py · iphone_usb.sh
```

## Instalación desde cero

Requiere **Python 3.12** exacto (mediapipe 0.10.21 no soporta 3.13+ todavía)
y, para YOLO por GPU, un driver CUDA. Con [`uv`](https://github.com/astral-sh/uv)
(recomendado — así es como está montado este repo):

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
./run --source data/test/demo.mp4 --benchmark 30   # ver arriba: así se verifica sin cámara
```

Los modelos de MediaPipe (`models/*.task`) y `yolo11n.pt` ya vienen incluidos en
el repo (`models/`, unos 34 MB en total) — no hacen falta descargas aparte para
arrancar. Solo se descargan solos los tamaños de YOLO más grandes si pides
`--yolo yolo11s.pt` (o `m`/`l`/`x`).

Sin `uv`, con `venv` + `pip` normal funciona igual siempre que el intérprete sea
3.12:

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Notas de instalación

Python **3.12** (en el venv, vía `uv`): mediapipe todavía no funciona con 3.14.
La versión de mediapipe está fijada a `0.10.21` — la 1.0.1 aborta el proceso al
crear el detector facial.

`onnxruntime-gpu` corre en CPU porque el paquete espera CUDA 12 y aquí hay
CUDA 13 (la que usa torch). No merece la pena romper torch por ello: el
reconocimiento de identidad no necesita GPU tal como está montado.
