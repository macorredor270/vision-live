#!/usr/bin/env bash
# Convierte el iPhone conectado por USB-C en una webcam del sistema (/dev/video9),
# usando la app "IP Camera Lite". Menos retardo que por WiFi y no depende de la red.
#
#   ./scripts/iphone_usb.sh          # monta la cámara y la deja funcionando
#   ./scripts/iphone_usb.sh --parar  # descarga todo
#
# Después:  ./run --source 9
set -euo pipefail

DISPOSITIVO=9
PUERTO=8081
USUARIO=admin:admin

if [[ "${1:-}" == "--parar" ]]; then
    pkill -f "ffmpeg.*video$DISPOSITIVO" 2>/dev/null || true
    pkill -f "iproxy $PUERTO" 2>/dev/null || true
    sudo modprobe -r v4l2loopback 2>/dev/null || true
    echo "Desmontado."
    exit 0
fi

faltan=()
for p in v4l2loopback-dkms libimobiledevice usbmuxd ffmpeg; do
    pacman -Q "$p" &>/dev/null || faltan+=("$p")
done
if (( ${#faltan[@]} )); then
    echo "Instalando: ${faltan[*]}"
    sudo pacman -S --needed "${faltan[@]}"
fi

echo "→ Arrancando usbmuxd"
sudo systemctl start usbmuxd
# Si el iPhone ya estaba enchufado antes de instalar usbmuxd, su nodo USB se
# quedó sin los permisos de las reglas udev: hay que reaplicarlas.
if ! idevice_id -l | grep -q .; then
    sudo udevadm control --reload-rules
    sudo udevadm trigger --subsystem-match=usb --action=add
    sleep 2
    sudo systemctl restart usbmuxd
    sleep 3
fi
if ! idevice_id -l | grep -q .; then
    echo "No veo el iPhone. Desconecta y vuelve a conectar el cable." >&2
    exit 1
fi

echo "→ Empareja el iPhone (desbloquéalo y pulsa «Confiar» si lo pide)"
idevicepair pair

echo "→ Creando /dev/video$DISPOSITIVO"
sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback video_nr=$DISPOSITIVO card_label="iPhone" exclusive_caps=1

echo "→ Túnel USB del puerto $PUERTO"
pkill -f "iproxy $PUERTO" 2>/dev/null || true
iproxy "$PUERTO" "$PUERTO" &
sleep 2

echo "→ Abre «IP Camera Lite» en el iPhone y pulsa «Turn on IP Camera Server»"
read -r -p "   Cuando esté emitiendo, pulsa Enter… "

# La ruta del MJPEG cambia según la versión de la app: se prueban las dos.
RUTA=""
for r in video live; do
    if curl -s -m 5 -o /dev/null "http://$USUARIO@127.0.0.1:$PUERTO/$r"; then
        RUTA="$r"; break
    fi
done
if [[ -z "$RUTA" ]]; then
    echo "Nada emite en el puerto $PUERTO del iPhone." >&2
    echo "Comprueba que la app está instalada y con el servidor encendido." >&2
    exit 1
fi
echo "   stream encontrado en /$RUTA"

ffmpeg -loglevel warning -f mjpeg -i "http://$USUARIO@127.0.0.1:$PUERTO/$RUTA" \
       -pix_fmt yuv420p -f v4l2 "/dev/video$DISPOSITIVO" &
sleep 3
echo
echo "Listo. La cámara del iPhone es /dev/video$DISPOSITIVO:"
echo "    ./run --source $DISPOSITIVO"
echo "Para desmontarlo:  ./scripts/iphone_usb.sh --parar"
wait
