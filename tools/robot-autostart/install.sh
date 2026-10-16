#!/bin/sh
# Installs (or with --uninstall removes) the medcare-reachy-app autostart on the robot. Run it on the robot as
# pollen; it asks for sudo itself:
#   sh ~/robot-autostart/install.sh
# Idempotent: run it again after changing any of the files. It never stops or restarts reachy-mini-daemon.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=/home/pollen/.medcare_reachy/bin
UNITS=/etc/systemd/system
TIMER=medcare-reachy-app.timer

if [ "$(id -u)" -ne 0 ]; then
    exec sudo sh "$0" "$@"
fi

if [ "$1" = "--uninstall" ]; then
    systemctl disable --now "$TIMER" 2>/dev/null || true
    rm -f "$UNITS/$TIMER" "$UNITS/medcare-reachy-app.service" "$BIN/medcare-reachy-app.sh"
    systemctl daemon-reload
    echo "medcare-reachy-app autostart removed (the running app is not touched)"
    exit 0
fi

mkdir -p "$BIN"
chown pollen:pollen /home/pollen/.medcare_reachy "$BIN"
install -m 755 -o pollen -g pollen "$HERE/medcare-reachy-app.sh" "$BIN/medcare-reachy-app.sh"
install -m 644 "$HERE/medcare-reachy-app.service" "$UNITS/medcare-reachy-app.service"
install -m 644 "$HERE/$TIMER" "$UNITS/$TIMER"
systemctl daemon-reload
systemctl enable --now "$TIMER"
systemctl list-timers "$TIMER"
