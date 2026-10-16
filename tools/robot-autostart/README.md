# Robot autostart for the medcare_reachy app

**What.** A systemd timer on the robot that starts the `medcare_reachy` app through the Reachy Mini daemon's app
manager whenever it is not running: 25 s after boot, then every 2 minutes.

**Why.** The daemon (1.8.3) has no startup-app setting (`GET /api/apps/startup-app` is 404): after a reboot nothing
starts the app, and a restart of `reachy-mini-daemon` ends the app with it. Verified on 14 Oct 2026. An app is started
only by `POST http://127.0.0.1:8000/api/apps/start-app/medcare_reachy`; its state comes from
`GET /api/apps/current-app-status` (`{"info":{"name":...},"state":"running"|...}`).

Files:

- `medcare-reachy-app.sh`: waits for the daemon (up to 120 s), exits quietly when the app runs, otherwise starts it
  and waits up to 60 s for `"state":"running"`. Logs with `logger -t medcare-autostart`. Never stops or restarts the
  daemon.
- `medcare-reachy-app.service` (oneshot, as `pollen`, after `reachy-mini-daemon.service`) runs the script.
- `medcare-reachy-app.timer`: `OnBootSec=25s`, `OnUnitActiveSec=2min`.
- `install.sh`: copies the script to `/home/pollen/.medcare_reachy/bin/` and the units to `/etc/systemd/system/`,
  then `systemctl daemon-reload` and `systemctl enable --now medcare-reachy-app.timer`. Idempotent.

## Install

From the laptop, copy the directory to the robot and run the installer there (it asks for sudo itself):

```sh
scp -r tools/robot-autostart pollen@reachy-mini.local:~/robot-autostart
ssh pollen@reachy-mini.local sh ~/robot-autostart/install.sh
```

The installer ends with `systemctl list-timers medcare-reachy-app.timer`, which shows the next run.

## Check

```sh
systemctl list-timers medcare-reachy-app.timer      # next and last run
journalctl -t medcare-autostart -n 20                # what the script did (nothing is logged while the app runs)
systemctl status medcare-reachy-app.service          # the last run's exit code
curl -s localhost:8000/api/apps/current-app-status   # the daemon's view of the app
```

A start by hand still works: `curl -s -X POST localhost:8000/api/apps/start-app/medcare_reachy`. To stop the app
for a while, stop the timer first (`sudo systemctl stop medcare-reachy-app.timer`), or it comes back within 2 minutes.

## Uninstall

```sh
ssh pollen@reachy-mini.local sh ~/robot-autostart/install.sh --uninstall
```

This disables the timer and removes the three installed files; a running app is not touched.
