#!/bin/sh
# Starts the medcare_reachy app through the Reachy Mini daemon's app manager when it is not running.
#
# Daemon 1.8.3 has no startup-app setting (GET /api/apps/startup-app is 404), so after a reboot, and after every
# restart of reachy-mini-daemon (which ends the app with it), nothing starts the app. medcare-reachy-app.timer runs
# this 25 s after boot and every 2 min; it exits 0 quietly when the app already runs. It never stops or restarts the
# daemon.
#
# Lines go to the journal (journalctl -t medcare-autostart) and to stdout. Exit 1 when the daemon or the app did not
# come up in time; the timer tries again.

DAEMON="http://127.0.0.1:8000"
APP="medcare_reachy"
DAEMON_WAIT=120   # seconds to wait for the daemon to answer after boot
START_WAIT=60     # seconds to wait for the app to reach "running" once started
STEP=3

say() {
    logger -t medcare-autostart "$1"
    echo "medcare-autostart: $1"
}

# The daemon's current-app-status ({"info":{"name":...},"state":"running"|...}), or nothing when it does not answer.
app_status() {
    curl -s -m 5 "$DAEMON/api/apps/current-app-status" 2>/dev/null
}

state_is() {
    app_status | grep -qE "\"state\": *\"$1\""
}

waited=0
until curl -s -m 5 -o /dev/null "$DAEMON/api/daemon/status"; do
    if [ "$waited" -ge "$DAEMON_WAIT" ]; then
        say "the daemon at $DAEMON did not answer within $DAEMON_WAIT s; giving up until the next run"
        exit 1
    fi
    sleep "$STEP"
    waited=$((waited + STEP))
done

if state_is running; then
    exit 0   # nothing to do, and nothing in the journal every 2 min
fi

if state_is starting; then
    say "the app is starting (current-app-status: $(app_status)); waiting for it"
else
    status=$(app_status)
    say "the app is not running (current-app-status: ${status:-no answer}); starting $APP"
    answer=$(curl -s -m 30 -X POST "$DAEMON/api/apps/start-app/$APP")
    say "start-app answered: ${answer:-nothing}"
fi

waited=0
until state_is running; do
    if [ "$waited" -ge "$START_WAIT" ]; then
        say "$APP did not reach running within $START_WAIT s (current-app-status: $(app_status))"
        exit 1
    fi
    sleep "$STEP"
    waited=$((waited + STEP))
done
say "$APP is running"
exit 0
