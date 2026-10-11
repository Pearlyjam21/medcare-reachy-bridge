"""CameraWatchdog (spec 2026-10-11 A2, A3): stall, recovery, give-up and the heartbeat snapshot, on a fake clock.

On 11 Oct 2026 the robot's camera reader gave one stale frame and then nothing, and the app searched for 10 minutes
without a log line: the TCU companion had released the daemon's camera.
"""

from medcare_reachy.bridge.camera_watch import (
    CAMERA_GIVE_UP_SECONDS, CAMERA_HEALTHY_SECONDS, CAMERA_RECOVER_INTERVAL, CAMERA_STALL_SECONDS, CameraWatchdog)
from medcare_reachy.bridge.tests.fakes import FakeClock

LOGGER = "medcare_reachy.bridge.camera_watch"
RELEASED = {"available": False, "released": True, "no_media": False}
WALL = 1_760_000_000.0
STEP = 0.25   # a read every 0.25 s: exact in binary, so the timings below are exact too


class Recovery:
    """reopen_camera() stand-in: the clock time of each call; answers from `results` in turn (the last one repeats;
    an exception is raised). `during` runs inside the call, as another thread would while a recovery blocks."""

    def __init__(self, clock, *results, during=None):
        self.clock, self.results, self.during = clock, list(results) or ["reopened"], during
        self.calls = []

    def __call__(self):
        result = self.results[min(len(self.calls), len(self.results) - 1)]
        self.calls.append(self.clock.t)
        if self.during is not None:
            self.during()
        if isinstance(result, Exception):
            raise result
        return result


def make_watchdog(*results, media=RELEASED, during=None, **limits):
    clock = FakeClock()
    recovery = Recovery(clock, *results, during=during)

    def media_status():
        if isinstance(media, Exception):
            raise media
        return media

    watchdog = CameraWatchdog(recovery, media_status, clock=clock.monotonic, wall=lambda: WALL + clock.t, **limits)
    return watchdog, clock, recovery


def reads(watchdog, clock, seconds):
    """Reads that find no new frame, one every STEP s, for `seconds`; the states tick() returned."""
    states = []
    for _ in range(round(seconds / STEP)):
        clock.t += STEP
        states.append(watchdog.tick())
    return states


def frames(watchdog, clock, seconds):
    """New frames, one every STEP s, for `seconds`."""
    for _ in range(round(seconds / STEP)):
        clock.t += STEP
        watchdog.frame()


def when_unavailable(watchdog, clock, seconds):
    """Reads that find no new frame, one every STEP s: the clock time of the first "unavailable", else None."""
    for _ in range(round(seconds / STEP)):
        clock.t += STEP
        if watchdog.tick() == "unavailable":
            return clock.t
    return None


def lines(caplog, level):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelname == level]


def test_the_limits_are_the_spec_s():
    assert (CAMERA_STALL_SECONDS, CAMERA_RECOVER_INTERVAL, CAMERA_GIVE_UP_SECONDS) == (2.0, 5.0, 15.0)
    assert CAMERA_HEALTHY_SECONDS == 5.0      # steady frames this long end a stall (review finding F1)


def test_idle_without_a_session_never_tries_a_recovery():
    watchdog, clock, recovery = make_watchdog()
    assert watchdog.state == "idle"
    assert set(reads(watchdog, clock, 30)) == {"idle"}
    assert recovery.calls == []
    assert watchdog.snapshot() == {"state": "idle", "fps": 0.0, "since": WALL + 1000.0, "media": None}


def test_a_steady_camera_stays_ok_and_reports_its_fps():
    watchdog, clock, recovery = make_watchdog()
    watchdog.attach()
    for _ in range(120):        # 12 s of a 10 fps camera, with reads that find nothing in between
        clock.t += 0.05
        assert watchdog.tick() == "ok"
        clock.t += 0.05
        watchdog.frame()
    snapshot = watchdog.snapshot()
    assert snapshot["state"] == "ok" and 9.9 <= snapshot["fps"] <= 10.1
    assert snapshot["since"] == WALL + 1000.0 and snapshot["media"] is None   # ok since the session started
    assert recovery.calls == []


def test_a_slow_camera_is_not_a_stall():
    """Review focus 2: a camera that delivers 5 fps, or one 1.75 s gap, must not count as stalled."""
    watchdog, clock, recovery = make_watchdog()
    watchdog.attach()
    for _ in range(50):         # 10 s at 5 fps, with a read that finds nothing between two frames
        clock.t += 0.1
        assert watchdog.tick() == "ok"
        clock.t += 0.1
        watchdog.frame()
    assert set(reads(watchdog, clock, 1.75)) == {"ok"}
    watchdog.frame()
    assert recovery.calls == []


def test_two_seconds_without_a_frame_is_one_stall_logged_with_the_daemon_s_media_status(caplog):
    watchdog, clock, recovery = make_watchdog()
    watchdog.attach()
    watchdog.frame()
    with caplog.at_level("INFO", logger=LOGGER):
        states = reads(watchdog, clock, 1.75)
        assert set(states) == {"ok"} and recovery.calls == []
        assert reads(watchdog, clock, STEP) == ["recovering"]    # 2.0 s: stalled, and a recovery at once
        reads(watchdog, clock, 4)
    assert recovery.calls == [1002.0]
    [error] = lines(caplog, "ERROR")
    assert "no new frame for 2.0 s" in error and "'released': True" in error
    assert watchdog.snapshot()["media"] == RELEASED


def test_recovery_is_tried_at_most_once_per_interval_until_it_gives_up(caplog):
    watchdog, clock, recovery = make_watchdog()
    watchdog.attach()
    with caplog.at_level("INFO", logger=LOGGER):
        states = reads(watchdog, clock, 30)
    assert recovery.calls == [1002.0, 1007.0, 1012.0]             # the stall began at 1002: given up at 1017
    given_up = states.index("unavailable")                       # states[i] is the read at 1000 + (i + 1) * STEP
    assert given_up == round(17 / STEP) - 1 and states[given_up - 1] == "recovering"
    assert len([line for line in lines(caplog, "ERROR") if line.startswith("camera unavailable")]) == 1
    assert len([line for line in lines(caplog, "ERROR") if line.startswith("camera stalled")]) == 1


def test_a_failed_attempt_leaves_the_camera_stalled_and_a_good_one_recovering(caplog):
    watchdog, clock, recovery = make_watchdog("failed: the daemon did not give the media back", "acquired")
    watchdog.attach()
    with caplog.at_level("INFO", logger=LOGGER):
        assert reads(watchdog, clock, 2)[-1] == "stalled"
        assert watchdog.snapshot()["state"] == "stalled"
        assert reads(watchdog, clock, 5)[-1] == "recovering"
    warnings = lines(caplog, "WARNING")
    assert "camera recovery attempt 1 failed: failed: the daemon did not give the media back" in warnings
    assert "camera recovery attempt 2: acquired; waiting for a new frame" in lines(caplog, "INFO")


def test_a_recovery_or_a_status_read_that_raises_is_logged_and_counted_and_never_raises(caplog):
    watchdog, clock, recovery = make_watchdog(RuntimeError("gstreamer died"), media=OSError("daemon gone"))
    watchdog.attach()
    with caplog.at_level("INFO", logger=LOGGER):
        states = reads(watchdog, clock, 18)
    assert states[round(2 / STEP) - 1] == "stalled" and states[-1] == "unavailable"
    assert len(recovery.calls) == 3
    assert any("camera recovery raised" in r.getMessage() and r.exc_info for r in caplog.records)
    assert any("could not read the daemon's media status" in line for line in lines(caplog, "WARNING"))
    assert "3 recovery attempt(s), 3 failed" in [line for line in lines(caplog, "ERROR")
                                                  if line.startswith("camera unavailable")][0]
    assert watchdog.snapshot()["media"] is None


def test_a_new_frame_after_a_stall_shows_ok_but_the_stall_ends_only_after_5_s_of_steady_frames(caplog):
    watchdog, clock, recovery = make_watchdog()
    watchdog.attach()
    with caplog.at_level("INFO", logger=LOGGER):
        reads(watchdog, clock, 3.5)
        watchdog.frame()                                          # 1003.5: the feed is back
        assert watchdog.state == "ok" and watchdog.snapshot()["media"] is None
        frames(watchdog, clock, 4.75)                             # steady for 4.75 s: not yet 5 s
        assert not [line for line in lines(caplog, "INFO") if line.startswith("camera recovered")]
        frames(watchdog, clock, STEP)                             # steady for 5.0 s: the stall is over
        [recovered] = [line for line in lines(caplog, "INFO") if line.startswith("camera recovered")]
        assert recovered == ("camera recovered: new frames for 5.0 s without a gap; the stall began 6.5 s ago "
                             "(1 recovery attempt(s), 0 failed)")
        reads(watchdog, clock, 2)                                 # a second stall: logged and recovered at once
    assert len([line for line in lines(caplog, "ERROR") if line.startswith("camera stalled")]) == 2
    assert recovery.calls == [1002.0, 1010.5]


def test_a_flapping_feed_is_recovered_at_most_every_5_s_and_still_ends_unavailable(caplog):
    """One frame after each recovery, then nothing (what the camera did on 11 Oct): the frame must not buy another
    recovery 2 s later, nor stretch the 15 s that count from the first stall."""
    watchdog, clock, recovery = make_watchdog()
    watchdog.attach()
    answered, given_up = 0, None
    with caplog.at_level("INFO", logger=LOGGER):
        for _ in range(round(60 / STEP)):
            clock.t += STEP
            if len(recovery.calls) > answered:                    # the recovery got one frame, then nothing again
                answered = len(recovery.calls)
                watchdog.frame()
            else:
                watchdog.tick()
            if given_up is None and watchdog.state == "unavailable":
                given_up = clock.t
    assert recovery.calls == [1002.0, 1007.0, 1012.0]
    assert all(later - earlier >= CAMERA_RECOVER_INTERVAL for earlier, later in zip(recovery.calls, recovery.calls[1:]))
    assert given_up == 1017.0                                     # 15 s after the first stall, at 1002.0
    assert watchdog.state == "unavailable"
    errors = lines(caplog, "ERROR")
    assert len([line for line in errors if line.startswith("camera stalled")]) == 1     # one episode
    assert len([line for line in errors if line.startswith("camera unavailable")]) == 1
    assert "3 recovery attempt(s), 0 failed" in [line for line in errors if line.startswith("camera unavailable")][0]
    assert len([line for line in lines(caplog, "WARNING") if line.startswith("camera stalled again")]) == 3


def test_a_feed_that_stays_healthy_for_5_s_ends_the_stall_and_the_next_one_gets_a_full_15_s(caplog):
    watchdog, clock, recovery = make_watchdog()
    watchdog.attach()
    reads(watchdog, clock, 12)                                    # a stall since 1002.0, three recoveries tried
    assert recovery.calls == [1002.0, 1007.0, 1012.0]
    frames(watchdog, clock, 5.5)                                  # healthy from 1012.25: over at 1017.25
    last_frame = clock.t
    with caplog.at_level("INFO", logger=LOGGER):
        assert when_unavailable(watchdog, clock, 20) == last_frame + 2.0 + 15.0   # a new stall: 15 s again
    assert recovery.calls[3:] == [last_frame + 2.0, last_frame + 7.0, last_frame + 12.0]   # at once, then every 5 s
    [unavailable] = [line for line in lines(caplog, "ERROR") if line.startswith("camera unavailable")]
    assert "3 recovery attempt(s), 0 failed" in unavailable      # counted afresh, not 6


def test_a_gap_the_reads_missed_still_breaks_the_healthy_stretch(caplog):
    watchdog, clock, recovery = make_watchdog()
    watchdog.attach()
    with caplog.at_level("INFO", logger=LOGGER):
        reads(watchdog, clock, 3.5)
        watchdog.frame()                                          # 1003.5: a healthy stretch begins
        clock.t += 3.0
        watchdog.frame()                                          # 1006.5: 3 s without a frame, no read in between
        frames(watchdog, clock, 4.75)                             # 4.75 s since the gap, 7.75 s since 1003.5
        assert not [line for line in lines(caplog, "INFO") if line.startswith("camera recovered")]
        frames(watchdog, clock, STEP)
        assert len([line for line in lines(caplog, "INFO") if line.startswith("camera recovered")]) == 1


def test_healthy_seconds_sets_how_long_the_frames_must_be_steady(caplog):
    watchdog, clock, recovery = make_watchdog(healthy_seconds=2.0)
    watchdog.attach()
    with caplog.at_level("INFO", logger=LOGGER):
        reads(watchdog, clock, 3.5)
        watchdog.frame()
        frames(watchdog, clock, 1.75)
        assert not [line for line in lines(caplog, "INFO") if line.startswith("camera recovered")]
        frames(watchdog, clock, STEP)
        assert len([line for line in lines(caplog, "INFO") if line.startswith("camera recovered")]) == 1


def test_steady_5_fps_frames_never_stall(caplog):
    watchdog, clock, recovery = make_watchdog()
    watchdog.attach()
    with caplog.at_level("INFO", logger=LOGGER):
        for _ in range(300):        # 60 s at 5 fps, with a read that finds nothing between two frames
            clock.t += 0.1
            assert watchdog.tick() == "ok"
            clock.t += 0.1
            watchdog.frame()
    snapshot = watchdog.snapshot()
    assert snapshot["state"] == "ok" and 4.9 <= snapshot["fps"] <= 5.1
    assert snapshot["since"] == WALL + 1000.0 and recovery.calls == []
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_the_state_gives_up_on_time_even_while_a_recovery_blocks():
    """The slot session reads `state` on the event loop while tick() runs a recovery in a worker thread."""
    seen = []
    watchdog, clock, recovery = make_watchdog(during=lambda: (setattr(clock, "t", clock.t + 16),
                                                              seen.append(watchdog.state)))
    watchdog.attach()
    reads(watchdog, clock, 2)
    assert seen == ["unavailable"]                 # 15 s after the stall began, during the 16 s recovery
    assert watchdog.state == "unavailable"


def test_unavailable_outlasts_its_session_until_the_next_one_starts():
    watchdog, clock, recovery = make_watchdog("failed: the daemon runs without media")
    watchdog.attach()
    reads(watchdog, clock, 17)
    assert watchdog.state == "unavailable"
    watchdog.detach()                                                # the search ended
    assert watchdog.state == "unavailable"
    assert watchdog.snapshot()["media"] == RELEASED                 # the card can still say why
    since = watchdog.snapshot()["since"]
    clock.t += 60
    assert watchdog.snapshot()["since"] == since
    watchdog.attach()                                                # the next task: a fresh try
    assert watchdog.state == "ok" and watchdog.snapshot()["media"] is None
    assert set(reads(watchdog, clock, 1.75)) == {"ok"}
    calls = len(recovery.calls)
    reads(watchdog, clock, STEP)
    assert len(recovery.calls) == calls + 1                          # the new session's stall tries again at once


def test_detach_otherwise_makes_the_camera_idle():
    watchdog, clock, recovery = make_watchdog()
    watchdog.attach()
    reads(watchdog, clock, 3)
    assert watchdog.state == "recovering"
    watchdog.detach()
    assert watchdog.state == "idle" and watchdog.snapshot()["media"] is None
    assert set(reads(watchdog, clock, 20)) == {"idle"} and len(recovery.calls) == 1


def test_a_recovery_that_outlives_its_session_changes_nothing():
    watchdog, clock, recovery = make_watchdog("failed: no robot connection", during=lambda: watchdog.detach())
    watchdog.attach()
    assert reads(watchdog, clock, 2)[-1] == "idle"
    assert watchdog.snapshot() == {"state": "idle", "fps": 0.0, "since": WALL + 1002.0, "media": None}


def test_since_is_the_wall_time_of_the_last_state_change():
    watchdog, clock, recovery = make_watchdog("failed: x")
    assert watchdog.snapshot()["since"] == WALL + 1000.0
    clock.t = 1010.0
    watchdog.attach()
    assert watchdog.snapshot()["since"] == WALL + 1010.0
    reads(watchdog, clock, 2)                                        # stalled at 1012 (via recovering, then failed)
    assert watchdog.snapshot()["since"] == WALL + 1012.0
    reads(watchdog, clock, 1)
    assert watchdog.snapshot()["since"] == WALL + 1012.0             # no change, no new time
