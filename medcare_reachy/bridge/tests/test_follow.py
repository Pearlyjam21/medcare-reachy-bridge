"""FaceFollower: moves only when active, when due, and when the face really moved; recentres when it is gone."""

from medcare_reachy.bridge.follow import MOVE_SECONDS, FaceFollower


class Head:
    def __init__(self):
        self.looks, self.recentred = [], 0

    def look_at(self, u, v, duration):
        self.looks.append((round(u, 3), round(v, 3), duration))

    def recentre(self):
        self.recentred += 1


def follower(t):
    head = Head()
    return FaceFollower(head.look_at, head.recentre, clock=lambda: t["now"], interval=0.4, deadband=0.06,
                        release_after=3.0), head


def test_follows_the_face_centre_only_when_active():
    t = {"now": 10.0}
    f, head = follower(t)
    assert f.update([0.1, 0.1, 0.4, 0.4]) is False and head.looks == []
    f.start()
    assert f.update([0.1, 0.1, 0.4, 0.4]) is True
    assert head.looks == [(0.3, 0.3, MOVE_SECONDS)]


def test_small_moves_and_quick_repeats_are_ignored():
    t = {"now": 10.0}
    f, head = follower(t)
    f.start()
    f.update([0.1, 0.1, 0.4, 0.4])
    t["now"] = 10.2
    assert f.update([0.5, 0.1, 0.4, 0.4]) is False          # too soon, even for a big move
    t["now"] = 10.5
    assert f.update([0.12, 0.1, 0.4, 0.4]) is False         # inside the deadband
    assert f.update([0.5, 0.1, 0.4, 0.4]) is True
    assert head.looks[-1][:2] == (0.7, 0.3) and f.moves == 2


def test_a_lost_face_recentres_once_after_a_while():
    t = {"now": 10.0}
    f, head = follower(t)
    f.start()
    f.update([0.1, 0.1, 0.4, 0.4])
    t["now"] = 12.0
    assert f.update(None) is False and head.recentred == 0
    t["now"] = 13.1
    assert f.update(None) is True and head.recentred == 1
    assert f.update(None) is False and head.recentred == 1   # only once
    t["now"] = 14.0
    assert f.update([0.6, 0.4, 0.2, 0.2]) is True              # back: follows again


def test_robot_errors_never_escape():
    t = {"now": 10.0}

    def broken(*args):
        raise RuntimeError("motors off")

    f = FaceFollower(broken, broken, clock=lambda: t["now"])
    f.start()
    assert f.update([0.1, 0.1, 0.4, 0.4]) is True
    f.stop()
    assert f.update([0.5, 0.5, 0.2, 0.2]) is False


def test_the_service_builds_the_follower_from_the_settings():
    """0.8.0-0.9.1 had the follower and its setting but never built one, so the head never followed."""
    from medcare_reachy.service import face_follower

    class Robot:
        def __init__(self):
            self.calls = []

        def look_at(self, u, v, duration):
            self.calls.append(("look_at", round(u, 2), round(v, 2)))

        def hold_head(self):
            self.calls.append(("hold_head",))

    robot = Robot()
    assert face_follower(robot, {"follow_face": False}) is None
    follower = face_follower(robot, {})            # on by default
    follower.start()
    assert follower.update([0.1, 0.1, 0.4, 0.4]) is True
    assert robot.calls == [("look_at", 0.3, 0.3)]
