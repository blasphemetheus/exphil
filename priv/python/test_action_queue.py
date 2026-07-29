"""Offline gate for the bridge delay queue (LATENCY_ARCHITECTURE #3).

Pins the exact-D contract before any live wiring: an action scheduled at
frame t for t+D fires at t+D — never earlier, never smeared — and a frame
regression (new game) drops stale scheduling.

Run: python3 priv/python/test_action_queue.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from melee_bridge import ActionQueue


def test_exact_d_shift():
    for d in (1, 2, 4, 7):
        q = ActionQueue()
        t = 100
        q.schedule(t + d, {"id": "a"})
        # Never early: frames t .. t+d-1 yield nothing
        for f in range(t, t + d):
            assert q.pop_due(f) == [], f"D={d}: fired early at frame {f}"
        # Exactly on time
        assert q.pop_due(t + d) == [{"id": "a"}], f"D={d}: missing at t+D"
        # Never twice
        assert q.pop_due(t + d + 1) == []


def test_catchup_ordering():
    # If the reader skips frames (lag spike), everything due fires at once,
    # oldest first — an action is late-but-ordered, never lost.
    q = ActionQueue()
    q.schedule(10, "a")
    q.schedule(11, "b")
    q.schedule(12, "c")
    assert q.pop_due(12) == ["a", "b", "c"]


def test_same_frame_order_preserved():
    q = ActionQueue()
    q.schedule(5, "first")
    q.schedule(5, "second")
    assert q.pop_due(5) == ["first", "second"]


def test_frame_regression_clears():
    # New game: Melee restarts at frame -123. Stale actions must not fire
    # into the new game's timeline.
    q = ActionQueue()
    q.pop_due(3000)
    q.schedule(3002, "stale")
    assert q.pop_due(-123) == []
    assert len(q) == 0
    # And the queue works normally afterwards
    q.schedule(-121, "fresh")
    assert q.pop_due(-121) == ["fresh"]


def test_monotone_menus_no_clear():
    # Forward jumps (menu -> game transitions skip frame numbers) keep the
    # queue; only regression clears.
    q = ActionQueue()
    q.pop_due(50)
    q.schedule(60, "kept")
    assert q.pop_due(55) == []
    assert q.pop_due(200) == ["kept"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")
