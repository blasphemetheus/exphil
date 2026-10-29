#!/usr/bin/env python3
"""slippi-ai opponent worker — runs vladfi1's agent as ExPhil's port-2 opponent.

Runs INSIDE the slippi-ai venv (~/git/slippi-ai/.venv), never ExPhil's own
python env: the only things it imports are slippi_ai + its libmelee fork
(0.47.3). It never talks to Dolphin. ExPhil's native Elixir bridge
(`ExPhil.Bridge.MeleePort`, libmelee_ex) owns the one Dolphin and BOTH pipe
controllers; the Elixir side forwards each in-game frame's
`ExPhil.Bridge.GameState` here as JSON, this worker rebuilds a real
`melee.GameState` from it, runs slippi-ai's unmodified `eval_lib.Agent.step`
(their Parser -> observation filter -> delayed async inference -> controller
decode), captures the decoded controller with a recorder standing in for
`melee.Controller`, and returns it as the bridge input map the runner sends
to port 2 (`AsyncRunner.drive_elixir_dummy`, rich `step_game/3` contract).

Framing: Erlang `{:packet, 4}` — 4-byte big-endian length prefix, JSON body,
both directions. stdout is reserved for the protocol; everything else
(TF/absl noise, our logs) goes to stderr — fd 1 is dup'd away before any
import that might print.

Commands:
  {"cmd": "init", "model_path", "port", "opponent_port", "console_delay",
   "character", "sample_temperature", "policy_delay_override"?}
      -> {"ok": true, "policy_delay", "effective_queue_delay",
          "allowed_characters", "agent_type", "warmup_s"}
  {"cmd": "step", "game_state": {...ExPhil GameState as JSON...}}
      -> {"ok": true, "input": {main_stick, c_stick, shoulder, buttons},
          "step_ms": float}
  {"cmd": "ping"} -> {"ok": true, "pong": true}
  {"cmd": "stop"} -> {"ok": true}  (then exits)

Launch (from Elixir, see scripts/play_vs_slippi_ai.exs):
  ~/git/slippi-ai/.venv/bin/python priv/python/slippi_ai_worker.py
"""

import json
import os
import struct
import sys
import time

# Protocol channel = the ORIGINAL stdout fd. Everything that would print to
# fd 1 from here on (TF, absl, stray prints) lands on stderr instead.
_PROTO_OUT = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)
sys.stdout = sys.stderr

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # their inference is CPU by design
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import logging  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[slippi_ai_worker] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("slippi_ai_worker")

import numpy as np  # noqa: E402
import melee  # noqa: E402
from melee import enums as menums  # noqa: E402
import melee.gamestate as mgs  # noqa: E402

from slippi_ai import eval_lib, nametags, saving  # noqa: E402
from slippi_ai.controller_lib import LEGAL_BUTTONS  # noqa: E402


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

def _read_packet():
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        return None
    (n,) = struct.unpack(">I", header)
    body = b""
    while len(body) < n:
        chunk = sys.stdin.buffer.read(n - len(body))
        if not chunk:
            return None
        body += chunk
    return body


def _send(obj):
    body = json.dumps(obj, separators=(",", ":")).encode()
    _PROTO_OUT.write(struct.pack(">I", len(body)) + body)


# ---------------------------------------------------------------------------
# Controller recorder — stands in for melee.Controller inside eval_lib.Agent.
# send_controller() calls press/release/tilt/shoulder on it; we keep the
# last decoded state and hand it back as the ExPhil bridge input map.
# ---------------------------------------------------------------------------

_BUTTON_KEY = {
    melee.Button.BUTTON_A: "a",
    melee.Button.BUTTON_B: "b",
    melee.Button.BUTTON_X: "x",
    melee.Button.BUTTON_Y: "y",
    melee.Button.BUTTON_Z: "z",
    melee.Button.BUTTON_L: "l",
    melee.Button.BUTTON_R: "r",
    melee.Button.BUTTON_D_UP: "d_up",
}


class RecordingController:
    def __init__(self, port):
        self.port = port
        self.reset()

    def reset(self):
        self.main = (0.5, 0.5)
        self.c = (0.5, 0.5)
        self.shoulder = 0.0
        self.buttons = {k: False for k in _BUTTON_KEY.values()}

    # melee.Controller surface used by slippi_ai.controller_lib.send_controller
    def press_button(self, b):
        k = _BUTTON_KEY.get(b)
        if k is not None:
            self.buttons[k] = True

    def release_button(self, b):
        k = _BUTTON_KEY.get(b)
        if k is not None:
            self.buttons[k] = False

    def tilt_analog(self, which, x, y):
        if which == melee.Button.BUTTON_MAIN:
            self.main = (float(x), float(y))
        elif which == melee.Button.BUTTON_C:
            self.c = (float(x), float(y))

    def press_shoulder(self, _which, amount):
        self.shoulder = float(amount)

    def release_all(self):
        self.reset()

    def flush(self):
        pass

    def connect(self):
        return True

    def disconnect(self):
        pass

    def as_input(self):
        return {
            "main_stick": {"x": self.main[0], "y": self.main[1]},
            "c_stick": {"x": self.c[0], "y": self.c[1]},
            "shoulder": self.shoulder,
            "buttons": dict(self.buttons),
        }


# ---------------------------------------------------------------------------
# ExPhil GameState JSON -> melee.GameState
# ---------------------------------------------------------------------------

def _enum_or_unknown(enum_cls, unknown_cls, value, default):
    if value is None:
        return default
    try:
        return enum_cls(int(value))
    except ValueError:
        return unknown_cls(int(value)) if unknown_cls else default


def _controller_state(d):
    cs = melee.ControllerState()
    if not d:
        return cs
    ms = d.get("main_stick") or {}
    c = d.get("c_stick") or {}
    cs.main_stick = (float(ms.get("x", 0.5)), float(ms.get("y", 0.5)))
    cs.c_stick = (float(c.get("x", 0.5)), float(c.get("y", 0.5)))
    cs.l_shoulder = float(d.get("l_shoulder") or 0.0)
    cs.r_shoulder = float(d.get("r_shoulder") or 0.0)
    for btn, key in _BUTTON_KEY.items():
        v = bool(d.get("button_" + key, False))
        cs.button[btn] = v
        # Their parser reads processed_button (the post-deadzone view); for a
        # pipe-driven bot the two coincide.
        cs.processed_button[btn] = v
    return cs


def _player(d, is_nana=False, parent=None):
    p = melee.PlayerState()
    p.character = _enum_or_unknown(
        melee.Character, None,
        d.get("character") if not is_nana else (parent.character.value if parent else None),
        melee.Character.UNKNOWN_CHARACTER,
    )
    p.position = mgs.Position(float(d.get("x") or 0.0), float(d.get("y") or 0.0))
    p.percent = float(d.get("percent") or 0.0)
    p.stock = int(d.get("stock") or 0)
    facing = d.get("facing")
    p.facing = bool(facing) if isinstance(facing, bool) else (facing is None or int(facing) >= 0)
    p.action = _enum_or_unknown(melee.Action, mgs.UnknownAnimation, d.get("action"), mgs.UnknownAnimation(-1))
    p.action_frame = int(d.get("action_frame") or 0)
    p.invulnerable = bool(d.get("invulnerable") or False)
    p.hitstun_frames_left = int(d.get("hitstun_frames_left") or 0)
    p.jumps_left = int(d.get("jumps_left") or 0)
    p.on_ground = bool(d.get("on_ground", True))
    p.shield_strength = float(d.get("shield_strength") if d.get("shield_strength") is not None else 60.0)
    p.speed_air_x_self = float(d.get("speed_air_x_self") or 0.0)
    p.speed_ground_x_self = float(d.get("speed_ground_x_self") or 0.0)
    p.speed_y_self = float(d.get("speed_y_self") or 0.0)
    p.speed_x_attack = float(d.get("speed_x_attack") or 0.0)
    p.speed_y_attack = float(d.get("speed_y_attack") or 0.0)
    p.controller_state = _controller_state(d.get("controller_state"))
    p.connectCode = d.get("connect_code") or ""
    if not is_nana and d.get("nana"):
        # ExPhil's compact Nana lacks jumps/shield/on_ground; defaults are
        # the neutral values (Ice Climbers are out of scope for this rig).
        p.nana = _player(d["nana"], is_nana=True, parent=p)
    return p


def _stage(v):
    if v is None:
        return melee.Stage.NO_STAGE
    v = int(v)
    s = menums.to_internal_stage(v)  # ExPhil.Bridge.GameState.stage is the EXTERNAL id
    if s is melee.Stage.NO_STAGE:
        try:
            s = melee.Stage(v)  # tolerate an already-internal id
        except ValueError:
            s = melee.Stage.NO_STAGE
    return s


def _projectile(i, d):
    pr = melee.Projectile()
    pr.position = mgs.Position(float(d.get("x") or 0.0), float(d.get("y") or 0.0))
    pr.speed = mgs.Position(float(d.get("speed_x") or 0.0), float(d.get("speed_y") or 0.0))
    pr.owner = int(d.get("owner") if d.get("owner") is not None else -1)
    pr.type = _enum_or_unknown(melee.ProjectileType, mgs.UnknownProjectileType, d.get("type"), melee.ProjectileType.UNKNOWN_PROJECTILE)
    pr.subtype = int(d.get("subtype") or 0)
    # LIMITATION: ExPhil.Bridge.Projectile carries no spawn_id, so their
    # ItemAssigner gets a synthetic id (list index). Slot identity across
    # frames is approximate when several projectiles coexist.
    pr.spawn_id = np.uint32(i)
    return pr


def build_gamestate(gs, ports):
    g = melee.GameState()
    g.frame = int(gs.get("frame"))
    g.stage = _stage(gs.get("stage"))
    g.menu_state = melee.Menu.IN_GAME
    g.distance = float(gs.get("distance") or 0.0)
    fl, fr = gs.get("fod_platform_left"), gs.get("fod_platform_right")
    if fl is not None and fr is not None:
        g.fod_platforms = mgs.FoDPlatforms(np.float32(fl), np.float32(fr))
    players = gs.get("players") or {}
    for port in ports:
        pd = players.get(str(port)) or players.get(port)
        if pd is None:
            raise ValueError(f"port {port} missing from game_state players {sorted(players)}")
        g.players[port] = _player(pd)
    g.projectiles = [_projectile(i, d) for i, d in enumerate(gs.get("projectiles") or [])]
    return g


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self):
        self.agent = None
        self.recorder = None
        self.ports = None
        self.started_game = False
        self.last_frame = None
        self.step_times = []
        self.steps = 0

    def init(self, cfg):
        eval_lib.disable_gpus()
        t0 = time.time()
        model_path = os.path.expanduser(cfg["model_path"])
        port = int(cfg.get("port", 2))
        opp = int(cfg.get("opponent_port", 1))
        console_delay = int(cfg.get("console_delay", 0))
        self.ports = (port, opp)

        state = saving.load_state_from_disk(model_path)
        policy_delay = int(state["config"]["policy"]["delay"])

        override = cfg.get("policy_delay_override")
        if override is not None and int(override) != policy_delay:
            # In-memory analogue of scripts/update_delay.py: shrinks the
            # agent's input->action queue. UNTRAINED regime for the network
            # (medium-v2 was trained at 21) — a symmetry experiment, not a
            # fair reading of their bot.
            log.warning("policy delay override %s -> %s (off-distribution for this network)",
                        policy_delay, override)
            state["config"]["policy"]["delay"] = int(override)
            policy_delay = int(override)

        if console_delay > policy_delay:
            raise ValueError(
                f"console_delay {console_delay} > policy delay {policy_delay}: "
                "lower ExPhil's --reaction-delay or raise --slippi-ai-delay")

        allowed = eval_lib.allowed_characters(state["config"])
        allowed_names = [c.name.lower() for c in allowed] if allowed else None
        want = (cfg.get("character") or "fox").lower()
        if allowed_names is not None and want not in allowed_names:
            raise ValueError(f"character {want} not in model's allowed set {allowed_names}")

        self.recorder = RecordingController(port)
        # Mirrors scripts/eval_two.py's per-port agent flags (defaults) with
        # async inference on, exactly as their two-agent harness runs it.
        self.agent = eval_lib.build_agent(
            port=port,
            opponent_port=opp,
            controller=self.recorder,
            state=state,
            console_delay=console_delay,
            name=nametags.DEFAULT_NAME,
            sample_temperature=float(cfg.get("sample_temperature", 1.0)),
            compile=True,
            async_inference=True,
            fake=False,
            batch_steps=0,
            platform=None,
            tf=dict(jit_compile=False),
            jax=dict(seed=int(cfg.get("seed", 0)), pack_args=False,
                     functionalize=False, dtype="float32"),
            name_change_mode=eval_lib.NameChangeMode.FIXED,
            mirror=False,
        )
        self.agent.start()
        agent_type = "rl" if ("rl_config" in state or "agent_config" in state) else "imitation"
        log.info("agent ready: %s type=%s policy_delay=%d console_delay=%d queue=%d chars=%s (%.1fs)",
                 model_path, agent_type, policy_delay, console_delay,
                 policy_delay - console_delay, allowed_names, time.time() - t0)
        return {
            "ok": True,
            "policy_delay": policy_delay,
            "console_delay": console_delay,
            "effective_queue_delay": policy_delay - console_delay,
            "allowed_characters": allowed_names,
            "agent_type": agent_type,
            "warmup_s": time.time() - t0,
        }

    def step(self, gs):
        if self.agent is None:
            return {"ok": False, "error": "not initialized"}
        g = build_gamestate(gs, self.ports)
        # eval_lib.Agent resets its parser/filter/recurrent state on frame
        # -123. If we attach mid-game (or a frame regression marks a new
        # game without -123 being seen), fake the reset frame once.
        regressed = self.last_frame is not None and g.frame < self.last_frame
        if not self.started_game or regressed:
            if g.frame != -123:
                g.frame = -123
            self.started_game = True
        self.last_frame = g.frame

        t0 = time.perf_counter()
        self.agent.step(g)  # sends into self.recorder via controller_lib.send_controller
        dt = (time.perf_counter() - t0) * 1000.0
        self.steps += 1
        self.step_times.append(dt)
        if len(self.step_times) >= 1800:  # every 30 s of frames
            arr = np.array(self.step_times)
            log.info("step_ms mean %.2f p99 %.2f max %.2f (n=%d)",
                     arr.mean(), np.percentile(arr, 99), arr.max(), len(arr))
            if arr.mean() > 12.0:
                log.warning("slippi-ai agent is slow: mean %.1f ms/frame (budget 16.7)", arr.mean())
            self.step_times = []
        return {"ok": True, "input": self.recorder.as_input(), "step_ms": dt}

    def stop(self):
        if self.agent is not None:
            try:
                self.agent.stop()
            except Exception:  # noqa: BLE001
                log.exception("agent.stop failed")
            self.agent = None
        return {"ok": True}


def main():
    w = Worker()
    log.info("worker started, waiting for init (python %s, libmelee %s)",
             sys.version.split()[0], getattr(melee, "__version__", "?"))
    while True:
        pkt = _read_packet()
        if pkt is None:
            log.info("stdin closed — exiting")
            w.stop()
            return
        try:
            req = json.loads(pkt)
        except json.JSONDecodeError as e:
            _send({"ok": False, "error": f"bad json: {e}"})
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "init":
                resp = w.init(req)
            elif cmd == "step":
                resp = w.step(req["game_state"])
            elif cmd == "ping":
                resp = {"ok": True, "pong": True}
            elif cmd == "stop":
                resp = w.stop()
                _send(resp)
                return
            else:
                resp = {"ok": False, "error": f"unknown cmd {cmd!r}"}
        except Exception as e:  # noqa: BLE001
            log.exception("command %s failed", cmd)
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        _send(resp)


if __name__ == "__main__":
    main()
