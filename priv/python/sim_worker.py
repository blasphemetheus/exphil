#!/usr/bin/env python3
"""
ExPhil melee-sim-light worker (SIM_INTEGRATION.md step 2).

Owns one `melee_sim.EnvBatch` and speaks the same line-delimited JSON
protocol as `melee_bridge.py`: one request per stdin line, one response per
stdout line. Logging goes to stderr only — stdout is the protocol.

  Request:  {"cmd": "...", ...params}
  Response: {"ok": true, ...data} or {"ok": false, "error": "message"}

Commands
  ping                      -> {"ok": true, "pong": true}
  init    stage, players[], batch_size=1, length=256, seed=0, stocks=4,
          max_frame=-1    -> configures every env identically, resets, and
                             returns {"frames": [row, ...]} (one row per env)
          players[i] = {"character": "fox" | internal kind, "costume": 0,
                        "start_percent": 0, "facing": 0|1|-1, "team": -1}
  step    controllers[env][player] = SimState.controller_to_row/1 output
          (buttons {A,B,X,Y,Z,L,R,D_UP} 0/1, main_stick_x/y, c_stick_x/y,
          shoulder)       -> {"frames": [...], "terminal": [...]}
  reset   env_ids=[...] (default all) -> {"frames": [...]}
  save    env, keep=false, no_blob=false -> {"state": base64, "bytes", "state_id"?}
  upload  state (base64)  -> {"state_id"}      (cache a client blob)
  forget  state_ids=[...] (default all) -> {"forgotten"}
  restore env, state | state_id -> {"frames": [row]}   (state_id = cached, no blob transfer)
  stop                    -> {"ok": true} then exit

Rows are the sim's `gamestate_dtype` rows converted to nested JSON with the
sim's field names verbatim (`ExPhil.Bridge.SimState` reads them by name).
Environment: MSL_CORE_LIBRARY / MSL_DATA_DIR / PYTHONPATH select the sim
checkout (main is the base: ~/git/msl-main).
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="[sim_worker] %(levelname)s: %(message)s", stream=sys.stderr)
log = logging.getLogger("sim_worker")

try:
    import melee_sim as msl
except ImportError as exc:  # pragma: no cover - environment error
    log.error("melee_sim not importable (%s); set PYTHONPATH to the sim checkout", exc)
    sys.exit(1)


# ---------------------------------------------------------------------------
# numpy structured row -> plain JSON
# ---------------------------------------------------------------------------

def _scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def row_to_json(row: np.void) -> Any:
    """Recursively convert a structured numpy scalar/array to JSON-able data."""
    dtype = row.dtype
    if dtype.names is None:
        if isinstance(row, np.ndarray):
            return [row_to_json(x) for x in row]
        return _scalar(row)
    out: dict[str, Any] = {}
    for name in dtype.names:
        if name.startswith("_pad"):
            continue
        value = row[name]
        if isinstance(value, np.ndarray) and value.dtype.names is not None:
            out[name] = [row_to_json(x) for x in value]
        elif isinstance(value, np.ndarray):
            out[name] = value.tolist()
        elif value.dtype.names is not None:
            out[name] = row_to_json(value)
        else:
            out[name] = _scalar(value)
    return out


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _character(value: Any) -> int:
    if isinstance(value, str):
        return int(msl.Character[value.upper()])
    return int(value)


def _stage(value: Any) -> int:
    if isinstance(value, str):
        return int(msl.Stage[value.upper()])
    return int(value)


class SimWorker:
    def __init__(self) -> None:
        self.env: msl.EnvBatch | None = None
        self.batch_size = 1
        self.length = 256
        self.num_players = 2
        self.frames_written = 0  # rows consumed in the current buffer window
        self.cache: dict[int, bytes] = {}  # state_id -> savestate (save keep=true / upload)
        self.next_state_id = 0

    # -- lifecycle ---------------------------------------------------------

    def init(self, req: dict[str, Any]) -> dict[str, Any]:
        self.close()
        players_spec = req.get("players") or [{"character": "fox"}, {"character": "fox", "costume": 1}]
        self.num_players = len(players_spec)
        self.batch_size = int(req.get("batch_size", 1))
        self.length = int(req.get("length", 256))
        env = msl.EnvBatch(batch_size=self.batch_size, length=self.length, num_players=self.num_players)
        self.env = env
        players = []
        for p in players_spec:
            kwargs = {k: p[k] for k in ("costume", "start_percent", "facing", "team_id", "controller_port", "handicap") if k in p}
            players.append(msl.PlayerConfig(_character(p["character"]), **kwargs))
        config_kwargs: dict[str, Any] = {"stage": _stage(req.get("stage", "final_destination")), "players": players}
        for key in ("seed", "stocks", "max_frame", "damage_ratio", "is_teams", "friendly_fire"):
            if key in req:
                config_kwargs[key] = req[key]
        env.configure_match(**config_kwargs)
        env.reset_all()
        self.frames_written = 0
        return {"frames": self._frames(), "batch_size": self.batch_size, "num_players": self.num_players}

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    # -- stepping ----------------------------------------------------------

    def step(self, req: dict[str, Any]) -> dict[str, Any]:
        env = self._env()
        controllers = req.get("controllers")
        if self.frames_written + 1 >= self.length:
            # Buffer window exhausted: the current frame stays the state; restart the cursor.
            env.reset_cursor()
            self.frames_written = 0
        if controllers is not None:
            self._write_controllers(controllers)
        env.step()
        self.frames_written += 1
        return {"frames": self._frames(), "terminal": self._terminal()}

    def _write_controllers(self, controllers: list[list[dict[str, Any]]]) -> None:
        env = self._env()
        view = env.controller_action_view  # shape (length, batch); dtype players[4]
        row_index = self.frames_written
        for env_id, per_player in enumerate(controllers):
            for player, c in enumerate(per_player):
                if c is None:
                    continue
                slot = view[row_index, env_id]["players"][player]
                b = c.get("buttons", {})
                for name in ("A", "B", "X", "Y", "Z", "L", "R", "D_UP"):
                    if name in slot["buttons"].dtype.names:
                        slot["buttons"][name] = 1 if b.get(name) else 0
                slot["main_stick_x"] = float(c.get("main_stick_x", 0.0))
                slot["main_stick_y"] = float(c.get("main_stick_y", 0.0))
                slot["c_stick_x"] = float(c.get("c_stick_x", 0.0))
                slot["c_stick_y"] = float(c.get("c_stick_y", 0.0))
                slot["shoulder"] = float(c.get("shoulder", 0.0))

    def reset(self, req: dict[str, Any]) -> dict[str, Any]:
        env = self._env()
        env_ids = req.get("env_ids")
        if env_ids is None:
            env.reset_all()
        else:
            env.reset_matches(np.asarray(env_ids, dtype=np.int64))
        self.frames_written = 0
        return {"frames": self._frames()}

    def save(self, req: dict[str, Any]) -> dict[str, Any]:
        """Serialize env `env`. With `keep: true` the blob also stays in the
        worker's cache under a `state_id` so later restores skip the ~1 MB
        base64 round trip (search restores the same start 64 times)."""
        state = self._env().save(int(req.get("env", 0)))
        out: dict[str, Any] = {"bytes": len(state)}
        if req.get("keep"):
            out["state_id"] = self._cache_put(state)
        if not req.get("no_blob"):
            out["state"] = base64.b64encode(state).decode("ascii")
        return out

    def upload(self, req: dict[str, Any]) -> dict[str, Any]:
        """Put a client-held blob into the cache; returns its state_id."""
        state = base64.b64decode(req["state"])
        return {"state_id": self._cache_put(state), "bytes": len(state)}

    def forget(self, req: dict[str, Any]) -> dict[str, Any]:
        ids = req.get("state_ids")
        if ids is None:
            n = len(self.cache)
            self.cache.clear()
            return {"forgotten": n}
        n = 0
        for i in ids:
            n += 1 if self.cache.pop(int(i), None) is not None else 0
        return {"forgotten": n}

    def _cache_put(self, state: bytes) -> int:
        self.next_state_id += 1
        self.cache[self.next_state_id] = state
        return self.next_state_id

    def restore(self, req: dict[str, Any]) -> dict[str, Any]:
        env = self._env()
        if "state_id" in req:
            state = self.cache.get(int(req["state_id"]))
            if state is None:
                raise KeyError(f"unknown state_id {req['state_id']}")
        else:
            state = base64.b64decode(req["state"])
        env.restore(int(req.get("env", 0)), state)
        # restore rewrites native state only; refresh the observation row so
        # current_frame reflects the restored match.
        env.observe()
        return {"frames": self._frames()}

    # -- views -------------------------------------------------------------

    def _frames(self) -> list[Any]:
        env = self._env()
        return [row_to_json(env.current_frame[i]) for i in range(self.batch_size)]

    def _terminal(self) -> list[Any]:
        env = self._env()
        t = env.terminal_view[self.frames_written - 1] if self.frames_written > 0 else env.terminal_view[0]
        return [row_to_json(t[i]) for i in range(self.batch_size)]

    def _env(self) -> msl.EnvBatch:
        if self.env is None:
            raise RuntimeError("sim not initialized; send init first")
        return self.env


# ---------------------------------------------------------------------------
# protocol loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# framing: every message is a 4-byte big-endian length + payload (Erlang
# `{:packet, 4}`). A payload starting with '{' is a JSON request; a payload
# starting with BIN_STEP is a binary step (see step_binary). Responses use
# the same framing: JSON, or BIN_STEP + raw rows.
# ---------------------------------------------------------------------------

BIN_STEP = b"\x01"


def dtype_layout(dt: np.dtype) -> dict[str, Any]:
    """Recursive layout descriptor (offsets, formats, shapes) so the Elixir side
    builds its decoder/encoder from the sim's own dtype, never by hand."""
    if dt.fields is None:
        if dt.subdtype:
            sub, shape = dt.subdtype
            return {"kind": "array", "shape": list(shape), "item": dtype_layout(sub), "itemsize": dt.itemsize}
        return {"kind": "scalar", "fmt": dt.str, "itemsize": dt.itemsize}
    fields = [
        {"name": name, "offset": off, **dtype_layout(fdt)}
        for name, (fdt, off) in sorted(((n, (v[0], v[1])) for n, v in dt.fields.items()), key=lambda kv: kv[1][1])
    ]
    return {"kind": "struct", "itemsize": dt.itemsize, "fields": fields}


def layouts() -> dict[str, Any]:
    from melee_sim import dtypes as _dt

    return {
        "gamestate": dtype_layout(_dt.gamestate_dtype()),
        "terminal": dtype_layout(_dt.terminal_dtype()),
        "controller_input": dtype_layout(_dt.controller_input_dtype()),
    }


def _write_frame(payload: bytes) -> None:
    try:
        out = sys.stdout.buffer
        out.write(len(payload).to_bytes(4, "big"))
        out.write(payload)
        out.flush()
    except BrokenPipeError:
        # Elixir closed the port (normal shutdown race); nothing to say to.
        raise SystemExit(0)


def respond(payload: dict[str, Any]) -> None:
    _write_frame(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _read_frame() -> bytes | None:
    inp = sys.stdin.buffer
    head = inp.read(4)
    if len(head) < 4:
        return None
    n = int.from_bytes(head, "big")
    payload = b""
    while len(payload) < n:
        chunk = inp.read(n - len(payload))
        if not chunk:
            return None
        payload += chunk
    return payload


def step_binary(worker: "SimWorker", body: bytes) -> bytes:
    """Binary step: body = controller_input rows (batch x 112 bytes) or empty.
    Reply = BIN_STEP + gamestate rows (batch x 980) + terminal rows (batch x 16)."""
    from melee_sim import dtypes as _dt

    env = worker._env()
    if worker.frames_written + 1 >= worker.length:
        env.reset_cursor()
        worker.frames_written = 0
    if body:
        rows = np.frombuffer(body, dtype=_dt.controller_input_dtype())
        if rows.shape[0] != worker.batch_size:
            raise ValueError(f"expected {worker.batch_size} controller rows, got {rows.shape[0]}")
        env.controller_action_view[worker.frames_written][:] = rows
    env.step()
    worker.frames_written += 1
    frame = env.current_frame
    term = env.terminal_view[worker.frames_written - 1]
    return BIN_STEP + np.ascontiguousarray(frame).tobytes() + np.ascontiguousarray(term).tobytes()


def main() -> int:
    worker = SimWorker()
    handlers = {
        "init": worker.init,
        "step": worker.step,
        "reset": worker.reset,
        "save": worker.save,
        "restore": worker.restore,
        "upload": worker.upload,
        "forget": worker.forget,
    }
    while True:
        payload = _read_frame()
        if payload is None:
            break
        if payload[:1] == BIN_STEP:
            try:
                _write_frame(step_binary(worker, payload[1:]))
            except Exception as exc:
                log.exception("binary step failed")
                respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            req = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            respond({"ok": False, "error": f"invalid JSON: {exc}"})
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "ping":
                respond({"ok": True, "pong": True, "melee_sim": msl.__file__, "layout": layouts()})
            elif cmd == "stop":
                worker.close()
                respond({"ok": True})
                return 0
            elif cmd == "init":
                respond({"ok": True, **handlers[cmd](req), "layout": layouts()})
            elif cmd in handlers:
                respond({"ok": True, **handlers[cmd](req)})
            else:
                respond({"ok": False, "error": f"unknown cmd: {cmd!r}"})
        except Exception as exc:  # protocol must never die silently
            log.exception("command %s failed", cmd)
            respond({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
