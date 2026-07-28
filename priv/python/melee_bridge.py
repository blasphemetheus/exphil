#!/usr/bin/env python3
"""
ExPhil Melee Bridge

Python-side bridge for communicating with Elixir via stdin/stdout.
Uses line-delimited JSON protocol.

Protocol:
  Request:  {"cmd": "...", ...params}
  Response: {"ok": true, ...data} or {"error": "message"}

Commands:
  - init: Initialize console and controller
  - step: Get next game state
  - send_controller: Send controller input
  - stop: Stop console and exit
"""

import json
import sys
import os
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# Configure logging to stderr AND file (stdout is for protocol)
logging.basicConfig(
    level=logging.DEBUG,
    format='[melee_bridge] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('/tmp/melee_bridge.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)

_LIBMELEE_INSTALL = (
    "pip install 'melee @ git+https://github.com/vladfi1/libmelee.git@v0.43.0'"
)

try:
    import melee
    from melee.enums import Button, Action
except ImportError:
    # NOT `pip install melee` — that pulls upstream from PyPI, which this
    # bridge cannot drive (see _require_forked_libmelee below).
    logger.error("libmelee not installed. Run: %s", _LIBMELEE_INSTALL)
    sys.exit(1)


def _require_forked_libmelee():
    """Fail early, and legibly, on upstream libmelee.

    The bridge needs vladfi1's fork. Upstream's MenuHelper.menu_helper_simple
    is a class-level function taking no `self`, so calling it on an instance
    binds the instance to `gamestate` and shifts every positional argument by
    one — surfacing much later as the baffling "got multiple values for
    argument 'connect_code'", after Dolphin has already been launched (and
    then leaked, because the crash skips teardown).

    requirements.txt pins the fork; this catches an environment that drifted
    from it.
    """
    import inspect

    try:
        params = list(
            inspect.signature(melee.MenuHelper.menu_helper_simple).parameters
        )
    except (AttributeError, TypeError, ValueError):
        return  # can't introspect — let the real call fail on its own terms

    if params and params[0] == "self":
        return

    version = getattr(melee, "__version__", None) or "unknown"
    raise RuntimeError(
        f"Incompatible libmelee ({version}): MenuHelper is not the stateful "
        f"instance API this bridge requires, so menu navigation would receive "
        f"shifted arguments. Upstream PyPI libmelee will not work — install "
        f"the fork ExPhil pins:\n    {_LIBMELEE_INSTALL}"
    )


# ============================================================================
# Game State Serialization
# ============================================================================

def serialize_player(player: melee.PlayerState) -> Dict[str, Any]:
    """Convert libmelee PlayerState to serializable dict."""
    if player is None:
        return None

    return {
        "character": player.character.value if hasattr(player.character, 'value') else player.character,
        "x": float(player.position.x) if player.position else 0.0,
        "y": float(player.position.y) if player.position else 0.0,
        "percent": float(player.percent),
        "stock": player.stock,
        "facing": player.facing.value if hasattr(player.facing, 'value') else (1 if player.facing else -1),
        "action": player.action.value if hasattr(player.action, 'value') else (player.action or 0),
        "action_frame": player.action_frame,
        "invulnerable": player.invulnerable,
        "invulnerability_left": player.invulnerability_left,
        "jumps_left": player.jumps_left,
        "on_ground": player.on_ground,
        "shield_strength": float(player.shield_strength),
        "hitstun_frames_left": player.hitstun_frames_left,
        "hitlag_left": player.hitlag_left,
        "speed_air_x_self": float(player.speed_air_x_self),
        "speed_ground_x_self": float(player.speed_ground_x_self),
        "speed_y_self": float(player.speed_y_self),
        "speed_x_attack": float(player.speed_x_attack),
        "speed_y_attack": float(player.speed_y_attack),
        # Nana (for Ice Climbers)
        "nana": serialize_nana(player.nana) if hasattr(player, 'nana') and player.nana else None,
        # Controller state (for imitation learning)
        "controller_state": serialize_controller_state(player.controller_state) if player.controller_state else None,
        # Port TYPE, so a caller can tell whether an opponent really is the CPU
        # it asked for. Without this the achieved level was unobservable from
        # Elixir and a silently-HUMAN dummy looked like a passive CPU
        # (GOTCHAS #57). libmelee zeroes cpu_level unless controller_status is
        # CONTROLLER_CPU, so read them together.
        "cpu_level": getattr(player, "cpu_level", 0),
        "controller_status": (
            player.controller_status.value
            if hasattr(getattr(player, "controller_status", None), "value")
            else getattr(player, "controller_status", None)
        ),
    }


def serialize_nana(nana) -> Optional[Dict[str, Any]]:
    """Serialize Nana (Ice Climbers partner)."""
    if nana is None:
        return None
    return {
        "x": float(nana.position.x) if nana.position else 0.0,
        "y": float(nana.position.y) if nana.position else 0.0,
        "percent": float(nana.percent),
        "stock": nana.stock,
        "action": nana.action.value if hasattr(nana.action, 'value') else (nana.action or 0),
        "facing": nana.facing.value if hasattr(nana.facing, 'value') else (1 if nana.facing else -1),
    }


def serialize_controller_state(cs) -> Dict[str, Any]:
    """Serialize controller state from replay/observation."""
    if cs is None:
        return None
    return {
        "main_stick": {"x": float(cs.main_stick[0]), "y": float(cs.main_stick[1])},
        "c_stick": {"x": float(cs.c_stick[0]), "y": float(cs.c_stick[1])},
        "l_shoulder": float(cs.l_shoulder),
        "r_shoulder": float(cs.r_shoulder),
        "button_a": cs.button[Button.BUTTON_A],
        "button_b": cs.button[Button.BUTTON_B],
        "button_x": cs.button[Button.BUTTON_X],
        "button_y": cs.button[Button.BUTTON_Y],
        "button_z": cs.button[Button.BUTTON_Z],
        "button_l": cs.button[Button.BUTTON_L],
        "button_r": cs.button[Button.BUTTON_R],
        "button_d_up": cs.button[Button.BUTTON_D_UP],
    }


def serialize_projectile(proj) -> Dict[str, Any]:
    """Serialize a projectile."""
    return {
        "owner": proj.owner,
        "x": float(proj.position.x) if proj.position else 0.0,
        "y": float(proj.position.y) if proj.position else 0.0,
        "type": proj.type.value if hasattr(proj.type, 'value') else (proj.type or 0),
        "subtype": proj.subtype,
        "speed_x": float(proj.speed.x) if proj.speed else 0.0,
        "speed_y": float(proj.speed.y) if proj.speed else 0.0,
    }


def serialize_game_state(gs: melee.GameState) -> Dict[str, Any]:
    """Convert full game state to serializable dict."""
    players = {}
    for port, player in gs.players.items():
        players[str(port)] = serialize_player(player)

    projectiles = [serialize_projectile(p) for p in gs.projectiles] if gs.projectiles else []

    return {
        "frame": gs.frame,
        "stage": gs.stage.value if hasattr(gs.stage, 'value') else (gs.stage or 0),
        "menu_state": gs.menu_state.value if hasattr(gs.menu_state, 'value') else (gs.menu_state or 0),
        "players": players,
        "projectiles": projectiles,
        "distance": float(gs.distance) if gs.distance else 0.0,
        # Stage-specific data
        "custom": {
            # Randall position for Yoshi's Story
            # FoD platform heights
            # etc.
        }
    }


# ============================================================================
# Controller Input
# ============================================================================

def apply_controller_input(controller: melee.Controller, input_data: Dict[str, Any]):
    """Apply controller input from Elixir command."""
    # Reset to neutral first
    controller.release_all()

    # Main stick
    main = input_data.get("main_stick", {})
    main_x = main.get("x", 0.5)
    main_y = main.get("y", 0.5)
    controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x, main_y)

    # C-stick
    c = input_data.get("c_stick", {})
    c_x = c.get("x", 0.5)
    c_y = c.get("y", 0.5)
    controller.tilt_analog(melee.enums.Button.BUTTON_C, c_x, c_y)

    # Shoulder (L trigger)
    shoulder = input_data.get("shoulder", 0.0)
    controller.press_shoulder(melee.enums.Button.BUTTON_L, shoulder)

    # Buttons
    buttons = input_data.get("buttons", {})
    button_map = {
        "a": Button.BUTTON_A,
        "b": Button.BUTTON_B,
        "x": Button.BUTTON_X,
        "y": Button.BUTTON_Y,
        "z": Button.BUTTON_Z,
        "l": Button.BUTTON_L,
        "r": Button.BUTTON_R,
        "d_up": Button.BUTTON_D_UP,
        # Start is SEND-ONLY (LRAS game-quit for replay finalization). It is
        # deliberately absent from serialize_controller_state: adding it there
        # would widen the 13-dim prev-action embedding contract.
        "start": Button.BUTTON_START,
    }

    for name, button in button_map.items():
        if buttons.get(name, False):
            controller.press_button(button)


# ============================================================================
# Bridge State
# ============================================================================

class MeleeBridge:
    """Manages the connection to Dolphin/Slippi."""

    def __init__(self):
        self.console: Optional[melee.Console] = None
        self.controller: Optional[melee.Controller] = None
        self.controller_port: int = 1
        self.opponent_port: int = 2
        self.menu_helper: Optional[melee.MenuHelper] = None
        self.running = False
        self.config = {}
        self._postgame_reported = False  # Track if we've reported postgame to Elixir
        self._last_in_game = False  # Track if we were in game last frame
        # Scripted port-2 dummy (drill opponents): none|stand|shield|jump|walk|cpu
        self.dummy_mode: str = "none"
        self.dummy_controller: Optional[melee.Controller] = None
        self.dummy_menu_helper: Optional[melee.MenuHelper] = None
        self._dummy_frame: int = 0

    def init(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the console and controller."""
        self.config = config

        dolphin_path = config.get("dolphin_path")
        iso_path = config.get("iso_path")
        self.controller_port = config.get("controller_port", 1)
        self.opponent_port = config.get("opponent_port", 2)
        online_delay = config.get("online_delay", 0)
        blocking_input = config.get("blocking_input", True)

        if not dolphin_path:
            return {"error": "dolphin_path is required"}
        if not iso_path:
            return {"error": "iso_path is required"}

        try:
            logger.info(f"Initializing console at {dolphin_path}")

            # Netplay (Slippi Direct, Phillip-style): the bot connects to an
            # opponent's connect code. Requires the machine's own Dolphin
            # User dir (with a logged-in Slippi account) copied into the
            # temp home, and the online input delay the account plays at.
            connect_code = config.get("connect_code") or ""
            online = bool(connect_code)
            self.connect_code = connect_code

            # Headless probes (task #5): Null gfx backend renders nothing
            # (no window, no GC adapter contention) and disable_audio drops
            # the audio pipeline entirely. With no video/audio throttle the
            # emulator runs unthrottled (GOTCHA #56) — which is SAFE here
            # only because blocking_input=True paces the game to the
            # controller loop: every frame waits for the policy's inputs, so
            # nothing is skipped and games run as fast as inference allows.
            self.headless = bool(config.get("headless"))

            console_kwargs = dict(
                path=dolphin_path,
                fullscreen=False,
                online_delay=int(config.get("online_delay") or 0),
                copy_home_directory=online,
                # Netplay account selection (#9 user_json wiring): the home
                # dir whose user.json/Config get copied into the temp User
                # dir. Unset => libmelee's default (the MAIN account's home)
                # — pass the BOT home (~/.config/SlippiOnline-bot) so Direct
                # sessions play as EXPH#288, never as DBTD#411.
                dolphin_home_path=(
                    os.path.expanduser(config["user_home"])
                    if config.get("user_home")
                    else None
                ),
                # Explicit gfx override (e.g. "Null" for a windowless
                # NETPLAY bot — the nix-ld wrapper's GL/Vulkan paths are
                # unaudited; Null is its only proven backend, 2026-07-21)
                **(
                    {"gfx_backend": config["gfx_backend"]}
                    if config.get("gfx_backend")
                    else {}
                ),
                # Distinct ports let parallel instances coexist (default 51441)
                slippi_port=int(config.get("slippi_port") or 51441),
                blocking_input=bool(config.get("blocking_input", self.headless)),
            )
            if self.headless:
                console_kwargs["gfx_backend"] = "Null"
                console_kwargs["disable_audio"] = True
                # emulation_speed for POLICY-DRIVEN headless games must be
                # 1.0 (real time): at unthrottled ~450fps the async
                # runner's ~8ms act cycle quantizes inputs to every ~3.6
                # game frames — short-hop presses (3f) mistimed, launches
                # rarely land, knockdowns/conversions collapse (2026-07-17,
                # r12 headless: ~0 kd/game vs 7-20 windowed). Frame-locked
                # input REPLAY (scenario suite) is timing-exact at any
                # speed and passes 0.0 (unthrottled) explicitly.
                console_kwargs["emulation_speed"] = float(
                    config.get("emulation_speed", 1.0)
                )
            # EXI inputs (WS5): the ExiAI build's native input path honors
            # analog trigger PRESSES that the pipe path drops (GOTCHAS
            # #66) — but analog RELEASE appears to LATCH: in every
            # EXI-mode probe game the TechRandom dummy's 20-40f analog
            # shield holds never released and it shield-broke into dizzy
            # (2026-07-17 validation). OPT-IN ONLY (config exi_inputs)
            # until release semantics are fixed; do NOT train on EXI-mode
            # games. libmelee raises ValueError on non-EXI builds — we
            # retry without it below.
            # Only PASS the kwarg when actually opting in. It is a no-op when
            # false, but libmelee builds that predate EXI support raise
            # TypeError on the unknown keyword — including stock 0.41.1, which
            # requirements.txt (melee>=0.40.0) happily allows. Passing it
            # unconditionally made the bridge silently require a newer/forked
            # libmelee than anything we pin.
            if config.get("exi_inputs"):
                console_kwargs["use_exi_inputs"] = True
            replay_dir = config.get("replay_dir")
            if replay_dir:
                # Flat per-instance replay dir -> unambiguous attribution
                # when several probes run at once (loops read exactly this
                # dir instead of mtime-scanning a shared ~/Slippi).
                replay_dir = os.path.expanduser(replay_dir)
                os.makedirs(replay_dir, exist_ok=True)
                console_kwargs["replay_dir"] = replay_dir
                console_kwargs["replay_monthly_folders"] = False

            # Minimal config - don't pass our logger, libmelee expects its own interface
            try:
                self.console = melee.Console(**console_kwargs)
            except ValueError as e:
                if console_kwargs.get("use_exi_inputs") and "custom dolphin build" in str(e):
                    logger.warning(
                        "EXI inputs unsupported by this Dolphin build — retrying "
                        "with pipe inputs (analog triggers will be dropped on "
                        "the ExiAI build, GOTCHAS #66)"
                    )
                    console_kwargs.pop("use_exi_inputs", None)
                    self.console = melee.Console(**console_kwargs)
                else:
                    raise
            except TypeError as e:
                # Older libmelee has no use_exi_inputs parameter at all (the
                # ValueError path above only covers a NEW libmelee talking to
                # an old Dolphin build). Drop it and retry on pipe inputs,
                # which is the default everything trains on anyway.
                if "use_exi_inputs" in str(e):
                    logger.warning(
                        "libmelee %s has no use_exi_inputs — EXI inputs need a "
                        "newer libmelee; retrying with pipe inputs",
                        getattr(melee, "__version__", "unknown"),
                    )
                    console_kwargs.pop("use_exi_inputs", None)
                    self.console = melee.Console(**console_kwargs)
                else:
                    raise
            if online:
                logger.info(f"Netplay mode: will connect to {connect_code}")
            if self.headless:
                logger.info(
                    "Headless mode: Null gfx, no audio, blocking input, "
                    f"exi_inputs={console_kwargs.get('use_exi_inputs')}"
                )

            # Enlarge the render window at the source: Console.__init__ wrote
            # Dolphin.ini into the temp User dir, and Dolphin reads it at
            # run() — so the window is CREATED at this size. WM-side resizes
            # (Hyprland rules) raced Dolphin's GL init and left the game
            # rendering 640x528 in a corner of the window.
            # Headless: no window and libmelee already configured the null
            # audio path via disable_audio — skip all ini editing (the DSP
            # Pulse/volume dance below exists to keep the REALTIME throttle,
            # which headless deliberately doesn't want).
            if not self.headless:
                try:
                    import configparser
                    ini_path = os.path.join(
                        self.console.dolphin_home_path, "Config", "Dolphin.ini"
                    )
                    # Default (lowercasing) parser ON PURPOSE: libmelee's own
                    # writer lowercases keys, and mixing cases creates DUPLICATE
                    # keys (RenderWindowWidth + renderwindowwidth) that crash
                    # strict configparser reads downstream. Dolphin reads keys
                    # case-insensitively — lowercase everywhere is fine (the old
                    # "SIDevice case" worry was a misdiagnosed udev problem).
                    ini = configparser.ConfigParser()
                    if os.path.isfile(ini_path):
                        ini.read(ini_path)
                    if not ini.has_section("Display"):
                        ini.add_section("Display")
                    if not ini.has_section("DSP"):
                        ini.add_section("DSP")
                    # ALSA-direct spews `alsa::poll() returned POLLERR` at full
                    # speed when the device is contended (once produced 29GB of
                    # log in 8 minutes); Pulse routes through pipewire-pulse and
                    # keeps audio working. Farm/unattended sessions pass
                    # no_audio to silence the game entirely.
                    if self.config.get("no_audio"):
                        # Keep a REAL backend even when silent: on Slippi's
                        # Dolphin 5.0 base the null backend ("No Audio Output")
                        # removes the emulation-speed throttle — games ran at
                        # ~520fps, the policy only saw 1 in ~8 frames, and the
                        # game-end transition hung the frame stream (observed
                        # 2026-07-12, four probes in a row). Volume=0 on Pulse
                        # is silent AND paced. (Headless mode skips this whole
                        # block: it WANTS the unthrottled path and paces via
                        # blocking_input instead.)
                        ini.set("DSP", "Backend", "Pulse")
                        ini.set("DSP", "Volume", "0")
                    else:
                        ini.set("DSP", "Backend", "Pulse")
                    # ~1.94x native 640x528, fits a 1080p screen with bar
                    ini.set("Display", "RenderWindowWidth",
                            str(self.config.get("window_width", 1240)))
                    ini.set("Display", "RenderWindowHeight",
                            str(self.config.get("window_height", 1023)))
                    ini.set("Display", "RenderWindowAutoSize", "False")
                    with open(ini_path, "w") as f:
                        ini.write(f)
                    logger.info(f"Render window size set in {ini_path}")
                except Exception as e:
                    logger.warning(f"Could not set render window size: {e}")

            self.controller = melee.Controller(
                console=self.console,
                port=self.controller_port,
                type=melee.ControllerType.STANDARD,
            )

            _require_forked_libmelee()
            self.menu_helper = melee.MenuHelper()

            # Scripted dummy on the opponent port (drill opponents). "cpu"
            # only uses the controller for menu setup (cpu_level does the
            # rest); scripted modes drive it every in-game frame; "external"
            # creates the controller for Elixir to drive via
            # send_controller(port: opponent_port).
            self.dummy_mode = config.get("dummy_mode", "none")
            if online and self.dummy_mode != "none":
                logger.warning("Netplay mode: opponent is remote — dummy disabled")
                self.dummy_mode = "none"
            # GOTCHAS #57 guard: a cpu_level > 0 makes the game AI own the
            # port at character select — driven dummies (external/scripted)
            # then press into a void while Slippi records their ghost
            # inputs. This exact combination silently replaced the
            # tech_random dummy with a lvl-3 CPU for the entire combo-drill
            # era (2026-07-08..13). Zero it and shout.
            driven_modes = ("external", "stand", "shield", "jump", "walk")
            if self.dummy_mode in driven_modes and \
                    int(config.get("dummy_cpu_level", 0)) > 0:
                logger.error(
                    "dummy_mode=%s is controller-driven but dummy_cpu_level=%s "
                    "would hand the port to the game AI (inputs ignored). "
                    "Forcing cpu_level=0. Use dummy_mode='cpu' for a CPU opponent.",
                    self.dummy_mode, config.get("dummy_cpu_level"),
                )
                config["dummy_cpu_level"] = 0
                self.config["dummy_cpu_level"] = 0

            # The mirror-image footgun: dummy_mode="cpu" with level 0 is NOT a
            # CPU. libmelee gates its whole CPU setup on `use_cpu = level > 0`,
            # so the port stays HUMAN with a connected-but-never-driven
            # controller — an idle standing dummy that reads as a passive CPU.
            # The CLI default IS 0, so `--dummy cpu` alone hits this every time.
            if self.dummy_mode == "cpu":
                lvl = int(config.get("dummy_cpu_level", 0))
                if lvl <= 0:
                    logger.error(
                        "dummy_mode='cpu' with dummy_cpu_level=%s is NOT a CPU — "
                        "libmelee skips CPU setup entirely when level <= 0 and the "
                        "port stays HUMAN and idle. Defaulting to level 1; pass "
                        "--dummy-cpu-level 1..9 explicitly.", lvl,
                    )
                    lvl = 1
                elif lvl > 9:
                    logger.error(
                        "dummy_cpu_level=%s is out of range; Melee levels are 1-9 "
                        "and an unreachable level makes libmelee drag the slider "
                        "forever. Clamping to 9.", lvl,
                    )
                    lvl = 9
                config["dummy_cpu_level"] = lvl
                self.config["dummy_cpu_level"] = lvl

                # libmelee raises ValueError("We can't force the CPU to pick
                # Sheik.") inside choose_character, and step()'s broad
                # except swallows it into a per-frame error while the session
                # limps on with no dummy. Fail at init instead.
                dchar = config.get("dummy_character", "fox")
                if isinstance(dchar, str) and dchar.upper() == "SHEIK":
                    return {
                        "error": "dummy_character='sheik' cannot be a CPU "
                                 "(libmelee refuses; Sheik is reached via Zelda). "
                                 "Use 'zelda'."
                    }

            if self.dummy_mode != "none":
                self.dummy_controller = melee.Controller(
                    console=self.console,
                    port=self.opponent_port,
                    type=melee.ControllerType.STANDARD,
                )
                self.dummy_menu_helper = melee.MenuHelper()
                logger.info(
                    f"Dummy enabled: mode={self.dummy_mode} port={self.opponent_port} "
                    f"character={self.config.get('dummy_character', 'fox')} "
                    f"cpu_level={self.config.get('dummy_cpu_level', 0)} "
                    f"(achieved level is logged at character select)"
                )

            # Run dolphin
            logger.info(f"Starting Dolphin with ISO: {iso_path}")
            logger.info(f"Console slippi_port: {self.console.slippi_port}")
            logger.info(f"Console slippi_address: {self.console.slippi_address}")
            self.console.run(iso_path=iso_path)
            logger.info("Dolphin process started")

            # Give Dolphin time to initialize before connecting
            import time
            sys.stderr.write("[melee_bridge] Waiting 3 seconds...\n")
            sys.stderr.flush()
            time.sleep(3)
            sys.stderr.write("[melee_bridge] Wait complete, now connecting...\n")
            sys.stderr.flush()

            # Connect to console
            sys.stderr.write("[melee_bridge] Calling console.connect()...\n")
            sys.stderr.flush()
            # Retry with backoff: a lone Dolphin is listening well inside
            # the 3s pre-wait, but PARALLEL instances boot slower (ISO read
            # + shader cache contention) and a single early connect() gave
            # up on instance 2 of 2 (observed 2026-07-16). Total budget
            # ~45s; each connect() attempt has its own internal timeout.
            connected = False
            for attempt in range(1, 6):
                connected = self.console.connect()
                if connected:
                    break
                wait = 2 * attempt
                sys.stderr.write(
                    f"[melee_bridge] connect() attempt {attempt} failed, retrying in {wait}s...\n"
                )
                sys.stderr.flush()
                time.sleep(wait)
            sys.stderr.write(f"[melee_bridge] connect() returned: {connected}\n")
            sys.stderr.flush()
            logger.info(f"Connect returned: {connected}")
            if not connected:
                return {"error": "Failed to connect to console"}

            logger.info("Connected to console")

            # Connect controller
            if not self.controller.connect():
                return {"error": "Failed to connect controller"}

            logger.info(f"Controller connected on port {self.controller_port}")

            if self.dummy_controller is not None:
                if not self.dummy_controller.connect():
                    return {"error": "Failed to connect dummy controller"}
                logger.info(f"Dummy controller connected on port {self.opponent_port}")

            self.running = True

            return {"ok": True, "controller_port": self.controller_port}

        except Exception as e:
            logger.exception("Failed to initialize")
            return {"error": str(e)}

    # Frames to wait at CSS for the dummy's CPU setup before starting anyway.
    # The dance is ~tens of frames; 600 (10s) is generous but bounded, so a
    # stuck slider degrades to "wrong level, logged" instead of hanging the
    # session forever.
    _DUMMY_SETUP_TIMEOUT_FRAMES = 600

    def _dummy_ready(self, gamestate) -> bool:
        """Whether port 1 may press START yet.

        True for every dummy mode except a CPU that has not finished being
        configured. See the autostart-race comment at the port-1 helper call.
        """
        if self.dummy_mode != "cpu":
            return True

        want = int(self.config.get("dummy_cpu_level", 0))
        if want <= 0:
            return True

        player = (gamestate.players or {}).get(self.opponent_port)
        if player is None:
            return False

        self._dummy_wait_frames = getattr(self, "_dummy_wait_frames", 0) + 1

        is_cpu = (
            getattr(player, "controller_status", None)
            == melee.enums.ControllerStatus.CONTROLLER_CPU
        )
        # libmelee zeroes cpu_level unless controller_status is CONTROLLER_CPU,
        # so both must be checked; and a held slider means the drag is mid-flight.
        ready = (
            is_cpu
            and getattr(player, "cpu_level", 0) == want
            and not getattr(player, "is_holding_cpu_slider", False)
        )

        if ready:
            if not getattr(self, "_dummy_ready_logged", False):
                logger.info(
                    "Dummy CPU configured: port=%s level=%s (after %s CSS frames)",
                    self.opponent_port, want, self._dummy_wait_frames,
                )
                self._dummy_ready_logged = True
            return True

        if self._dummy_wait_frames > self._DUMMY_SETUP_TIMEOUT_FRAMES:
            if not getattr(self, "_dummy_timeout_logged", False):
                logger.error(
                    "Dummy CPU setup TIMED OUT after %s frames — starting anyway. "
                    "Requested level=%s, port %s reports controller_status=%s "
                    "cpu_level=%s. The opponent will NOT be the requested CPU; "
                    "verify with the replay header (type 0=HUMAN, 1=CPU).",
                    self._dummy_wait_frames, want, self.opponent_port,
                    getattr(player, "controller_status", None),
                    getattr(player, "cpu_level", None),
                )
                self._dummy_timeout_logged = True
            return True

        return False

    def step(self, auto_menu: bool = True) -> Dict[str, Any]:
        """Get the next game state, optionally handling menu navigation."""
        if not self.running or not self.console:
            return {"error": "Console not initialized"}

        try:
            gamestate = self.console.step()

            if gamestate is None:
                # This can happen during menu transitions - try to recover
                logger.warning("console.step() returned None, attempting recovery...")
                import time
                time.sleep(0.1)
                try:
                    gamestate = self.console.step()
                except BrokenPipeError:
                    logger.info("Dolphin disconnected (BrokenPipeError during recovery)")
                    self.running = False
                    return {"error": "game_ended", "reason": "dolphin_disconnected"}
                if gamestate is None:
                    return {"error": "Console returned None (timeout or disconnected)"}

            # Handle menu navigation if requested
            is_in_game = gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]

            # Reset the dummy-setup watchdog once the game is running, so its
            # frame budget is PER character-select visit. Without this the
            # counter carries over and the timeout fires spuriously on the
            # post-game CSS (where the dummy port reads UNPLUGGED), logging a
            # scary false failure after a run that configured its CPU fine.
            if is_in_game and getattr(self, "_dummy_wait_frames", 0):
                self._dummy_wait_frames = 0
                self._dummy_timeout_logged = False
            is_menu = not is_in_game

            # Detect game-ending states
            is_postgame = gamestate.menu_state == melee.Menu.POSTGAME_SCORES

            # Track game state transitions
            if is_in_game and not self._last_in_game:
                # Just entered game - reset postgame flag
                self._postgame_reported = False
            self._last_in_game = is_in_game

            # Log state periodically (every 60 frames = 1 second)
            if gamestate.frame % 60 == 0:
                menu_name = gamestate.menu_state.name if hasattr(gamestate.menu_state, 'name') else str(gamestate.menu_state)
                # Log player stocks for debugging
                p1 = gamestate.players.get(1)
                p2 = gamestate.players.get(2)
                p1_info = f"P1:{p1.percent:.0f}%/{p1.stock}stk" if p1 else "P1:?"
                p2_info = f"P2:{p2.percent:.0f}%/{p2.stock}stk" if p2 else "P2:?"
                logger.info(f"Frame {gamestate.frame}: {menu_name} | {p1_info} {p2_info}")

            # Log game end detection (only on first postgame frame)
            if is_postgame and not self._postgame_reported:
                logger.info(f"Game ended - transitioning to {gamestate.menu_state.name if hasattr(gamestate.menu_state, 'name') else gamestate.menu_state}")

            # Skip menu navigation on first postgame frame to let Elixir handle it
            # On subsequent frames, allow navigation (for restart mode)
            skip_menu_nav = is_postgame and not self._postgame_reported
            if is_postgame:
                self._postgame_reported = True  # Mark as reported after this frame

            if is_menu and auto_menu and self.menu_helper and not skip_menu_nav:
                # Get character from config
                character = self.config.get("character", melee.Character.FOX)
                if isinstance(character, int):
                    character = melee.Character(character)
                elif isinstance(character, str):
                    character = melee.Character[character.upper()]

                stage = self.config.get("stage", melee.Stage.FINAL_DESTINATION)
                if isinstance(stage, int):
                    stage = melee.Stage(stage)
                elif isinstance(stage, str):
                    stage = melee.Stage[stage.upper()]

                # Dummy picks first (no autostart) so the main helper's
                # autostart can't fire before the opponent is on the roster.
                # ONLY during character select: stage select has one shared
                # cursor, and a second controller pressing buttons there
                # fights port 1 — observed as a session stuck at
                # STAGE_SELECT indefinitely (race; sometimes won, sometimes
                # not).
                if self.dummy_menu_helper is not None and \
                        gamestate.menu_state == melee.Menu.CHARACTER_SELECT:
                    dummy_char = self.config.get("dummy_character", "fox")
                    if isinstance(dummy_char, int):
                        dummy_char = melee.Character(dummy_char)
                    elif isinstance(dummy_char, str):
                        dummy_char = melee.Character[dummy_char.upper()]

                    self.dummy_menu_helper.menu_helper_simple(
                        gamestate,
                        self.dummy_controller,
                        dummy_char,
                        stage,
                        cpu_level=int(self.config.get("dummy_cpu_level", 0)),
                        autostart=False,
                        swag=False,
                    )

                # MenuHelper is a stateful class in libmelee >= 0.40 —
                # call on the instance created in init (static call shifts
                # every argument into the wrong slot via self).
                # With a connect_code, libmelee navigates Slippi Direct and
                # connects to the remote opponent instead of local VS.
                #
                # autostart is GATED on the dummy finishing its CPU setup.
                # libmelee configures a CPU as a multi-frame state machine
                # (walk to the HMN/CPU box, press A to flip the type, walk to
                # the level slider, grab, drag, release — one micro-step per
                # frame). The instant the port flips to CPU, Melee defaults it
                # to level 1, which ALREADY marks the roster ready — so an
                # unconditional autostart here presses START mid-dance and the
                # game begins with a level-1 CPU, or with the port still HMN.
                # Measured 2026-07-26: 5 of 6 recordings came up type=HUMAN
                # despite --dummy-cpu-level 9, which is why the "level 9 CPU"
                # stood in Wait 76% of frames and never jumped. The dummy
                # helper is also gated to CHARACTER_SELECT, so once START
                # advances the menu an unfinished drag can never complete.
                self.menu_helper.menu_helper_simple(
                    gamestate,
                    self.controller,
                    character,
                    stage,
                    connect_code=getattr(self, "connect_code", "") or "",
                    autostart=self._dummy_ready(gamestate),
                    swag=False,
                )

            # Scripted dummy behaviors run every in-game frame. "cpu" needs
            # no driving (game AI owns the port after menu setup); "external"
            # means Elixir drives the port via send_controller(port: N) —
            # python must not fight it.
            if is_in_game and self.dummy_controller is not None and \
                    self.dummy_mode in ("stand", "shield", "jump", "walk"):
                self._drive_dummy(gamestate)

            return {
                "ok": True,
                "is_menu": is_menu,
                "is_postgame": is_postgame,
                "game_state": serialize_game_state(gamestate),
            }

        except BrokenPipeError:
            logger.info("Dolphin disconnected (BrokenPipeError)")
            self.running = False
            return {"error": "game_ended", "reason": "dolphin_disconnected"}
        except Exception as e:
            logger.exception("Error during step")
            return {"error": str(e)}

    def _drive_dummy(self, gamestate):
        """Drive the scripted port-2 dummy for one frame.

        Modes (drill opponents, see docs/planning/DRILL_INFRASTRUCTURE.md):
          stand  - training dummy: neutral every frame
          shield - holds shield in a 2s-on / 1s-off pulse (never shield-breaks)
          jump   - hops on a 1.5s cycle
          walk   - strolls back and forth around center stage
        (tech_random needs a knockdown state machine — follow-up.)
        """
        c = self.dummy_controller
        self._dummy_frame += 1
        t = self._dummy_frame
        c.release_all()

        if self.dummy_mode == "shield":
            if t % 180 < 120:
                c.press_button(melee.Button.BUTTON_R)

        elif self.dummy_mode == "jump":
            if t % 90 == 0:
                c.press_button(melee.Button.BUTTON_Y)

        elif self.dummy_mode == "walk":
            p = gamestate.players.get(self.opponent_port)
            x = float(p.position.x) if p and p.position else 0.0
            # Half-tilt = walk, not dash; wander a lane around center
            if x > 20.0:
                c.tilt_analog(melee.Button.BUTTON_MAIN, 0.35, 0.5)
            elif x < -20.0:
                c.tilt_analog(melee.Button.BUTTON_MAIN, 0.65, 0.5)
            else:
                # Inside the lane: drift with a slow square wave
                going_right = (t % 240) < 120
                c.tilt_analog(
                    melee.Button.BUTTON_MAIN, 0.65 if going_right else 0.35, 0.5
                )
        # "stand" (and anything unrecognized): stay neutral after release_all

    def send_controller(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send controller input.

        Optional "port" routes to a specific controller — the dummy port's
        controller when it matches opponent_port (Elixir-driven dummies /
        self-play), otherwise the main controller. Requires the dummy to be
        enabled at init (dummy_mode != "none") so the controller exists.
        """
        if not self.running or not self.controller:
            return {"error": "Controller not initialized"}

        try:
            target = self.controller
            port = input_data.get("port")
            if port is not None and int(port) == self.opponent_port:
                if self.dummy_controller is None:
                    return {"error": "No controller on port %s (enable a dummy_mode at init)" % port}
                target = self.dummy_controller

            apply_controller_input(target, input_data)
            return {"ok": True}
        except BrokenPipeError:
            logger.info("Dolphin disconnected while sending controller input")
            self.running = False
            return {"error": "game_ended", "reason": "dolphin_disconnected"}
        except Exception as e:
            logger.exception("Error sending controller input")
            return {"error": str(e)}

    def stop(self) -> Dict[str, Any]:
        """Stop the console and clean up."""
        self.running = False
        errors = []

        # Controller disconnect - may fail if pipe already broken
        if self.controller:
            try:
                self.controller.disconnect()
            except BrokenPipeError:
                logger.debug("Controller pipe already closed")
            except Exception as e:
                errors.append(f"controller: {e}")

        # Console stop - may fail if already stopped
        if self.console:
            try:
                self.console.stop()
            except BrokenPipeError:
                logger.debug("Console pipe already closed")
            except Exception as e:
                errors.append(f"console: {e}")

            # console.stop() sometimes leaves Dolphin alive (observed frozen
            # at stage select after a sudden-death game). Force-kill by the
            # session's UNIQUE temp User dir — matches only our Dolphin.
            try:
                home = self.console.dolphin_home_path
                if home and "/tmp/" in str(home):
                    import subprocess
                    subprocess.run(["pkill", "-f", str(home)], timeout=5)
            except Exception as e:
                errors.append(f"dolphin force-kill: {e}")

        if errors:
            logger.warning(f"Cleanup warnings: {errors}")

        return {"ok": True}


# ============================================================================
# Main Protocol Loop
# ============================================================================

def make_json_serializable(obj):
    """Recursively convert numpy types to Python native types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def send_response(response: Dict[str, Any]):
    """Send a JSON response to stdout (Elixir)."""
    response = make_json_serializable(response)
    json_str = json.dumps(response, separators=(',', ':'))
    print(json_str, flush=True)


def _orphan_watchdog(bridge):
    """Exit if the Elixir side dies while the main thread is blocked.

    Stdin EOF only ends the main loop when the main thread is actually
    reading stdin — a bridge blocked inside a libmelee call (e.g. a dead
    Slippi connection) never returns to the loop and lives forever as an
    orphan. Observed: two such orphans held a deleted 29GB ALSA-spam log's
    inode and kept the disk full. When the parent (beam) dies we get
    reparented to init (ppid 1) — force-exit then.
    """
    import time
    while True:
        if os.getppid() == 1:
            logger.info("Parent process died — force exiting")
            try:
                if bridge.console:
                    bridge.console.stop()
            except Exception:
                pass
            os._exit(0)
        time.sleep(5)


def main():
    """Main protocol loop - read commands from stdin, write responses to stdout."""
    bridge = MeleeBridge()

    import threading
    threading.Thread(target=_orphan_watchdog, args=(bridge,), daemon=True).start()

    logger.info("Melee bridge started, waiting for commands...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            send_response({"error": f"Invalid JSON: {e}"})
            continue

        cmd = request.get("cmd")

        if cmd == "init":
            response = bridge.init(request.get("config", {}))
        elif cmd == "step":
            auto_menu = request.get("auto_menu", True)
            response = bridge.step(auto_menu=auto_menu)
        elif cmd == "send_controller":
            response = bridge.send_controller(request.get("input", {}))
        elif cmd == "stop":
            response = bridge.stop()
            send_response(response)
            break
        elif cmd == "ping":
            response = {"ok": True, "pong": True}
        else:
            response = {"error": f"Unknown command: {cmd}"}

        send_response(response)

    logger.info("Melee bridge shutting down")


if __name__ == "__main__":
    main()
