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

try:
    import melee
    from melee.enums import Button, Action
except ImportError:
    logger.error("libmelee not installed. Run: pip install melee")
    sys.exit(1)


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

            # Minimal config - don't pass our logger, libmelee expects its own interface
            self.console = melee.Console(
                path=dolphin_path,
                fullscreen=False,
            )

            # Enlarge the render window at the source: Console.__init__ wrote
            # Dolphin.ini into the temp User dir, and Dolphin reads it at
            # run() — so the window is CREATED at this size. WM-side resizes
            # (Hyprland rules) raced Dolphin's GL init and left the game
            # rendering 640x528 in a corner of the window.
            try:
                import configparser
                ini_path = os.path.join(
                    self.console.dolphin_home_path, "Config", "Dolphin.ini"
                )
                ini = configparser.ConfigParser()
                # Preserve key case: the default optionxform lowercases keys
                # on rewrite (SIDevice0 -> sidevice0), which Dolphin's
                # case-sensitive reader then ignores
                ini.optionxform = str
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
                    ini.set("DSP", "Backend", "No Audio")
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

            self.menu_helper = melee.MenuHelper()

            # Scripted dummy on the opponent port (drill opponents). "cpu"
            # only uses the controller for menu setup (cpu_level does the
            # rest); scripted modes drive it every in-game frame; "external"
            # creates the controller for Elixir to drive via
            # send_controller(port: opponent_port).
            self.dummy_mode = config.get("dummy_mode", "none")
            if self.dummy_mode != "none":
                self.dummy_controller = melee.Controller(
                    console=self.console,
                    port=self.opponent_port,
                    type=melee.ControllerType.STANDARD,
                )
                self.dummy_menu_helper = melee.MenuHelper()
                logger.info(
                    f"Dummy enabled: mode={self.dummy_mode} port={self.opponent_port}"
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
            connected = self.console.connect()
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
                self.menu_helper.menu_helper_simple(
                    gamestate,
                    self.controller,
                    character,
                    stage,
                    autostart=True,
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
