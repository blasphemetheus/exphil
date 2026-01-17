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

# Configure logging to stderr (stdout is for protocol)
logging.basicConfig(
    level=logging.INFO,
    format='[melee_bridge] %(levelname)s: %(message)s',
    stream=sys.stderr
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
        "character": player.character.value if player.character else None,
        "x": float(player.position.x) if player.position else 0.0,
        "y": float(player.position.y) if player.position else 0.0,
        "percent": float(player.percent),
        "stock": player.stock,
        "facing": player.facing.value if player.facing else 1,
        "action": player.action.value if player.action else 0,
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
        "action": nana.action.value if nana.action else 0,
        "facing": nana.facing.value if nana.facing else 1,
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
        "type": proj.type.value if proj.type else 0,
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
        "stage": gs.stage.value if gs.stage else 0,
        "menu_state": gs.menu_state.value if gs.menu_state else 0,
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

            self.console = melee.Console(
                path=dolphin_path,
                online_delay=online_delay,
                blocking_input=blocking_input,
                setup_gecko_codes=True,
            )

            self.controller = melee.Controller(
                console=self.console,
                port=self.controller_port,
                type=melee.ControllerType.STANDARD,
            )

            self.menu_helper = melee.MenuHelper()

            # Run dolphin
            logger.info(f"Starting Dolphin with ISO: {iso_path}")
            self.console.run(iso_path=iso_path)

            # Connect to console
            logger.info("Connecting to console...")
            if not self.console.connect():
                return {"error": "Failed to connect to console"}

            logger.info("Connected to console")

            # Connect controller
            if not self.controller.connect():
                return {"error": "Failed to connect controller"}

            logger.info(f"Controller connected on port {self.controller_port}")
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
                return {"error": "Console returned None (timeout or disconnected)"}

            # Handle menu navigation if requested
            is_menu = gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]

            if is_menu and auto_menu and self.menu_helper:
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

                self.menu_helper.menu_helper_simple(
                    gamestate,
                    self.controller,
                    character_selected=character,
                    stage_selected=stage,
                    autostart=True,
                    swag=False,
                )

            return {
                "ok": True,
                "is_menu": is_menu,
                "game_state": serialize_game_state(gamestate),
            }

        except Exception as e:
            logger.exception("Error during step")
            return {"error": str(e)}

    def send_controller(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send controller input."""
        if not self.running or not self.controller:
            return {"error": "Controller not initialized"}

        try:
            apply_controller_input(self.controller, input_data)
            return {"ok": True}
        except Exception as e:
            logger.exception("Error sending controller input")
            return {"error": str(e)}

    def stop(self) -> Dict[str, Any]:
        """Stop the console and clean up."""
        try:
            if self.controller:
                self.controller.disconnect()
            if self.console:
                self.console.stop()
            self.running = False
            return {"ok": True}
        except Exception as e:
            logger.exception("Error stopping")
            return {"error": str(e)}


# ============================================================================
# Main Protocol Loop
# ============================================================================

def send_response(response: Dict[str, Any]):
    """Send a JSON response to stdout (Elixir)."""
    json_str = json.dumps(response, separators=(',', ':'))
    print(json_str, flush=True)


def main():
    """Main protocol loop - read commands from stdin, write responses to stdout."""
    bridge = MeleeBridge()

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
