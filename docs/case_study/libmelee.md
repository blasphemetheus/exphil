# libmelee Case Study

**Repository**: https://github.com/altf4/libmelee
**Author**: altf4 (also author of SmashBot)
**Status**: Active (foundational infrastructure)
**Language**: Python 3
**Documentation**: https://libmelee.readthedocs.io/

## Overview

libmelee is an open Python 3 API that provides programmatic access to Super Smash Bros. Melee game state through the Dolphin emulator. It is the **foundational infrastructure** that enables all modern Melee AI development.

**Key Capability**: Real-time read/write access to game memory, allowing observation of complete game state and input of controller commands.

## Why libmelee Matters

Without libmelee (or equivalent), Melee AI would require:
- Screen capture and computer vision
- Controller hardware emulation
- Reverse engineering memory addresses

libmelee provides all of this out of the box, enabling researchers to focus on AI rather than infrastructure.

## Architecture

```
┌─────────────────────────────────────────┐
│             Dolphin Emulator            │
│  ┌─────────────────────────────────┐   │
│  │        Melee Game Memory        │   │
│  │  - Player states                │   │
│  │  - Stage data                   │   │
│  │  - Menu state                   │   │
│  └──────────────┬──────────────────┘   │
│                 │ Memory Read          │
│  ┌──────────────┴──────────────────┐   │
│  │      Dolphin Memory API         │   │
│  └──────────────┬──────────────────┘   │
└─────────────────┼───────────────────────┘
                  │ Socket
┌─────────────────┼───────────────────────┐
│                 │                       │
│  ┌──────────────┴──────────────────┐   │
│  │         libmelee Console        │   │
│  │  - GameState parsing            │   │
│  │  - Menu navigation              │   │
│  │  - Controller abstraction       │   │
│  └──────────────┬──────────────────┘   │
│                 │                       │
│  ┌──────────────┴──────────────────┐   │
│  │        Controller Pipes         │   │
│  │  - Named pipes for input        │   │
│  │  - Virtual GC controller        │   │
│  └──────────────────────────────────┘   │
│              libmelee (Python)          │
└─────────────────────────────────────────┘
```

## Core Components

### Console

The main interface for connecting to Dolphin:

```python
import melee

console = melee.Console(
    path="/path/to/dolphin",
    slippi_address="127.0.0.1",  # For Slippi netplay
    online_delay=0,              # Frame delay
    fullscreen=False,
)

console.run(iso_path="/path/to/melee.iso")
console.connect()
```

### GameState

Complete game state observation per frame:

```python
gamestate = console.step()

# Access player data
p1 = gamestate.players[1]
print(f"Position: ({p1.position.x}, {p1.position.y})")
print(f"Action: {p1.action}")
print(f"Damage: {p1.percent}")

# Access stage data
print(f"Stage: {gamestate.stage}")
print(f"Frame: {gamestate.frame}")
```

### Controller

Virtual controller input:

```python
controller = melee.Controller(console=console, port=1)
controller.connect()

# Send inputs
controller.press_button(melee.Button.BUTTON_A)
controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.5, 0.5)  # Neutral
controller.release_all()
```

## Game State Fields

### Player State

libmelee exposes comprehensive player information:

| Field | Type | Description |
|-------|------|-------------|
| `position.x` | float | X coordinate |
| `position.y` | float | Y coordinate |
| `percent` | float | Damage percentage |
| `stock` | int | Remaining stocks |
| `facing` | bool | True = facing right |
| `action` | Action | Current action state |
| `action_frame` | int | Frame within action |
| `character` | Character | Character enum |
| `invulnerable` | bool | Invulnerability active |
| `invulnerability_left` | int | Remaining i-frames |
| `jumps_left` | int | Remaining jumps |
| `on_ground` | bool | Grounded vs airborne |
| `shield_strength` | float | Shield HP remaining |
| `hitlag_left` | int | Remaining hitlag frames |
| `hitstun_frames_left` | int | Remaining hitstun |

### Velocity Fields (Critical)

**Key Insight**: Melee tracks 5 separate velocity components, not 2:

| Field | Description |
|-------|-------------|
| `speed_air_x_self` | Self-induced horizontal (airborne) |
| `speed_ground_x_self` | Self-induced horizontal (grounded) |
| `speed_y_self` | Self-induced vertical |
| `speed_x_attack` | Knockback-induced horizontal |
| `speed_y_attack` | Knockback-induced vertical |

This distinction is **essential** for:
- Tech chase prediction (distinguishing self-movement from being hit)
- Combo calculations
- Recovery prediction

### ECB (Extended Collision Box)

```python
# Collision box corners
p1.ecb.top
p1.ecb.bottom
p1.ecb.left
p1.ecb.right
```

### Stage Data

```python
gamestate.stage  # Stage enum
gamestate.stage_select_cursor_x  # Menu cursor
gamestate.stage_select_cursor_y

# Platform positions (for dynamic stages)
# Accessible via stage-specific APIs
```

### Projectiles

```python
for projectile in gamestate.projectiles:
    print(f"Type: {projectile.type}")
    print(f"Position: ({projectile.position.x}, {projectile.position.y})")
    print(f"Owner: {projectile.owner}")
```

## Action States

libmelee provides an extensive `Action` enum with 399 possible states:

```python
from melee import Action

# Movement
Action.STANDING
Action.WALKING
Action.RUNNING
Action.DASH
Action.WAVEDASH_SLIDE

# Attacks
Action.NEUTRAL_ATTACK_1
Action.FORWARD_TILT
Action.DOWN_SMASH
Action.NEUTRAL_B_ATTACKING

# Defense
Action.SHIELD
Action.SPOT_DODGE
Action.ROLL_FORWARD
Action.ROLL_BACKWARD
Action.AIR_DODGE

# Recovery
Action.EDGE_HANGING
Action.EDGE_JUMP_1
Action.EDGE_ROLL_SLOW

# Hitstun
Action.DAMAGE_FLY_HIGH
Action.DAMAGE_FLY_NEUTRAL
Action.TUMBLING
```

## Menu Navigation

libmelee includes utilities for automatic menu handling:

```python
from melee import MenuHelper

# Navigate character select
MenuHelper.menu_helper_simple(
    gamestate,
    controller,
    character=melee.Character.FOX,
    stage=melee.Stage.FINAL_DESTINATION,
    costume=0,
    autostart=True,
)
```

## Frame Timing

Melee runs at 60 FPS with specific frame timing:

```python
# Main loop
while True:
    gamestate = console.step()  # Blocks until next frame

    # Process state
    action = my_ai.decide(gamestate)

    # Send input (will apply next frame)
    controller.tilt_analog(melee.Button.BUTTON_MAIN, action.stick_x, action.stick_y)
    if action.a_button:
        controller.press_button(melee.Button.BUTTON_A)
```

**Important**: Inputs sent during frame N take effect on frame N+1 (minimum 1 frame delay).

## Slippi Integration

libmelee integrates with Slippi for:
- Replay recording
- Online netplay
- Spectator mode

```python
console = melee.Console(
    path="/path/to/slippi-dolphin",
    slippi_address="127.0.0.1",
    slippi_port=51441,
)
```

## Character and Stage Enums

### Characters

```python
from melee import Character

Character.FOX
Character.FALCO
Character.MARTH
Character.SHEIK
Character.JIGGLYPUFF
Character.PEACH
Character.CAPTAIN_FALCON
Character.ICE_CLIMBERS
Character.PIKACHU
Character.SAMUS
Character.GANONDORF
Character.MEWTWO
Character.LINK
Character.ZELDA
Character.GAME_AND_WATCH
# ... all 26 characters
```

### Stages

```python
from melee import Stage

Stage.FINAL_DESTINATION
Stage.BATTLEFIELD
Stage.DREAMLAND
Stage.YOSHIS_STORY
Stage.FOUNTAIN_OF_DREAMS
Stage.POKEMON_STADIUM
# ... all legal stages
```

## Frame Data Utilities

libmelee includes frame data for decision making:

```python
from melee import framedata

# Get attack data
attack_data = framedata.attack_data(
    character=melee.Character.FOX,
    action=melee.Action.NAIR,
)

# Check if action is an attack
framedata.is_attack(character, action)

# Get roll distance
framedata.roll_distance(character, action)
```

## Practical Example

Complete agent loop:

```python
import melee

# Setup
console = melee.Console(path="/path/to/dolphin")
controller = melee.Controller(console=console, port=1)

console.run(iso_path="/path/to/melee.iso")
console.connect()
controller.connect()

# Game loop
while True:
    gamestate = console.step()

    # Handle menus
    if gamestate.menu_state in [
        melee.Menu.CHARACTER_SELECT,
        melee.Menu.STAGE_SELECT
    ]:
        melee.MenuHelper.menu_helper_simple(
            gamestate, controller,
            character=melee.Character.FOX,
            stage=melee.Stage.FINAL_DESTINATION,
        )
        continue

    # In-game logic
    if gamestate.menu_state == melee.Menu.IN_GAME:
        my_player = gamestate.players[1]
        opponent = gamestate.players[2]

        # Simple AI: approach opponent
        if my_player.position.x < opponent.position.x:
            controller.tilt_analog(melee.Button.BUTTON_MAIN, 1.0, 0.5)
        else:
            controller.tilt_analog(melee.Button.BUTTON_MAIN, 0.0, 0.5)
```

## Limitations

### What libmelee Cannot Do

1. **Training Data**: Cannot generate replays (use Slippi)
2. **Parallel Environments**: One console per process
3. **Headless Mode**: Requires Dolphin (though can be minimized)
4. **Perfect Timing**: Subject to Dolphin frame timing

### Known Issues

1. **Memory Addresses**: May need updates for Dolphin versions
2. **Platform Differences**: Some features Linux-only
3. **Slippi Compatibility**: Version matching required

## Integration with AI Projects

### slippi-ai

Uses libmelee for:
- Real-time game state during RL training
- Dolphin console management
- Controller input during evaluation

### SmashBot

Built entirely on libmelee for:
- Rule-based decision making
- Complete game state access
- Frame-perfect inputs

### ExPhil

Uses libmelee via Python bridge for:
- Dolphin integration (async and sync modes)
- Game state parsing
- Controller output

## Installation

```bash
pip install melee

# Or from source
git clone https://github.com/altf4/libmelee
cd libmelee
pip install -e .
```

## References

- [Repository](https://github.com/altf4/libmelee)
- [Documentation](https://libmelee.readthedocs.io/)
- [PyPI](https://pypi.org/project/melee/)
- [SmashBot (example usage)](https://github.com/altf4/SmashBot)
