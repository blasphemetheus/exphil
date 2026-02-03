# SmashBot Case Study

**Repository**: https://github.com/altf4/SmashBot
**Author**: altf4 (also author of libmelee)
**Status**: Active (rule-based baseline)
**Language**: Python 3
**Approach**: Hand-coded behavior trees (no ML)

## Overview

SmashBot is a rule-based Melee bot that uses hand-coded strategies, tactics, and chains to play the game. Unlike ML approaches, SmashBot demonstrates what's achievable with pure frame data knowledge and explicit programming.

**Key Value**: Provides a baseline for comparing ML approaches and demonstrates the complexity of Melee decision-making.

## Why Study SmashBot?

1. **Baseline Comparison**: What can rules achieve without learning?
2. **Frame Data Understanding**: How to use Melee mechanics explicitly
3. **Architecture Patterns**: Goal → Strategy → Tactic → Chain hierarchy
4. **Edge Cases**: Explicit handling of Melee quirks

## Current Limitations

From the README:
- Only plays Fox (as main character)
- Only plays against Marth (opponent)
- Only plays on Final Destination or frozen Pokemon Stadium
- No adaptive learning

## Architecture

### Behavior Tree Hierarchy

```
┌─────────────────────────────────────────┐
│                 Goals                    │
│  High-level objectives (KO opponent)    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│              Strategies                  │
│  Mid-level plans (edgeguard, recover)   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│               Tactics                    │
│  Specific maneuvers (bait, approach)    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│                Chains                    │
│  Frame-perfect sequences (wavedash)     │
└─────────────────────────────────────────┘
```

### Goals

Top-level objectives:

| Goal | Description |
|------|-------------|
| `KillOpponent` | Primary objective |
| `Survive` | Stay alive, recover |
| `Recover` | Get back to stage |
| `EdgeGuard` | Prevent opponent recovery |

### Strategies

Mid-level planning:

| Strategy | Description |
|----------|-------------|
| `Approach` | Close distance to opponent |
| `Retreat` | Create space |
| `Zone` | Control space with projectiles |
| `Punish` | Capitalize on openings |
| `EdgeGuard` | Intercept recovery |

### Tactics

Specific maneuvers:

| Tactic | Description |
|--------|-------------|
| `Pressure` | Shield pressure sequences |
| `Bait` | Induce opponent action |
| `Parry` | Powershield attempts |
| `Laser` | Fox laser spacing |
| `Juggle` | Keep opponent airborne |
| `Tech_Chase` | Follow tech options |

### Chains

Frame-perfect sequences:

| Chain | Frames | Description |
|-------|--------|-------------|
| `Wavedash` | ~14 | Air dodge into ground |
| `ShortHop` | 3 | Minimum jump |
| `L_Cancel` | 7 | Landing lag reduction |
| `WavelandOnPlatform` | ~20 | Platform movement |
| `ShineSpike` | Variable | Offstage shine |

## Code Structure

```
SmashBot/
├── Chains/               # Frame-perfect sequences
│   ├── wavedash.py
│   ├── shorthop.py
│   ├── lcancel.py
│   ├── grabandthrow.py
│   └── ...
├── Tactics/              # Specific maneuvers
│   ├── approach.py
│   ├── pressure.py
│   ├── retreat.py
│   ├── edgeguard.py
│   └── ...
├── Strategies/           # Mid-level plans
│   ├── approach.py
│   ├── punish.py
│   ├── recover.py
│   └── ...
├── Goals/                # High-level objectives
│   ├── kill.py
│   ├── survive.py
│   └── ...
├── difficultysettings.py # Configurable difficulty
├── framedata.py          # Attack/defense data
├── smashbot.py           # Main entry point
└── requirements.txt
```

## Decision Making

### Main Loop

```python
def decide(gamestate):
    # 1. Update current goal based on game state
    goal = select_goal(gamestate)

    # 2. Goal selects strategy
    strategy = goal.select_strategy(gamestate)

    # 3. Strategy selects tactic
    tactic = strategy.select_tactic(gamestate)

    # 4. Tactic selects/continues chain
    chain = tactic.select_chain(gamestate)

    # 5. Chain outputs controller state
    return chain.step(gamestate)
```

### Goal Selection

```python
def select_goal(gamestate):
    my_player = gamestate.players[BOT_PORT]
    opponent = gamestate.players[OPPONENT_PORT]

    # Offstage? Recover
    if my_player.position.y < -20:
        return RecoverGoal()

    # Opponent offstage? Edgeguard
    if opponent.position.y < -10:
        return EdgeGuardGoal()

    # Default: kill
    return KillGoal()
```

### Strategy Example: Edgeguard

```python
class EdgeGuardStrategy:
    def select_tactic(self, gamestate):
        opponent = gamestate.players[OPPONENT_PORT]

        # Below stage? Shine spike
        if opponent.position.y < -50:
            return ShineSpikesTactic()

        # Near ledge? Cover options
        if opponent.position.y < 0:
            return CoverLedgeTactic()

        # High recovery? Intercept
        return InterceptTactic()
```

### Chain Example: Wavedash

```python
class Wavedash(Chain):
    def __init__(self, direction):
        self.direction = direction
        self.frame = 0

    def step(self, gamestate):
        controller = Controller()

        if self.frame == 0:
            # Frame 1: Jump
            controller.press_button(Button.Y)
        elif self.frame == 3:
            # Frame 4: Air dodge at angle
            controller.press_button(Button.L)
            controller.tilt_analog(
                Button.MAIN,
                self.direction,
                0.35  # Slight down angle
            )
        elif self.frame > 14:
            # Done
            self.interruptable = True

        self.frame += 1
        return controller
```

## Frame Data Usage

SmashBot relies heavily on frame data:

```python
# Attack properties
NAIR_STARTUP = 4
NAIR_ACTIVE = 4, 5, 6, 7
NAIR_ENDLAG = 8

# Is opponent in a punishable state?
def is_punishable(opponent):
    # Check if in endlag
    if opponent.action == Action.LANDING:
        return opponent.action_frame < 4  # Landing lag frames

    # Check for whiffed attack
    if is_attack(opponent.action):
        return opponent.action_frame > attack_active_end(opponent)

    return False
```

### Tech Chase Logic

```python
def predict_tech(opponent):
    # Opponent must be in knockdown
    if opponent.action not in KNOCKDOWN_STATES:
        return None

    # Calculate where they'll end up for each tech option
    tech_in_place_pos = opponent.position
    tech_left_pos = opponent.position.x - TECH_ROLL_DISTANCE
    tech_right_pos = opponent.position.x + TECH_ROLL_DISTANCE
    missed_tech_pos = opponent.position  # They stay put

    return {
        'in_place': tech_in_place_pos,
        'left': tech_left_pos,
        'right': tech_right_pos,
        'missed': missed_tech_pos,
    }
```

## Difficulty Settings

SmashBot can be tuned:

```python
class DifficultySettings:
    # Reaction time (frames)
    REACTION_TIME = 12  # ~200ms human

    # Execution errors
    WAVEDASH_SUCCESS_RATE = 0.95
    L_CANCEL_SUCCESS_RATE = 0.90

    # Strategy randomization
    APPROACH_MIX = {
        'nair': 0.4,
        'drill': 0.3,
        'grab': 0.3,
    }
```

## Strengths

### What Rules Do Well

1. **Frame-Perfect Tech**: Wavedash, L-cancel, shine timing
2. **Reaction Time Control**: Can be tuned to human-like
3. **Deterministic**: Same situation → same response
4. **Explainable**: Clear decision logic
5. **No Training Required**: Works immediately

### Specific Capabilities

| Capability | Quality |
|------------|---------|
| Wavedashing | Excellent |
| L-canceling | Excellent |
| Shield pressure | Good |
| Tech chasing | Good |
| Edgeguarding | Good |
| Combo execution | Limited |
| Adaptation | None |

## Weaknesses

### What Rules Struggle With

1. **Adaptation**: Cannot learn opponent patterns
2. **Generalization**: New matchups require new code
3. **Creativity**: Limited to programmed options
4. **Mixups**: Predictable over time
5. **Scaling**: More matchups = exponential code

### Specific Limitations

| Limitation | Impact |
|------------|--------|
| Fox only | Cannot play other characters |
| Marth matchup | No other opponent knowledge |
| 2 stages | No platform stage handling |
| No learning | Cannot improve from experience |
| Predictable | Humans learn to exploit |

## Comparison to ML

| Aspect | SmashBot | ML (slippi-ai) |
|--------|----------|----------------|
| Training | None | Days-weeks |
| Adaptation | None | Continuous |
| Characters | 1 | Any with data |
| Matchups | 1 | Many |
| Execution | Perfect | Human-like |
| Creativity | None | Emergent |
| Explainability | High | Low |
| Development | Months | Compute-bound |

## Lessons for ExPhil

### What to Learn

1. **Behavior Tree Structure**: Goal → Strategy → Tactic → Chain
2. **Frame Data Importance**: Attack windows, tech timings
3. **Edge Case Handling**: Explicit Melee quirks
4. **Modular Design**: Separable decision components

### What ML Should Improve

1. **Adaptation**: Learn from opponent behavior
2. **Generalization**: Handle multiple matchups
3. **Creativity**: Discover novel strategies
4. **Scalability**: Add characters without code

### Hybrid Potential

Consider combining approaches:
- ML for high-level strategy selection
- Rules for frame-perfect execution
- ML for adaptation, rules for safety

## Running SmashBot

```bash
# Install
git clone https://github.com/altf4/SmashBot
cd SmashBot
pip install -r requirements.txt

# Run
python smashbot.py \
    --dolphin /path/to/dolphin \
    --iso /path/to/melee.iso \
    --port 1 \
    --opponent 2 \
    --stage FINAL_DESTINATION
```

## References

- [Repository](https://github.com/altf4/SmashBot)
- [libmelee](https://github.com/altf4/libmelee) - Underlying API
- [Melee Frame Data](https://ikneedata.com/) - Attack properties
- [SmashWiki](https://www.ssbwiki.com/) - Game mechanics
