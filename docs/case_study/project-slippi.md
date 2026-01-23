# Project Slippi Case Study

**Repository**: https://github.com/project-slippi/project-slippi
**Website**: https://slippi.gg
**Author**: Fizzi
**Status**: Active
**License**: GPL-3.0

## Overview

Project Slippi is the foundational infrastructure for modern Melee, providing rollback netcode, replay recording, and the data format that enables all Melee AI research.

## Ecosystem Components

| Component | Repository | Purpose |
|-----------|------------|---------|
| Dolphin Fork | [Ishiiruka](https://github.com/project-slippi/Ishiiruka) | Modified emulator |
| Launcher | [slippi-launcher](https://github.com/project-slippi/slippi-launcher) | Desktop app |
| slippi-js | [slippi-js](https://github.com/project-slippi/slippi-js) | JavaScript SDK |
| Set Stats | [slippi-set-stats](https://github.com/project-slippi/slippi-set-stats) | Tournament stats |
| ASM Codes | [slippi-ssbm-asm](https://github.com/project-slippi/slippi-ssbm-asm) | Gecko codes |

## .slp Replay Format

**Format**: UBJSON (Universal Binary JSON)

### File Structure

```
[File Signature: 11 bytes]
[UBJSON Header]
[Size: 4-byte big-endian]
[Payload Sizes Table: 256 entries]
[GameStart Event]
[Frame Events (repeating)]
[GameEnd Event]
[Metadata (UBJSON)]
```

### Event Types

| Code | Event | Purpose |
|------|-------|---------|
| 0x35 | Event Payloads | Size declarations |
| 0x36 | GameStart | Initialization |
| 0x37 | Pre-Frame Update | Input reconstruction |
| 0x38 | Post-Frame Update | State for analysis |
| 0x39 | GameEnd | Conclusion marker |
| 0x3A | Frame Start | RNG seed |
| 0x3B | Item Update | Projectile state |
| 0x3C | Frame Bookend | Frame complete signal |

### Frame Data

**Pre-frame** (input):
- Random seed
- Button inputs
- Stick positions
- Trigger values

**Post-frame** (result):
- Position, velocity
- Damage %, shield
- Action state, stocks
- Hit information

## Ishiiruka (Dolphin Fork)

### Modifications

1. **EXI Port B Communication**
   - Receives game state from gecko codes
   - Responds to playback requests
   - Intermediary for external systems

2. **Rust Integration**
   - Custom Rust submodule linked
   - EXI device management
   - Authentication, reporting

3. **Build Variants**
   - Netplay: Online competitive
   - Playback: Replay viewing

### Fast-Forward for AI Training

**Gecko codes enable**:
- Skip rendering/graphics
- Accelerate frame advancement
- 100x+ speed for training

## slippi-js SDK

```javascript
import { SlippiGame } from "@slippi/slippi-js";

const game = new SlippiGame("match.slp");
const stats = game.getStats();
const frames = game.getFrames();

// Live processing
const game = new SlippiGame("match.slp", { processOnTheFly: true });
```

**Capabilities**:
- Parse .slp files
- Compute built-in statistics
- Frame-level data access
- Live replay processing

## slippi-set-stats

**Purpose**: Tournament broadcast statistics

```bash
# Generate stats from set
./slippi-set-stats /path/to/replays/
# Outputs: output.json
```

**Output includes**:
- Per-game: Stage, characters, winner, duration
- Summary: Kill moves, neutral openers, damage
- BTS format: Broadcast-ready simplified stats

## Desktop Launcher

**Features**:
- Slippi Online matchmaking
- Dolphin auto-updater
- Replay browser and analysis
- Player authentication

**Stack**:
- TypeScript/React
- Electron
- Material-UI

## AI Integration Points

### 1. Replay Files
- Parse via slippi-js or community parsers
- Extract frame-level training data

### 2. EXI Communication
- Real-time game state exchange
- Direct agent integration

### 3. Fast-Forward Codes
- Train at 100x+ speed
- Essential for RL

### Data Flow

```
Melee (with Gecko Codes)
    ↓ EXI Port B
Ishiiruka (Slippi Dolphin)
    ↓ .slp files
Slippi Launcher
    ↓
slippi-js / peppi / py-slippi
    ↓
AI Training Pipeline (ExPhil)
```

## Relevance to ExPhil

**Critical dependencies**:
1. **.slp format**: All training data comes from Slippi replays
2. **Fast-forward codes**: Enable faster-than-realtime RL
3. **Dolphin integration**: Required for live play

**ExPhil uses**:
- Peppi for replay parsing (faster than slippi-js)
- MeleePort for Dolphin communication
- Fast-forward codes for self-play training

## References

- [Project Slippi](https://github.com/project-slippi/project-slippi)
- [slippi-js](https://github.com/project-slippi/slippi-js)
- [Slippi Wiki](https://github.com/project-slippi/slippi-wiki)
- [Slippi.gg](https://slippi.gg)
