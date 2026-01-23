# Slippipedia Case Study

**Repository**: https://github.com/cbartsch/Slippipedia
**Author**: cbartsch
**Status**: Active
**Language**: QML (88.8%), C++ (7.2%)
**Framework**: Felgo/Qt

## Overview

Slippipedia is a desktop replay manager that analyzes large Slippi replay collections and provides filterable statistics with persistent database caching. Replays only need to be analyzed once.

## Key Features

- **Batch Processing**: Analyze thousands of replays
- **Persistent Database**: Stats survive restarts
- **Rich Filtering**: Multi-dimensional queries
- **Video Export**: MP4 generation via FFmpeg
- **Slippi Desktop Integration**: Launch replays directly

## UI Structure

### Tab-Based Interface

1. **Setup/Configuration** - Analysis settings
2. **Statistics Dashboard** - Aggregate metrics
3. **Player Directory** - Tag/code selection
4. **Analytics Views** - Character, matchup, stage breakdowns
5. **Replay Browser** - Sequential navigation
6. **Punish Tracker** - Detailed punish analysis

### Filtering System

Filter by:
- Player tags and opponent codes
- Characters (player and opponent)
- Game result (win/loss)
- Duration and stock counts
- Timeframe
- Stage
- Punish properties (damage, kill, moves)

## Statistics Provided

**Aggregate**:
- Win rates
- Character usage (self and opponent)
- Stage frequency
- Opponent history

**Detailed**:
- Time-based trends
- Matchup-specific stats
- Stage-specific performance
- Punish effectiveness

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | QML (88.8%) |
| Core Logic | C++ (7.2%) |
| Build | CMake, QMake |
| Framework | Felgo SDK (Qt-based) |
| Parsing | slippc (C++ library) |
| Video | FFmpeg |

## Architecture

```
┌─────────────────────────────────┐
│         QML UI Layer            │
│  - Declarative interface        │
│  - Responsive design            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│         C++ Backend             │
│  - slippc integration           │
│  - Database management          │
│  - FFmpeg coordination          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│      Persistent Database        │
│  - Cached analysis results      │
│  - Fast filtering queries       │
└─────────────────────────────────┘
```

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| macOS | Full | FFmpeg via Homebrew |
| Windows | Full | FFmpeg bundled |

## Video Export

### Workflow

1. Enable "Dump frames" and "Dump audio" in Dolphin
2. Play replay in Slippi Desktop
3. Slippipedia combines with FFmpeg
4. Output: MP4 with configurable quality

### Quality Settings

- Resolution via Dolphin graphics
- "Full Resolution Frame Dumps" option
- Bitrate via `GFX.ini`

## Code Structure

```
Slippipedia/
├── qml/                 # QML UI components
├── src/                 # C++ source
├── slippc/              # Replay parser (submodule)
├── CMakeLists.txt
├── Slippipedia.pro      # QMake project
└── README.md
```

## Key Design Decisions

1. **QML for UI**: Responsive, cross-platform, declarative
2. **C++ for performance**: Replay parsing, database ops
3. **Persistent caching**: Only analyze new replays
4. **Modular filtering**: Compose complex queries
5. **FFmpeg integration**: Professional video export

## Relevance to ExPhil

**Not directly applicable** - Replay manager, not AI training.

**However**:
- Shows useful statistics for competitive analysis
- Database design patterns for replay metadata
- Could inform training data curation:
  - Filter by skill level (player codes)
  - Filter by matchup
  - Identify high-quality punishes for imitation

## References

- [Repository](https://github.com/cbartsch/Slippipedia)
- [Felgo SDK](https://felgo.com/)
- [slippc](https://github.com/pcrain/slippc)
