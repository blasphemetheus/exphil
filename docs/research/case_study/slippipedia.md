# Slippipedia

**Repository:** https://github.com/cbartsch/Slippipedia
**Author:** Christian Bartsch (cbartsch)
**Language:** QML (88.8%), C++ (7.2%)
**Status:** Active (v2.2 released January 2025)
**Purpose:** Desktop replay analysis and management tool

## Overview

Slippipedia is a cross-platform desktop application for analyzing and managing large collections of Slippi replay files. It functions as "Your flexible replay manager" - parsing replays once into a SQLite database for fast subsequent queries and comprehensive statistics.

## Technology Stack

| Component | Technology |
|-----------|------------|
| UI Framework | QML + Felgo SDK (Qt-based) |
| Backend | C++ for replay parsing |
| Parser | slippc (Git submodule) |
| Database | SQLite (local persistence) |
| Video Export | FFmpeg integration |
| Platforms | macOS, Windows, Linux |

## Key Features

### Statistics & Analytics
- **Win rate tracking** by character, matchup, stage, time period
- **Opponent tracking** with detailed per-opponent statistics
- **Character usage** breakdown for player and opponents
- **Stage frequency** analysis

### Punish Analysis
- Extraction and cataloging of offensive combo sequences
- Damage dealt and move count per punish
- Kill property tracking (whether punishes result in KOs)
- Frame duration of punish sequences
- Filtering by damage range, move count, player percent

### Replay Management
- **One-time parsing** with persistent SQLite storage
- **Fast database lookup** for subsequent queries
- **Auto-analysis** option for new replays
- Support for multiple connect-codes and player tags
- Game mode filtering (Ranked, Unranked, Direct)

### Video Export
- MP4 conversion from Dolphin recordings
- Export options: full game, single stocks, or specific punishes
- Audio synchronization support (v2.2+)

## Database Schema

Replay metadata stored includes:
- Player characters, colors, stocks, display names
- Stage ID and game duration
- Win/loss outcomes
- Performance metrics (punish, defense, neutral, speed)
- Connect codes and timestamps

## Data Flow

```
.slp files → slippc parser → SQLite database → QML UI
                                    ↓
                            Fast queries for:
                            - Statistics
                            - Filtering
                            - Punish analysis
```

## Relevance to AI Training

### Training Data Organization
- **Normalized game states**: Database extracts position, action, damage, stocks - maps directly to training embeddings
- **Punish sequences**: Identified combo sequences align with temporal dependency learning
- **Matchup filtering**: Organize data for character-specific agent training

### Quality Indicators
- Game duration and stock counts indicate competitive vs casual matches
- Opponent characterization enables filtering for high-level play
- Performance metrics (neutral win rate, conversion rate) could inform sample weighting

### Practical Integration
- slippc parsing pipeline demonstrates efficient binary replay extraction
- Multi-dimensional statistics suggest patterns for curriculum learning
- Database architecture shows how to manage large replay collections

## Version History

| Version | Date | Key Features |
|---------|------|--------------|
| v2.0 | Oct 2022 | MP4 export, neutral game analysis |
| v2.1 | June 2023 | Game mode filtering, UI improvements |
| v2.2 | Jan 2025 | Multi-filter, Linux build, audio fixes |

## Comparison to ExPhil

| Aspect | Slippipedia | ExPhil |
|--------|-------------|--------|
| Purpose | Post-hoc analysis | Real-time gameplay |
| Data format | Aggregated statistics | Frame-by-frame state |
| Output | Visualizations | Controller actions |
| Use case | Player improvement | Bot development |

## Key Insight

Slippipedia demonstrates that Melee gameplay produces statistically distinctive patterns at the matchup, stage, and temporal level. The success of its filtering and grouping features validates that these dimensions are meaningful for organizing training data.

## Links

- [GitHub Repository](https://github.com/cbartsch/Slippipedia)
- [Releases](https://github.com/cbartsch/Slippipedia/releases)
- [slippc Parser](https://github.com/pcrain/slippc)
