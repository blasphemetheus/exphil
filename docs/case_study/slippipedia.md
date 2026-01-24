# Slippipedia

**Repository:** https://github.com/cbartsch/Slippipedia
**Author:** cbartsch
**Language:** QML/Qt (C++ backend)
**Status:** Active, maintained
**Purpose:** Desktop replay browser with SQLite persistence and advanced analytics

## Overview

Slippipedia is a desktop application for browsing, searching, and analyzing Slippi replay collections. Unlike browser-based tools, it uses SQLite for persistent storage, enabling fast queries across thousands of replays without re-parsing.

## Architecture

### Technology Stack

```
┌─────────────────────────────────────────────────┐
│                 Qt/QML Frontend                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │   Replay    │  │   Stats     │  │  Filter │ │
│  │   Browser   │  │   Views     │  │  Panel  │ │
│  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────┬───────────────────────┘
                          │
┌─────────────────────────┴───────────────────────┐
│                  C++ Backend                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │   Replay    │  │   SQLite    │  │  Stats  │ │
│  │   Parser    │  │   Storage   │  │  Engine │ │
│  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────┘
```

### Key Components

1. **Replay Parser**: Native C++ parsing for speed
2. **SQLite Database**: Indexed storage for fast queries
3. **QML UI**: Modern, responsive interface
4. **Stats Engine**: Computes analytics from stored data

## Features

### Replay Management

- **Batch Import**: Scan directories for .slp files
- **Metadata Extraction**: Character, stage, duration, winner
- **Connect Code Tracking**: Associate replays with player identities
- **Duplicate Detection**: Avoid re-importing same replays

### Search & Filtering

```
Filters:
├── Character (own/opponent)
├── Stage
├── Date range
├── Duration
├── Player codes
├── Win/loss
└── Game mode (ranked/unranked/direct)
```

### Analytics

#### Punish Analysis
- Opening type (neutral win, whiff punish, edgeguard)
- Damage per opening
- Kill confirms
- Combo trees

#### Matchup Statistics
- Win rate by character matchup
- Stage win rates
- Performance over time trends

#### Technical Execution
- L-cancel rates
- Wavedash frequency
- Tech success rates

## Database Schema

```sql
-- Core tables (simplified)
CREATE TABLE replays (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    hash TEXT,
    date DATETIME,
    stage INTEGER,
    duration_frames INTEGER,
    winner INTEGER
);

CREATE TABLE players (
    id INTEGER PRIMARY KEY,
    replay_id INTEGER REFERENCES replays(id),
    port INTEGER,
    character INTEGER,
    connect_code TEXT,
    stocks_remaining INTEGER,
    damage_dealt REAL
);

CREATE TABLE punishes (
    id INTEGER PRIMARY KEY,
    replay_id INTEGER REFERENCES replays(id),
    player_port INTEGER,
    start_frame INTEGER,
    end_frame INTEGER,
    damage REAL,
    kills BOOLEAN,
    opening_type TEXT
);

-- Indexes for fast queries
CREATE INDEX idx_players_code ON players(connect_code);
CREATE INDEX idx_players_char ON players(character);
CREATE INDEX idx_replays_date ON replays(date);
```

## Punish Detection Algorithm

Slippipedia implements sophisticated punish detection:

```
Punish Detection:
1. Track opponent hitstun/tumble states
2. Punish starts when opponent enters hitstun
3. Punish continues while:
   - Opponent in hitstun/tumble/tech situation
   - OR gap < N frames (combo window)
4. Punish ends when:
   - Opponent returns to actionable state
   - Gap exceeds threshold
   - Stock lost

Opening Classification:
- Neutral win: Both players actionable before punish
- Whiff punish: Opponent in lag when hit
- Edgeguard: Opponent offstage or on ledge
- Tech chase: Opponent missed tech or in tech animation
```

## Performance

### Import Speed
- ~100-200 replays/second on modern hardware
- Initial import builds SQLite database
- Subsequent queries are instant

### Query Performance
- Character matchup stats: <100ms for 10K replays
- Full-text search on player codes: <50ms
- Date range filtering: <10ms (indexed)

## Comparison with Other Tools

| Feature | Slippipedia | slippi-stats | slip.py |
|---------|-------------|--------------|---------|
| Platform | Desktop (Qt) | Browser | Browser |
| Storage | SQLite | None | Server |
| Offline | Yes | Yes | No |
| Punish analysis | Detailed | Basic | Basic |
| Batch import | Fast | Manual | Batch |
| Custom queries | SQL | No | No |

## Installation

### Pre-built Binaries
Available for Windows, macOS, Linux from GitHub releases.

### Building from Source
```bash
# Requires Qt 5.15+ with QML
git clone https://github.com/cbartsch/Slippipedia
cd Slippipedia
qmake
make
```

## Use Cases

### For Players
- Track improvement over time
- Identify weak matchups
- Find replays of specific situations
- Compare stats against opponents

### For Analysts
- Study punish game patterns
- Matchup research
- Tournament set analysis

### For Developers
- SQLite database as data source
- Export stats for further analysis
- Reference implementation for punish detection

## Limitations

1. **Desktop Only**: No web/mobile version
2. **Qt Dependency**: Large runtime requirement
3. **Single User**: No cloud sync or sharing
4. **Manual Import**: Must point to replay directories

## ExPhil Relevance

### Data Pipeline
- SQLite schema is a reference for replay indexing
- Punish detection algorithm useful for reward shaping
- Batch import patterns for large replay collections

### Analytics Features
- Opening classification maps to reward signals
- Tech execution metrics for training evaluation
- Matchup filtering for character-specific datasets

## Links

- **Repository**: https://github.com/cbartsch/Slippipedia
- **Releases**: GitHub releases page for binaries
- **Qt Framework**: https://www.qt.io/
