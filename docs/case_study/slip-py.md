# slip.py Case Study

**Repository**: https://github.com/pcrain/slip.py
**Author**: pcrain
**Status**: Active
**License**: GPL-3.0
**Language**: Python (14.7%), GLSL (55.6%)

## Overview

slip.py is a Flask-based Slippi replay browser, search engine, and analyzer. It provides faster replay discovery than the standard browser with indexed search and detailed per-replay statistics.

## Features

### Replay Browser
- Fully indexed replay database
- Visual thumbnails with match info
- Bulk deletion
- Automatic folder scanning

### Search Engine
- Character/costume filtering
- Stage filtering
- Stock count filtering
- Game duration filtering
- Player name/tag search

### Analyzer
- Ledgedash precision calculations
- Move-by-move punish breakdowns
- Color-coded interaction visualization
- Win/loss by matchup
- Frequent/recent opponent tracking

## Architecture

### Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | Flask |
| Templates | Jinja2 |
| ORM | Flask-SQLAlchemy |
| Parsing | slippc (C++) |
| Rendering | GLSL shaders |
| Frontend | CSS, JavaScript |

### Data Flow

```
.slp Files
    ↓
slippc Parser (C++)
    ↓
JSON / Internal Structs
    ↓
Flask Database Ingestion
    ↓
SQLAlchemy Models
    ↓
Flask API / Jinja2 Templates
    ↓
Browser UI
```

## Code Structure

```
slip.py/
├── slipdotpy/
│   ├── slip.py              # Entry point
│   ├── slip-server.py       # Server startup
│   ├── app/
│   │   ├── __init__.py      # Flask app
│   │   ├── config.py        # Configuration
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── helpers.py       # Utilities
│   │   ├── api/             # REST endpoints
│   │   ├── main/            # Blueprints
│   │   └── static/          # CSS, JS
│   ├── playback-dolphin/
│   └── slippc/              # C++ parser
└── pyinstaller.py           # Standalone build
```

## slippc Integration

The C++ slippc library provides:
- High-performance .slp parsing
- JSON export
- 93-97% compression
- Analysis statistics

### Compression Pipeline

```
Raw .slp (UBJSON)
    ↓ Delta coding
    ↓ Predictive coding
    ↓ Event shuffling
    ↓ Column shuffling
    ↓ LZMA compression
Compressed output (93-97% smaller)
```

## Database Schema

**Indexed fields**:
- Character selections
- Player names/tags
- Stage selections
- Match dates
- File paths

**Relationships**:
- Replays ↔ Players
- Replays ↔ Matches
- Players ↔ Statistics

## Deployment

### Options

1. **Desktop app**: PyInstaller executable
2. **Web server**: Flask development/production
3. **Windows**: Dedicated installer script

### Configuration

- Path to Melee 1.02 ISO
- Slippi Playback emulator location
- Replay storage directories

## Limitations

- **1v1 only**: Cannot index FFA/team games
- Designed for tournament-legal singles

## Relevance to ExPhil

**Not directly applicable** - Replay browser, not AI training.

**However**:
- **slippc** could be useful for high-performance parsing
- Database schema patterns for replay metadata
- Search/filter patterns for training data curation
- Statistics computed could inform reward design

## References

- [slip.py](https://github.com/pcrain/slip.py)
- [slippc](https://github.com/pcrain/slippc)
- [Flask](https://flask.palletsprojects.com/)
