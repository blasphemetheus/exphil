# slip.py

**Repository:** https://github.com/pcrain/slip.py
**Author:** pcrain
**Language:** Python (Flask), JavaScript, C++ (slippc)
**Status:** Active
**Purpose:** Web-based replay browser, search engine, and analyzer

## Overview

slip.py is a production-grade web application for indexing, searching, and analyzing Slippi replay files. Built on Flask with a C++ backend parser (slippc), it provides comprehensive replay management with advanced search capabilities and detailed frame-by-frame analysis.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Web Framework | Flask (Python) |
| Database | SQLite + Flask-SQLAlchemy |
| Parser | slippc (C++ subprocess) |
| Templates | Jinja2 (25 templates) |
| Desktop Wrapper | PyFladesk (optional) |
| Compression | LZMA (93-97% ratios) |
| Background Tasks | Flask-Executor |

## Architecture

```
Slippi Replay (.slp)
        ↓
[slippc C++ Parser]
    ├→ UBJSON deserialization
    ├→ Frame reorganization
    └→ Internal structures
        ↓
[slippc Analyzer]
    ├→ Interaction classification
    ├→ Punish detection
    └→ JSON export
        ↓
[slip.py Flask App]
    ├→ Database insertion
    ├→ Search indexing
    └→ Web UI
```

## Key Features

### Search Engine
- Filter by character, costume/color, stage
- Filter by stock count, game duration
- Keyword search across player names and filenames
- Complex query combinations
- Paginated results (60 per page default)

### Per-Player Statistics
- Character-based win/loss records
- Most frequent opponents
- Performance metrics per matchup
- Game count and win percentage

### Frame-by-Frame Analysis
- Complete frame breakdown
- Move identification and damage calculation
- Punish (combo) detection with frame ranges
- Interaction classification:
  - Neutral (positioning, footsies)
  - Offensive (pressure, combos)
  - Defensive (shielding, teching)
  - Edge-guarding and recovery

### Technical Metrics
- Ledgedash success rates and angles
- Short hop vs full hop rates
- Defense efficiency during pressure
- Neutral game control metrics
- Input accuracy and speed

### Replay Management
- Quarantine system for unwanted replays
- Batch operations for cleanup
- Public/private visibility settings
- Multi-threaded scanning

## Database Schema

```sql
Replay:
  - checksum, filename, filedir, filesize
  - played, uploaded (timestamps)
  - frames, stage, winner, timer
  - p1char, p1color, p1stocks, p1display
  - p2char, p2color, p2stocks, p2display
  - p1punish, p1defense, p1neutral, p1speed
  - ver_slippi, ver_stats, ver_viz
  - is_public
```

## slippc Parser

The C++ backend parser (slippc) handles:
- Replay versions 0.1.0 through 3.12.0
- UBJSON to internal structure conversion
- Big-endian to little-endian conversion
- Compressed replay support (.zlp format)

### Compression Performance
- LZMA compression: 93-97% size reduction
- Enables efficient large-scale storage

## Interaction Classification

slippc automatically classifies all player interactions:

| Type | Description |
|------|-------------|
| Neutral | Positioning, footsies, pokes |
| Offensive | Pressure, punishes, combos |
| Defensive | Shielding, teching, escaping |
| Edgeguard | Off-stage offense |
| Recovery | Off-stage defense |

## Relevance to AI Training

### Pre-computed Features
- Interaction labels provide supervision signals
- Move identification maps to action prediction
- Damage accounting enables reward signal design

### Data Organization
- Searchable database enables filtered dataset creation
- Character/stage/skill filtering for curriculum learning
- Compressed storage for large-scale datasets

### Analysis Pipeline Parallels

| slip.py | ExPhil Equivalent |
|---------|-------------------|
| Game state parsing | Embedding layer |
| Interaction classification | Auxiliary prediction task |
| Move identification | Controller action targets |
| Damage tracking | Reward signal |

### Limitations
- Designed for 1v1 singles (ICs partially supported)
- Assumes tournament-legal rulesets
- Some damage sources imperfectly handled

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /raw/<r>` | Raw analysis JSON export |
| `POST /scan/progress` | Real-time scan status |
| Search endpoints | Filtered replay queries |

## Key Insight

slip.py demonstrates a complete production architecture for replay analysis at scale. The separation of C++ parsing (slippc) from Python web serving shows how to build efficient pipelines that could be adapted for training data preparation.

## Links

- [slip.py GitHub](https://github.com/pcrain/slip.py)
- [slippc Parser](https://github.com/pcrain/slippc)
- [Project Slippi](https://github.com/project-slippi)
