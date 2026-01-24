# slip.py

**Repository:** https://github.com/pcrain/slip.py
**Author:** pcrain
**Language:** Python (Flask), C++ (slippc parser)
**Status:** Active
**Purpose:** Web-based replay browser and search engine with server-side parsing

## Overview

slip.py is a Flask-based web application for browsing and searching Slippi replay collections. It features a fast C++ parser (slippc) for server-side processing and GLSL shaders for visualization. The project focuses on being a searchable replay database with rich metadata extraction.

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────┐
│                   Web Browser                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Search    │  │   Replay    │  │   Stats     │ │
│  │   Interface │  │   Viewer    │  │   Display   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────┬───────────────────────────┘
                          │ HTTP
┌─────────────────────────┴───────────────────────────┐
│                   Flask Server                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Routes    │  │   Search    │  │   Parser    │ │
│  │   API       │  │   Engine    │  │   Wrapper   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────┐
│                    slippc (C++)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   .slp      │  │   Frame     │  │   Stats     │ │
│  │   Parser    │  │   Extractor │  │   Computer  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Key Components

1. **Flask Web Server**: Routes, templates, API endpoints
2. **slippc Parser**: High-performance C++ replay parsing
3. **Search Engine**: Query replays by various criteria
4. **GLSL Visualizations**: Shader-based replay rendering

## slippc Parser

The C++ parser is the performance-critical component:

### Features
- Parses .slp files 10-100x faster than pure Python
- Extracts all frame data and metadata
- Computes derived statistics
- Outputs JSON for Python consumption

### Usage
```bash
# Parse single replay
./slippc replay.slp --json

# Parse with stats
./slippc replay.slp --stats --json

# Batch processing
find replays/ -name "*.slp" | xargs -P4 -I{} ./slippc {} --json
```

### Output Format
```json
{
  "metadata": {
    "startAt": "2024-01-15T10:30:00Z",
    "playedOn": "dolphin",
    "lastFrame": 7200
  },
  "players": [
    {
      "port": 0,
      "character": 2,
      "connectCode": "TEST#123",
      "stocks": 2,
      "damage": 142.5
    }
  ],
  "stats": {
    "lcancelRate": 0.85,
    "apm": 320,
    "openingsPerKill": 4.2
  }
}
```

## Web Interface

### Search Capabilities

```
Search Parameters:
├── Player
│   ├── Connect code (exact/partial)
│   ├── Character
│   └── Tag/display name
├── Match
│   ├── Stage
│   ├── Date range
│   ├── Duration (min/max frames)
│   └── Game mode
├── Outcome
│   ├── Winner
│   ├── Stock count
│   └── Timeout
└── Technical
    ├── L-cancel rate
    ├── APM range
    └── Damage dealt
```

### API Endpoints

```python
# Flask routes (simplified)
@app.route('/api/search')
def search_replays():
    """Search replays with filters"""
    pass

@app.route('/api/replay/<replay_id>')
def get_replay(replay_id):
    """Get full replay data"""
    pass

@app.route('/api/stats/<player_code>')
def get_player_stats(player_code):
    """Aggregate stats for player"""
    pass

@app.route('/api/parse', methods=['POST'])
def parse_replay():
    """Upload and parse new replay"""
    pass
```

## GLSL Visualizations

slip.py includes shader-based visualizations:

### Stage Rendering
- 2D stage layouts with platforms
- Player position heatmaps
- Movement trails

### Input Display
- Controller state over time
- Stick position plots
- Button press timelines

### Combat Analysis
- Hit/hurtbox visualization
- Combo flowcharts
- Damage accumulation graphs

## Database Design

```sql
-- Replay metadata
CREATE TABLE replays (
    id TEXT PRIMARY KEY,  -- Hash of file
    path TEXT,
    parsed_at TIMESTAMP,
    duration_frames INTEGER,
    stage INTEGER,
    winner_port INTEGER
);

-- Player data per replay
CREATE TABLE replay_players (
    replay_id TEXT REFERENCES replays(id),
    port INTEGER,
    character INTEGER,
    connect_code TEXT,
    tag TEXT,
    stocks_remaining INTEGER,
    damage_dealt REAL,
    PRIMARY KEY (replay_id, port)
);

-- Computed statistics
CREATE TABLE replay_stats (
    replay_id TEXT REFERENCES replays(id),
    player_port INTEGER,
    lcancel_rate REAL,
    apm REAL,
    openings INTEGER,
    conversions INTEGER,
    PRIMARY KEY (replay_id, player_port)
);

-- Full-text search index
CREATE VIRTUAL TABLE replay_search USING fts5(
    connect_code, tag, content='replay_players'
);
```

## Installation

### Requirements
- Python 3.8+
- Flask
- C++ compiler (for slippc)
- SQLite

### Setup
```bash
git clone https://github.com/pcrain/slip.py
cd slip.py

# Build C++ parser
cd slippc
make
cd ..

# Install Python dependencies
pip install -r requirements.txt

# Initialize database
python init_db.py

# Run server
flask run
```

## Performance

### Parser Benchmarks
| Method | Time per Replay | Notes |
|--------|-----------------|-------|
| slippc (C++) | ~5-10ms | Metadata only |
| slippc (C++) | ~50-100ms | Full frame data |
| py-slippi | ~200-500ms | Pure Python |

### Server Performance
- Search queries: <100ms for 10K replays
- Replay detail: <50ms (cached)
- Batch import: ~100 replays/second

## Comparison with Alternatives

| Feature | slip.py | slippi-stats | Slippipedia |
|---------|---------|--------------|-------------|
| Platform | Web | Browser | Desktop |
| Parser | C++ (fast) | JS | C++ |
| Search | Full-text | Basic | SQL |
| Multi-user | Yes | No | No |
| Visualizations | GLSL | Basic | None |
| Self-hosted | Yes | N/A | N/A |

## Limitations

1. **Server Required**: Not a standalone client
2. **Setup Complexity**: C++ compilation needed
3. **Storage**: Requires disk space for database
4. **No Real-time**: Post-hoc analysis only

## ExPhil Relevance

### C++ Parser Integration
- slippc could be wrapped for faster Elixir parsing via NIF
- JSON output format compatible with Elixir's Jason
- Batch processing patterns for large datasets

### Search Architecture
- Full-text search patterns for replay filtering
- Metadata indexing strategies
- Player code lookup for dataset curation

### Visualization Ideas
- GLSL patterns could inform training visualizations
- Heatmap generation for position analysis
- Movement trail analysis for character-specific patterns

## Links

- **Repository**: https://github.com/pcrain/slip.py
- **slippc Parser**: Included in repository
- **Flask**: https://flask.palletsprojects.com/
