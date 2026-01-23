# Slippi Replay Ecosystem

This document provides a comprehensive overview of the Slippi replay ecosystem - the file format, available parsers, data sources, and community infrastructure that enables Melee AI training.

## Why This Matters for AI

Every modern Melee AI project depends on Slippi:
- **slippi-ai**: Trained on ~100K Fizzi anonymized replays
- **Eric Gu's Transformer**: 3 billion frames from pro player replays
- **ExPhil**: Uses peppi for parsing, Fizzi collections for training

Understanding the replay ecosystem is essential for:
1. Obtaining quality training data
2. Parsing replays efficiently
3. Building custom data pipelines
4. Contributing to the community

---

## The .slp File Format

### Overview

The `.slp` format was created by [Fizzi](https://twitter.com/Fizzi36) as part of [Project Slippi](https://slippi.gg). It uses [UBJSON](http://ubjson.org/) (Universal Binary JSON) as the container format.

**Key design goals**:
- Compact binary storage (typical game: 200-500KB)
- Streamable (can write during gameplay)
- Self-describing (metadata embedded)
- Backward compatible (handles unknown events gracefully)

### File Structure

```
[File Signature: 11 bytes]
    └── "{U\x03raw[$U#l" (UBJSON header)
[Size: 4-byte big-endian integer]
    └── Size of raw block
[Payload Sizes Table: 256 entries]
    └── Maps event codes to payload sizes
[Event Stream]
    ├── GameStart (0x36)
    ├── Frame Data (repeating)
    │   ├── Frame Start (0x3A)
    │   ├── Pre-Frame Update (0x37) × players
    │   ├── Post-Frame Update (0x38) × players
    │   ├── Item Update (0x3B) × items
    │   └── Frame Bookend (0x3C)
    └── GameEnd (0x39)
[Metadata: UBJSON object]
    └── Start time, platform, player names
```

### Event Types

| Code | Event | Purpose | Typical Size |
|------|-------|---------|--------------|
| 0x35 | Event Payloads | Size declarations | Variable |
| 0x36 | GameStart | Initialization, characters, stage | 352+ bytes |
| 0x37 | Pre-Frame Update | Inputs, RNG seed | 59+ bytes |
| 0x38 | Post-Frame Update | Position, damage, state | 76+ bytes |
| 0x39 | GameEnd | Method, LRAS initiator | 3+ bytes |
| 0x3A | Frame Start | RNG seed, scene frame | 8 bytes |
| 0x3B | Item Update | Projectile positions | 45+ bytes |
| 0x3C | Frame Bookend | Frame completeness signal | 8 bytes |
| 0x3D | Gecko Codes | Custom code list | Variable |
| 0x3E | Frame Start (v3.10+) | Extended with subframe count | 12 bytes |
| 0x3F | Split Message | Continuation of previous event | Variable |

### Frame Data Content

**Pre-Frame Update** (input reconstruction):
```
- Frame number (int32)
- Player index (uint8)
- Random seed (uint32)
- Joystick X/Y (float32 × 2)
- C-stick X/Y (float32 × 2)
- Trigger (float32)
- Buttons (uint32)
- Physical buttons (uint16)
- Physical L/R (float32 × 2)
```

**Post-Frame Update** (state after physics):
```
- Position X/Y (float32 × 2)
- Facing direction (float32)
- Damage percent (float32)
- Shield size (float32)
- Stock count (uint8)
- Action state (uint16)
- Animation frame (float32)
- State bit flags (uint8)
- Hitstun remaining (float32)
- Ground ID (uint16)
- Jumps remaining (uint8)
- L-cancel status (uint8)
- Hurtbox state (uint8)
- Airborne velocity X/Y (float32 × 2)
- Knockback X/Y (float32 × 2)
- Hitlag remaining (float32)
```

### Metadata Section

Pure UBJSON, parseable separately from frame data:
```json
{
  "startAt": "2024-01-15T18:30:45Z",
  "lastFrame": 8234,
  "playedOn": "network",
  "players": {
    "0": {
      "names": {
        "netplay": "PlayerTag",
        "code": "TAG#123"
      }
    },
    "1": { ... }
  },
  "consoleNick": "DESKTOP-ABC123"
}
```

### In-Progress Replays

Slippi initially sets the raw block length to zero (unknown until game ends). Parsers handle this by:
1. Reading until GameEnd event
2. Or calculating from file size

### Version History

| Version | Release | Key Additions |
|---------|---------|---------------|
| 0.1.0 | 2018 | Initial format |
| 1.0.0 | 2019 | Item tracking |
| 2.0.0 | 2020 | Rollback netcode |
| 3.0.0 | 2021 | Online matchmaking |
| 3.10+ | 2023 | Extended frame data |

**Specification**: [project-slippi/slippi-wiki/SPEC.md](https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md)

---

## Parser Comparison

### Overview Table

| Parser | Language | Speed | Features | Best For |
|--------|----------|-------|----------|----------|
| **[peppi](https://github.com/hohav/peppi)** | Rust | **Fastest** | Arrow output, round-trip | Production, large datasets |
| **[peppi-py](https://github.com/hohav/peppi-py)** | Python | Fast | Zero-copy Arrow | Python ML pipelines |
| **[slippi-js](https://github.com/project-slippi/slippi-js)** | JavaScript | Moderate | Official, stats | Node/browser apps |
| **[py-slippi](https://py-slippi.readthedocs.io/)** | Python | Moderate | Easy API | Quick analysis |
| **[slippc](https://github.com/pcrain/slippc)** | C++ | Fast | Compression | Archiving |

### peppi (Rust)

**The fastest parser**, designed for production use.

**Installation**:
```bash
cargo add peppi
```

**Usage**:
```rust
use peppi::game::Game;
use std::fs::File;
use std::io::BufReader;

let file = File::open("game.slp")?;
let game = peppi::game::read(&mut BufReader::new(file))?;

// Access frame data (Arrow columnar)
let frames = &game.frames;
let p1_x = &frames.ports[0].leader.post.position.x;
```

**Key features**:
- Apache Arrow columnar format (struct-of-arrays)
- Round-trip preservation (parse → write = identical)
- .slpp compressed format support
- 5-50ms parse time for typical replay

### peppi-py (Python Bindings)

Zero-copy Python bindings via PyO3 and Arrow.

**Installation**:
```bash
pip install peppi-py
```

**Usage**:
```python
from peppi_py import read_slippi

game = read_slippi('match.slp')

# Struct-of-arrays access pattern
x_positions = game.frames.ports[0].leader.post.position.x

# Convert to numpy for ML
import numpy as np
x_array = x_positions.to_numpy()

# Access metadata
print(f"Stage: {game.start.stage}")
print(f"Characters: {game.start.players}")
```

**Performance**: ~10ms for typical replay, zero serialization overhead.

### slippi-js (Official)

**The official JavaScript SDK** from Project Slippi.

**Installation**:
```bash
npm install @slippi/slippi-js
```

**Usage (Node.js)**:
```javascript
const { SlippiGame } = require("@slippi/slippi-js/node");

const game = new SlippiGame("match.slp");

// Get metadata
const settings = game.getSettings();
const metadata = game.getMetadata();

// Get computed statistics
const stats = game.getStats();
console.log(stats.overall[0].killCount);

// Get raw frames
const frames = game.getFrames();
```

**Usage (Browser)**:
```javascript
import { SlippiGame } from "@slippi/slippi-js";

const response = await fetch("match.slp");
const buffer = await response.arrayBuffer();
const game = new SlippiGame(buffer);
```

**Live processing**:
```javascript
// For in-progress replays
const game = new SlippiGame("live.slp", { processOnTheFly: true });
```

**Built-in statistics**:
- Overall: Kill count, neutral wins, damage dealt
- Conversions: Combo sequences, damage per opening
- Stocks: Duration, kills, deaths
- Action counts: Wavedash, L-cancel, tech

### py-slippi (Pure Python)

**Easy-to-use Python parser** with both object and event-based APIs.

**Installation**:
```bash
pip install py-slippi
```

**Object-based**:
```python
from slippi import Game

game = Game('match.slp')
print(game.metadata.date)
print(game.start.players[0].character)

# Skip frames for faster metadata-only parsing
game = Game('match.slp', skip_frames=True)
```

**Event-based** (for streaming):
```python
from slippi.parse import parse, ParseEvent

def on_frame(frame):
    print(frame.ports[0].leader.post.position)

handlers = {
    ParseEvent.FRAME: on_frame,
    ParseEvent.METADATA: print
}

parse('match.slp', handlers)
```

**Documentation**: [py-slippi.readthedocs.io](https://py-slippi.readthedocs.io/en/latest/)

### slippc (C++)

**Optimized for compression and archiving**.

**Features**:
- 93-97% compression ratio (.slp → .zlp)
- JSON export
- Batch analysis
- LZMA compression

**Usage**:
```bash
# Compress
slippc -x -i match.slp -o match.zlp

# Export to JSON
slippc -j -i match.slp -o match.json

# Batch analysis
slippc -a -i replays/ > analysis.json
```

**Compression comparison**:
| Method | Size | Speed |
|--------|------|-------|
| Original .slp | 100% | - |
| gzip | ~50% | Fast |
| LZMA (slippc) | ~5% | Slow |
| .slpp (ZSTD) | ~50% | Fast |

---

## Alternative Formats

### .slpp (Peppi Format)

A more efficient alternative to .slp developed by hohav.

**Structure**:
```
archive.slpp (GNU tar)
├── metadata.json      # Game start, end, metadata
├── frames.arrow       # Arrow IPC with all frame data
└── gecko_codes.raw    # Optional gecko code list
```

**Benefits**:
| Aspect | .slp | .slpp |
|--------|------|-------|
| Format | UBJSON + binary | tar + Arrow IPC |
| Compression | None | LZ4/ZSTD optional |
| Size | Baseline | **45-55% smaller** |
| Random access | Sequential only | Full Arrow support |
| Columnar ops | Requires conversion | Native |

**Conversion**:
```bash
# slp → slpp
slp -f peppi -o game.slpp --compression zstd game.slp

# slpp → slp (lossless round-trip)
slp -f slp -o game_restored.slp game.slpp
```

### .zlp (slippc Compressed)

LZMA-compressed .slp for archival:
```bash
# Compress
slippc -x -i match.slp -o match.zlp

# Can still parse directly
slippc -j -i match.zlp -o match.json
```

---

## Data Sources

### 1. Fizzi's Anonymized Ranked Collections (Primary)

**Location**: [Slippi Discord](https://discord.gg/slippi) → #ai-ml channel

The primary training data source, released by Fizzi (Project Slippi creator).

**Characteristics**:
- **Anonymized**: Player tags removed for privacy
- **Ranked only**: Competitive matches, not casual
- **Wide skill range**: Bronze to Grandmaster
- **~100K+ games** available
- **Regular updates** with new data

**How to access**:
1. Join [Slippi Discord](https://discord.gg/slippi)
2. Navigate to #ai-ml or related channel
3. Check pinned messages for download links

**Known download links** (may be outdated):
- Processed replays: `drive.google.com/u/0/uc?id=1O6Njx85-2Te7VAZP6zP51EHa1oIFmS1B`
- Fox dittos only: `drive.google.com/uc?id=1ZIfDgkdQdu-ldCx_34e-VxYJwQCpV-i3`

### 2. Tournament Archives

Major tournaments often release replay archives:

| Tournament | Archive Status | Notes |
|------------|---------------|-------|
| Genesis | Partial | Some years on slippi.gg |
| Big House | Limited | Community archives |
| Pound | Available | Via Slippi Discord |
| EVO | Very limited | Not all setups recorded |

**Finding archives**:
- Check Slippi Discord for `!replaydumps` command
- Search tournament-specific Discord servers
- Contact TOs directly

### 3. slippi.gg Website

The official Slippi website hosts some tournament replay collections:
- [slippi.gg/downloads](https://slippi.gg/downloads) - Launcher and tools
- Tournament replays (selected majors)

### 4. SmashLadder Legacy

Historical replays from pre-rollback era:
- [SmashLadder](https://www.smashladder.com/) hosted early Slippi builds
- Some legacy replay collections may exist

### 5. Personal Collections

Your own replays:
- **Linux/Mac**: `~/.slippi-launcher/replays/`
- **Windows**: `%APPDATA%\Slippi Launcher\replays\`

Useful for character-specific or personal playstyle training.

### 6. ThePlayerDatabase (Metadata Only)

**Repository**: [github.com/smashdata/ThePlayerDatabase](https://github.com/smashdata/ThePlayerDatabase)

SQLite database of tournament metadata:
- 96,689 players
- 39,675 tournaments
- 1,795,681 sets

**Note**: Does NOT contain replay files, only tournament results and player statistics.

---

## Data Quality Considerations

### Skill Distribution

Not all replays are created equal:

| Rank | % of Data | Training Value |
|------|-----------|----------------|
| Grandmaster | ~0.1% | Highest |
| Master | ~2% | High |
| Diamond | ~10% | Good |
| Platinum | ~30% | Moderate |
| Gold/Silver | ~40% | Low |
| Bronze | ~18% | Very low |

**Recommendation**: Filter to Diamond+ for quality training.

### Character Imbalance

Tournament character distribution:
| Character | Usage |
|-----------|-------|
| Fox | 29% |
| Sheik/Falco/Marth | 10-13% each |
| Captain Falcon | 7% |
| Low-tier total | < 5% |

**For ExPhil targets** (Mewtwo, Ganondorf, Link, G&W, Zelda): Data is extremely scarce. Expect 1-2% of total replays at best.

### Quality Indicators

Heuristics for data quality:
```python
def estimate_quality(replay):
    score = 0

    # L-cancel rate (major indicator)
    if replay.l_cancel_rate > 0.8:
        score += 3
    elif replay.l_cancel_rate > 0.5:
        score += 1

    # Wavedash frequency
    if replay.wavedashes_per_minute > 2:
        score += 2

    # Action diversity
    if len(replay.unique_actions) > 50:
        score += 1

    # Game length (avoid stomps/timeouts)
    if 3000 < replay.frame_count < 8000:
        score += 1

    return score
```

---

## Community Tools

### Analysis & Visualization

| Tool | Type | Purpose | Link |
|------|------|---------|------|
| **Slippipedia** | Desktop | SQLite replay manager | [GitHub](https://github.com/cbartsch/Slippipedia) |
| **slip.py** | Web | Flask search engine | [GitHub](https://github.com/pcrain/slip.py) |
| **slippi-stats** | Browser | Quick stats viewer | [GitHub](https://github.com/vinceau/slippi-stats) |
| **SlippiLab** | Web | Browser replay viewer | [GitHub](https://github.com/frankborden/slippilab) |
| **Chart.slp** | Web | Global statistics | [chartslp.com](https://chartslp.com/global) |

### Conversion & Processing

| Tool | Purpose |
|------|---------|
| **peppi-slp** | CLI for format conversion |
| **slp-to-mp4** | Replay to video |
| **slippi-set-stats** | Tournament broadcast stats |

### Discord Bots

| Bot | Function |
|-----|----------|
| **SlippiStatsBot** | Upload .slp, get stats |

---

## Pipeline Architecture

### Recommended Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Acquisition                         │
├─────────────────────────────────────────────────────────────┤
│  Slippi Discord     Tournament Archives    Personal Replays │
│       │                    │                     │          │
│       └────────────────────┼─────────────────────┘          │
│                            ▼                                 │
│                    Raw .slp Files                           │
│                    (100K+ games)                            │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      Filtering Stage                         │
├─────────────────────────────────────────────────────────────┤
│  Character Filter  →  Skill Filter  →  Quality Filter       │
│  (target chars)       (Diamond+)       (L-cancel, length)   │
│                                                              │
│  Tools: cloud_filter_replays.py, Slippipedia                │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      Parsing Stage                           │
├─────────────────────────────────────────────────────────────┤
│  peppi-py / peppi-slp                                       │
│       │                                                      │
│       ▼                                                      │
│  Arrow/Parquet format (columnar, ML-ready)                  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    Augmentation Stage                        │
├─────────────────────────────────────────────────────────────┤
│  Mirror augmentation (2x data)                              │
│  Noise injection (robustness)                               │
│  Frame delay simulation (online play)                       │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      Training Stage                          │
├─────────────────────────────────────────────────────────────┤
│  Behavioral Cloning → PPO Self-Play → Evaluation            │
│                                                              │
│  ExPhil: lib/exphil/training/                               │
└─────────────────────────────────────────────────────────────┘
```

### Storage Recommendations

```
replays/
├── raw/                    # Original .slp files
│   ├── fizzi_ranked_2024/
│   ├── tournament_genesis/
│   └── personal/
├── filtered/               # Character-filtered
│   ├── mewtwo/
│   ├── ganondorf/
│   └── link/
├── parsed/                 # Arrow/Parquet format
│   ├── mewtwo.parquet
│   └── all_low_tier.parquet
└── metadata/
    └── dataset_stats.json
```

### Cloud Storage Options

| Provider | Storage | Egress | Best For |
|----------|---------|--------|----------|
| Backblaze B2 | $0.005/GB | $0.01/GB | Simple, cheap |
| Cloudflare R2 | $0.015/GB | **Free** | Frequent pulls |
| Google Drive | Free (15GB) | Free | Small datasets |

See [REPLAY_STORAGE.md](../REPLAY_STORAGE.md) for detailed setup.

---

## ExPhil Integration

### Current Approach

ExPhil uses peppi for parsing via Python preprocessing:

```elixir
# lib/exphil/bridge/replay_loader.ex
defmodule ExPhil.Bridge.ReplayLoader do
  @doc """
  Load replays from .slp files using peppi-py.
  """
  def load(path) do
    # Call Python via Port
    Python.call("load_replay", [path])
  end
end
```

### Parsing Script

```python
# priv/python/parse_replays.py
from peppi_py import read_slippi
import numpy as np

def parse_for_training(slp_path):
    game = read_slippi(slp_path)

    # Extract frame data as numpy arrays
    frames = []
    for frame_idx in range(len(game.frames.id)):
        frame_data = extract_frame(game.frames, frame_idx)
        frames.append(frame_data)

    return np.stack(frames)
```

### Integration Options

1. **Python preprocessing** (current): Parse with peppi-py, save to Parquet
2. **Rustler bindings**: Call peppi directly from Elixir
3. **CLI pipeline**: peppi-slp → Arrow files → Elixir reads

---

## Best Practices

### For Data Collection

1. **Start with Fizzi's collections** - Best quality/quantity ratio
2. **Filter early** - Don't parse what you won't use
3. **Track statistics** - Know your dataset composition
4. **Version your datasets** - Reproducibility matters

### For Parsing

1. **Use peppi for production** - Fastest, most robust
2. **Use py-slippi for exploration** - Easy API
3. **Batch processing** - Parse in bulk, not per-frame
4. **Cache parsed results** - Don't re-parse repeatedly

### For AI Training

1. **Balance characters** - Upsample rare characters
2. **Filter by skill** - Diamond+ for quality
3. **Mirror augment** - Standard for all training
4. **Validate splits** - Don't leak test data

---

## Troubleshooting

### Common Issues

**"Invalid .slp file"**
- Check file isn't corrupted (incomplete download)
- Verify Slippi version compatibility
- Try different parser

**"Out of memory when parsing"**
- Use streaming API (`skip_frames=True` then process incrementally)
- Parse in batches
- Use .slpp for compressed storage

**"Frame data mismatch"**
- Check for Ice Climbers (2 characters per port)
- Handle variable player count (1v1 vs 2v2)
- Verify frame indexing (-123 start)

### Parser-Specific

**peppi**: Check Rust toolchain version
**py-slippi**: Requires Python 3.7+
**slippi-js**: May need Node 16+

---

## References

### Official Resources
- [Slippi.gg](https://slippi.gg/) - Official website
- [Slippi Discord](https://discord.gg/slippi) - Community hub
- [slippi-wiki/SPEC.md](https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md) - Format specification
- [slippi-js](https://github.com/project-slippi/slippi-js) - Official SDK

### Parser Documentation
- [peppi](https://github.com/hohav/peppi) - Rust parser
- [peppi-py](https://github.com/hohav/peppi-py) - Python bindings
- [py-slippi docs](https://py-slippi.readthedocs.io/) - Pure Python
- [slippc](https://github.com/pcrain/slippc) - C++ with compression

### Related Case Studies
- [project-slippi.md](project-slippi.md) - Slippi infrastructure
- [peppi-ecosystem.md](peppi-ecosystem.md) - Peppi parser details
- [dataset-curation.md](dataset-curation.md) - Data preparation

### AI Training References
- [slippi-ai](https://github.com/vladfi1/slippi-ai) - Primary reference implementation
- [Eric Gu's Transformer](https://ericyuegu.com/melee-pt1) - 20M param GPT-style
