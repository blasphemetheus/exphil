# Peppi Ecosystem Case Study

**Core Repository**: https://github.com/hohav/peppi
**Author**: hohav
**Status**: Active (v2.1.2)
**License**: MIT

## Overview

Peppi is a high-performance Rust parser for Slippi replay files (.slp), with bindings for Python and Julia, plus a CLI tool. It's the fastest parser available and supports a compressed Arrow-based format (.slpp).

## Ecosystem Components

| Project | Language | Purpose |
|---------|----------|---------|
| [peppi](https://github.com/hohav/peppi) | Rust | Core parser library |
| [peppi-py](https://github.com/hohav/peppi-py) | Rust/Python | Python bindings |
| [peppi-slp](https://github.com/hohav/peppi-slp) | Rust | CLI tool |
| [peppi-jl](https://github.com/jph6366/peppi-jl) | Julia | Julia bindings |

## Core Parser (peppi)

### Architecture

```
src/
├── io/
│   ├── slippi/          # .slp format
│   │   ├── de.rs        # Deserialization (1,037 lines)
│   │   └── ser.rs       # Serialization
│   ├── peppi/           # .slpp format
│   └── ubjson/          # UBJSON encoding
├── frame/
│   ├── mutable.rs       # During-parsing (1,309 lines)
│   ├── immutable/       # Arrow conversion
│   └── transpose.rs     # Array-to-struct
└── game/                # Metadata structures
```

### Data Structures

**Game**:
```rust
pub struct Game {
    pub start: Start,
    pub end: Option<End>,
    pub frames: Frame,      // Arrow columnar
    pub metadata: Option<Map<String, Value>>,
    pub gecko_codes: Option<GeckoCodes>,
    pub hash: Option<String>,
}
```

**Frame Data** (Arrow2 struct-of-arrays):
- `id`: Frame counter (starts at -123)
- `ports`: 4 player slots with leader/follower
- `items`: Projectile tracking
- `stage_elements`: FoD, Whispy, Stadium

### Event Types

| Code | Event |
|------|-------|
| 0x35 | Event Payloads |
| 0x36 | GameStart |
| 0x37 | Pre-Frame Update |
| 0x38 | Post-Frame Update |
| 0x39 | GameEnd |
| 0x3A | Frame Start |
| 0x3B | Item Update |
| 0x3C | Frame Bookend |

### Performance

| Operation | Time |
|-----------|------|
| Full parsing | 5-50ms |
| Skip frames | <1ms |
| JSON export | 100-200ms |

## .slp vs .slpp Formats

| Aspect | .slp | .slpp |
|--------|------|-------|
| Format | UBJSON + binary | tar + Arrow IPC |
| Compression | None | LZ4/ZSTD optional |
| Size | Baseline | 45-55% smaller |
| Random access | Sequential only | Full Arrow support |

## Python Bindings (peppi-py)

### Usage

```python
from peppi_py import read_slippi, read_peppi

game = read_slippi('replay.slp')

# Struct-of-arrays access
x_positions = game.frames.ports[0].leader.post.position.x
x_numpy = x_positions.to_numpy()
```

### Zero-Copy Arrow FFI

```rust
// Rust exports Arrow array to C pointers
let array_ptr = export_array_to_c(arr);

// Python imports via PyArrow
pyarrow.Array._import_from_c(array_ptr, schema_ptr)
```

**Benefits**:
- No data duplication
- No serialization overhead
- Full Arrow functionality in Python

## CLI Tool (peppi-slp)

```bash
# Print as JSON
slp game.slp

# Convert to .slpp with compression
slp -f peppi -o game.slpp --compression zstd game.slp

# Skip frame data
slp -s game.slp

# Validate
slp -f null game.slp
```

### Round-Trip Verification

Supports lossless conversion:
```
.slp → .slpp → .slp (bit-identical)
```

## Julia Bindings (peppi-jl)

```julia
using Peppi
game = Peppi.read("match.slp")
x_pos = game.frames.ports[1].leader.pre.position.x[1]
```

**Status**: Early prototype, API may change

## Dependencies

```toml
arrow2 = "0.17"
byteorder = "1"
serde/serde_json
encoding_rs      # Shift-JIS
num_enum
xxhash-rust      # XXH3 hashing
tar              # .slpp archives
```

## Relevance to ExPhil

**Why Peppi matters**:
- Fastest replay parsing for training data
- Arrow format enables efficient batch processing
- Cross-language (can call from Elixir via Rustler or CLI)
- .slpp format saves storage for large replay archives

**Integration options**:
1. Call peppi-slp CLI from Elixir
2. Use peppi-py for Python preprocessing
3. Write Rustler bindings directly to peppi

## References

- [peppi](https://github.com/hohav/peppi)
- [peppi-py](https://github.com/hohav/peppi-py)
- [peppi-slp](https://github.com/hohav/peppi-slp)
- [peppi-jl](https://github.com/jph6366/peppi-jl)
- [Apache Arrow](https://arrow.apache.org/)
