# Melee State Representation

This document covers how to represent Melee game state for machine learning, including embedding strategies, dimensionality considerations, and practical implementation patterns.

## Why Representation Matters

The quality of state embedding directly impacts:
1. **Learning efficiency**: Better representation = faster convergence
2. **Generalization**: Appropriate abstraction enables transfer
3. **Model capacity**: More dimensions = more parameters needed
4. **Inference speed**: Larger embeddings = slower real-time play

## Game State Components

### Complete State Hierarchy

```
GameState
├── Frame number
├── Stage
│   ├── Stage ID
│   ├── Platform positions (dynamic stages)
│   ├── Stage hazards (Randall, etc.)
│   └── Blast zones
├── Players [0..1]
│   ├── Position (x, y)
│   ├── Velocity (5 components)
│   ├── Action state
│   ├── Action frame
│   ├── Character
│   ├── Damage %
│   ├── Stocks
│   ├── Shield strength
│   ├── Facing direction
│   ├── On ground
│   ├── Invulnerability
│   ├── Jumps remaining
│   ├── Hitstun frames
│   ├── Hitlag frames
│   └── ECB corners
├── Projectiles [0..N]
│   ├── Type
│   ├── Position
│   ├── Velocity
│   └── Owner
└── Items [0..N]
    ├── Type
    ├── Position
    └── State
```

## Embedding Strategies

### 1. Float Embedding

**For continuous values** (position, velocity, damage):

```python
class FloatEmbedding:
    def __init__(self, scale=1.0, clip_range=None):
        self.scale = scale
        self.clip_range = clip_range

    def embed(self, value):
        scaled = value * self.scale
        if self.clip_range:
            scaled = clip(scaled, *self.clip_range)
        return scaled
```

**Typical scales**:
| Feature | Scale | Reason |
|---------|-------|--------|
| Position | 0.05-0.1 | Stage is ~200 units wide |
| Velocity | 0.5 | Velocities typically < 5 |
| Damage | 0.01 | Damage ranges 0-999 |
| Shield | 0.01 | Shield is 0-60 |

### 2. One-Hot Embedding

**For categorical values** (action state, character, stage):

```python
class OneHotEmbedding:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def embed(self, value):
        one_hot = zeros(self.num_classes)
        one_hot[value] = 1.0
        return one_hot
```

**Sizes**:
| Feature | Classes | Notes |
|---------|---------|-------|
| Action | 399 | 0x18F max action ID |
| Character | 33 | 26 + variants |
| Stage | 64 | All stages |
| Jumps | 7 | 0-6 jumps |

### 3. Learned Embedding

**Alternative to one-hot** for high-cardinality categoricals:

```python
class LearnedEmbedding:
    def __init__(self, num_classes, embed_dim):
        self.embedding_table = Parameter(
            shape=(num_classes, embed_dim),
            initializer='normal'
        )

    def embed(self, value):
        return self.embedding_table[value]
```

**Benefits**:
- Much smaller: 64 dims vs 399 for actions
- Learns semantic relationships (similar actions → similar embeddings)
- Reduces total parameter count

**ExPhil action_mode:**
- `:one_hot` → 399 dims (default)
- `:learned` → 64 dims (saves ~670 dims total)

### 4. Boolean Embedding

**For binary flags**:

```python
class BoolEmbedding:
    def embed(self, value):
        return 1.0 if value else 0.0
        # Or: return 1.0 if value else -1.0 (centered)
```

**Boolean features**:
- On ground
- Facing right
- Invulnerable
- Charging smash

## Player Embedding (Detailed)

### slippi-ai Style (484 dims)

```
Position:          2 dims (x, y scaled)
Velocity:          5 dims (air_x, ground_x, y, attack_x, attack_y)
Facing:            1 dim (boolean)
On ground:         1 dim (boolean)
Invulnerable:      1 dim (boolean)
Shield:            1 dim (scaled)
Damage:            1 dim (scaled)
Jumps:             7 dims (one-hot)
Action:          399 dims (one-hot)
Character:        33 dims (one-hot)
Action frame:      1 dim (scaled)
Hitstun:           1 dim (scaled)
Hitlag:            1 dim (scaled)
─────────────────────────────────
Total:           ~454 dims base
With Nana:       +39 dims = ~493 dims
```

### ExPhil Configurations

**Full one-hot (default)**:
```
Player:           488 dims
Game (2 players): 1204 dims total
```

**With learned actions**:
```
Player:           91 dims base + action ID
Game:             408 dims + 2 action IDs
Action embedding: 64 dims (trained with network)
```

**512-dim target (compact)**:
```
Continuous:       254 dims
Action IDs:       4 × 64 embedding = 256 dims
Total:            510 dims
```

### Nana (Ice Climbers) Handling

**Modes**:

| Mode | Dims | Contents |
|------|------|----------|
| `:compact` | 39 | Essential state only |
| `:enhanced` | 14 + ID | Minimal + learned action |
| `:full` | 449 | Complete player state |

**Why Nana matters**:
- Ice Climbers tech requires tracking Nana state
- Desyncs, handoffs, regrabs depend on Nana position/action
- Can't just ignore partner for competitive play

## Game Embedding (Full State)

### Components

```
┌──────────────────────────────────────┐
│           Game Embedding             │
├──────────────────────────────────────┤
│ Player 0 embedding     : 484 dims    │
│ Player 1 embedding     : 484 dims    │
│ Stage (one-hot)        : 64 dims     │
│ Platform heights (FoD) : 2 dims      │
│ Randall position       : 2 dims      │
│ Projectiles (15 × MLP) : 480 dims    │
├──────────────────────────────────────┤
│ Total                  : ~1516 dims  │
└──────────────────────────────────────┘
```

### Spatial Features

**Platform positions** (Fountain of Dreams):
```python
left_platform_height = FloatEmbedding(scale=0.05)
right_platform_height = FloatEmbedding(scale=0.05)
```

**Randall** (Yoshi's Story):
```python
randall_x = FloatEmbedding(scale=0.05)
randall_y = FloatEmbedding(scale=0.05)
```

### Projectile Embedding

**Challenge**: Variable number of projectiles (0-15)

**Solution 1: Fixed slots with exists flag**
```python
for i in range(15):
    if i < len(projectiles):
        embed_projectile(projectiles[i])
    else:
        embed_null_projectile()  # All zeros
```

**Solution 2: MLP compression**
```python
# Raw per-projectile: ~252 dims
raw_embed = concat(exists, type_onehot, state, x, y)

# MLP: 252 → 128 → 32
compressed = MLP([128, 32])(raw_embed)

# 15 projectiles × 32 = 480 dims
```

**Solution 3: Set/attention encoding**
```python
# Embed each projectile
projectile_embeds = [embed(p) for p in projectiles]

# Self-attention over projectiles
attended = SelfAttention(projectile_embeds)

# Pool to fixed size
pooled = mean(attended)  # Or max, or learned query
```

## Temporal Considerations

### Frame Stacking

**Simple approach**: Concatenate N previous frames

```python
state_t = embed(frame_t)
state_t1 = embed(frame_t-1)
state_t2 = embed(frame_t-2)

input = concat(state_t, state_t1, state_t2)
# 3 × 1200 = 3600 dims
```

**Problem**: Dimensionality explodes, redundant information

### Recurrent Embedding

**Better approach**: Let RNN/Mamba handle temporal

```python
# Single-frame embedding
state_t = embed(frame_t)  # 1200 dims

# Network processes sequence
hidden = network.step(state_t, prev_hidden)
```

**Benefits**:
- Fixed input dimension regardless of history
- Network learns what's important to remember
- Handles variable-length sequences

### Action History

**Include previous actions** for context:

```python
state = concat(
    game_embed,           # Current game state
    prev_controller,      # What we did last frame
    prev_prev_controller  # Two frames ago (optional)
)
```

**Why**:
- Techniques span multiple frames (wavedash)
- Self-consistency (don't rapidly switch)
- Reaction time modeling

## Normalization

### Per-Feature Normalization

```python
# Standardize continuous features
x_normalized = (x - mean_x) / std_x

# Computed from replay dataset statistics
POSITION_MEAN = 0.0
POSITION_STD = 50.0  # Approximate stage half-width
```

### Layer Normalization

```python
# Applied to full embedding
embedding = LayerNorm()(raw_embedding)
```

**When to use**:
- After concatenating heterogeneous features
- Before feeding to transformer/attention
- Generally good practice

## Character-Specific Considerations

### Universal Features

These apply to all characters:
- Position, velocity
- Damage, stocks, shield
- Action state (though actions differ)
- On ground, facing

### Character-Specific Features

**Ice Climbers**: Need Nana state
**Peach**: Turnip type, float state
**Zelda/Sheik**: Transform state
**Game & Watch**: Bucket count, hammer RNG

**Handling**:
1. **Universal model**: Include all features, zeros for irrelevant
2. **Character-specific heads**: Different embedding per character
3. **Hybrid**: Universal core + character-specific additions

### ExPhil Target Characters

| Character | Special Considerations |
|-----------|----------------------|
| Mewtwo | Confusion state, disable, teleport angle |
| Ganondorf | Standard (simple moveset) |
| Link | Projectile state (boomerang, bomb, arrow) |
| G&W | Bucket count, judgment outcome |
| Zelda | Transform availability |
| Ice Climbers | Full Nana state required |

## Implementation Patterns

### Struct-Based Embedding (Elixir)

```elixir
defmodule ExPhil.Embeddings.Player do
  def embed(player, config) do
    [
      position_embed(player.x, player.y, config.xy_scale),
      velocity_embed(player, config.speed_scale),
      action_embed(player.action_state, config.action_mode),
      character_embed(player.character),
      flags_embed(player)
    ]
    |> Nx.concatenate()
  end
end
```

### Configurable Embedding

```elixir
config = %{
  action_mode: :learned,     # :one_hot or :learned
  nana_mode: :compact,       # :compact, :enhanced, :full
  jumps_normalized: true,    # single dim or one-hot
  with_speeds: true,
  xy_scale: 0.05,
  speed_scale: 0.5
}

embedding = ExPhil.Embeddings.Game.embed(game_state, config)
```

### Caching

**For training efficiency**: Pre-compute embeddings

```elixir
# Parse replay → cache embeddings
replay
|> parse_frames()
|> Enum.map(&embed_frame/1)
|> save_to_cache(replay_id)

# Training loads from cache
cached_embeddings = load_cache(replay_id)
```

**ExPhil speedup**: 2-3× from embedding caching

## Debugging Embeddings

### Sanity Checks

```elixir
# Check dimensions
assert Nx.shape(embedding) == {expected_dims}

# Check ranges
assert Nx.all(embedding >= -10)
assert Nx.all(embedding <= 10)

# Check no NaN
assert not Nx.any(Nx.is_nan(embedding))
```

### Visualization

```python
# t-SNE of action embeddings
from sklearn.manifold import TSNE

action_embeds = model.action_embedding.weight.numpy()
tsne = TSNE(n_components=2)
projected = tsne.fit_transform(action_embeds)

# Similar actions should cluster
plot_with_labels(projected, ACTION_NAMES)
```

## References

- [slippi-ai embed.py](https://github.com/vladfi1/slippi-ai) - Reference implementation
- [libmelee GameState](https://libmelee.readthedocs.io/) - State structure
- [ExPhil embeddings/](../../lib/exphil/embeddings/) - Elixir implementation
