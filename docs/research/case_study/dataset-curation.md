# Dataset Curation for Melee AI

This document covers best practices for obtaining, filtering, and preparing Slippi replay data for training Melee AI agents.

## Why Dataset Quality Matters

Training data quality directly impacts model quality:
- **Garbage in, garbage out** - Bad habits in data become bad habits in the model
- **Class imbalance** - 29% Fox vs 2% low-tier characters creates bias
- **Skill distribution** - Mixing Bronze and Grandmaster creates contradictory signals
- **Data volume** - More high-quality data generally beats less perfect data

---

## Data Sources

### 1. Fizzi's Anonymized Ranked Collections (Primary)

**Location**: [Slippi Discord](https://slippi.gg/) → AI/ML channel

Fizzi (creator of Slippi) has released large anonymized replay collections from ranked play. These are the primary training source for [slippi-ai](https://github.com/vladfi1/slippi-ai).

**Characteristics**:
- Anonymized player tags (privacy-preserving)
- Ranked matches only (competitive intent)
- Wide skill distribution (Bronze to Grandmaster)
- ~100K+ games available
- Regular updates

**How to access**:
1. Join [Slippi Discord](https://discord.gg/slippi)
2. Navigate to #ai-ml or similar channel
3. Find pinned download links

### 2. ThePlayerDatabase (Tournament Metadata)

**Repository**: [github.com/smashdata/ThePlayerDatabase](https://github.com/smashdata/ThePlayerDatabase)
**Downloads**: [Releases page](https://github.com/smashdata/ThePlayerDatabase/releases)

SQLite database of tournament sets from smashdata.gg:
- 96,689 players
- 39,675 tournaments
- 1,795,681 sets (2015-2024)

**Use cases**:
- Player skill level estimation
- Matchup statistics
- Historical trend analysis
- Does NOT contain replay files, only metadata

### 3. Tournament Archives

Various community members archive tournament replays:
- Major tournaments (Genesis, EVO, Big House)
- Regional weeklies
- Pro player streams

**Finding archives**:
- Search Slippi Discord
- Check tournament Twitch VODs
- Ask directly in community channels

### 4. Personal Replay Collections

Your own Slippi replays from:
- Local: `~/.slippi-launcher/replays/` (Linux/Mac)
- Windows: `%APPDATA%\Slippi Launcher\replays\`

Useful for specific character/matchup focus.

---

## Skill Level Filtering

### The Problem

Mixing skill levels creates contradictory training signal:
```
Bronze player: Spams smash attacks, never L-cancels
Grandmaster: Frame-perfect tech, optimal punishes
→ Model learns confused average behavior
```

### Slippi Rank Tiers

| Rank | Rating Range | Description |
|------|-------------|-------------|
| Grandmaster | 2350+ | Top ~0.1% |
| Master 3 | 2200-2350 | Top ~1% |
| Master 2 | 2000-2200 | Very strong |
| Master 1 | 1800-2000 | Strong |
| Diamond 3 | 1700-1800 | Advanced |
| Diamond 2 | 1600-1700 | Good |
| Diamond 1 | 1500-1600 | Above average |
| Platinum | 1200-1500 | Average |
| Gold | 900-1200 | Below average |
| Silver | 600-900 | Beginner |
| Bronze | 0-600 | New player |

### Filtering Strategies

#### Strategy 1: High-Skill Only

Train only on Master+ replays:
```python
def filter_by_rank(replay):
    """Keep only high-skill matches"""
    p1_rating = get_player_rating(replay.player1)
    p2_rating = get_player_rating(replay.player2)
    min_rating = min(p1_rating, p2_rating)
    return min_rating >= 1800  # Master 1+
```

**Pros**: Clean signal, optimal behavior
**Cons**: Much smaller dataset, less diversity

#### Strategy 2: Weighted Sampling

Include all ranks but weight by skill:
```python
def get_sample_weight(replay):
    """Higher weight for higher skill"""
    avg_rating = (replay.p1_rating + replay.p2_rating) / 2
    if avg_rating >= 2000:
        return 3.0  # Master 2+
    elif avg_rating >= 1500:
        return 1.5  # Diamond+
    else:
        return 0.5  # Platinum and below
```

**Pros**: Uses all data, still prioritizes quality
**Cons**: Some noise remains

#### Strategy 3: Curriculum Learning

Start with all data, progressively filter:
```python
def get_min_rating(epoch, total_epochs):
    """Progressively increase minimum rating"""
    progress = epoch / total_epochs
    # Start at 1000, end at 1800
    return 1000 + int(progress * 800)
```

**Pros**: Maximum early data, refined later
**Cons**: More complex pipeline

### Rating Inference

Anonymized replays don't include ratings. Options:

1. **Game length heuristic**: Longer games suggest closer skill
2. **Technique detection**: L-cancel rate, wavedash frequency
3. **Action state variety**: Better players use more actions
4. **Damage efficiency**: High-skill = more damage per opening

```python
def estimate_skill(replay):
    """Heuristic skill estimation"""
    stats = compute_stats(replay)

    score = 0

    # L-cancel rate (major indicator)
    if stats.l_cancel_rate > 0.8:
        score += 3
    elif stats.l_cancel_rate > 0.5:
        score += 1

    # Wavedash frequency
    if stats.wavedashes_per_minute > 2:
        score += 2

    # Action diversity
    if stats.unique_actions > 50:
        score += 1

    # Damage per opening
    if stats.damage_per_opening > 30:
        score += 2

    return score  # 0-8 scale
```

---

## Character/Matchup Balance

### The Character Imbalance Problem

Tournament character distribution ([Liquipedia 2024](https://liquipedia.net/smash/Portal:Statistics/Melee/2024)):

| Character | Usage | Training Issue |
|-----------|-------|----------------|
| Fox | 29.2% | Overrepresented |
| Sheik | 13% | Common |
| Falco | 12.5% | Common |
| Marth | 11.5% | Common |
| Captain Falcon | 7.1% | Moderate |
| Jigglypuff | 6.5% | Moderate |
| Peach | 5.9% | Moderate |
| Ice Climbers | 3.6% | Underrepresented |
| **Low-tier** | **< 5%** | **Severely underrepresented** |

For ExPhil's target characters (Mewtwo, Ganondorf, Link, G&W, Zelda), data is extremely scarce.

### Balancing Strategies

#### Strategy 1: Character Filtering

Filter replays to target characters only:
```python
TARGET_CHARACTERS = {
    "MEWTWO", "GANONDORF", "LINK",
    "ZELDA", "GAME_AND_WATCH", "ICE_CLIMBERS"
}

def filter_target_characters(replay):
    """Keep games with at least one target character"""
    chars = {replay.p1_character, replay.p2_character}
    return bool(chars & TARGET_CHARACTERS)
```

**ExPhil implementation**: See `scripts/cloud_filter_replays.py` and `docs/RUNPOD_FILTER.md`

#### Strategy 2: Upsampling

Repeat underrepresented character data:
```python
def get_sample_count(character):
    """More samples for rare characters"""
    base_count = {
        "FOX": 1,      # Common
        "MARTH": 1,
        "MEWTWO": 10,  # Rare - 10x more samples
        "GANONDORF": 8,
        "LINK": 6,
    }
    return base_count.get(character, 1)
```

#### Strategy 3: Loss Weighting

Weight loss by character rarity:
```python
def get_loss_weight(character):
    """Higher weight for rare characters"""
    frequencies = compute_character_frequencies(dataset)
    return 1.0 / (frequencies[character] + 0.01)
```

### Matchup Considerations

Some matchups are common, others rare:
- Fox vs Fox: Very common (mirror)
- Mewtwo vs Ganondorf: Extremely rare

Options:
1. **Accept imbalance**: Let model learn what's available
2. **Matchup upsampling**: Repeat rare matchups
3. **Synthetic generation**: Generate rare matchup data via self-play

---

## Data Augmentation

ExPhil implements augmentation in `lib/exphil/training/augmentation.ex`.

### Mirror Augmentation (Essential)

Horizontally flip game states to double effective data:

```elixir
def mirror(frame) do
  %{frame |
    game_state: mirror_game_state(game_state),
    controller: mirror_controller(controller)
  }
end

defp mirror_player(%Player{} = p) do
  %{p |
    x: -p.x,                    # Flip position
    facing: -p.facing,          # Flip direction
    speed_air_x_self: -p.speed_air_x_self,
    speed_ground_x_self: -p.speed_ground_x_self
  }
end
```

**Benefits**:
- Effectively 2x data
- Prevents left/right bias
- Melee stages are symmetric

[Research shows](https://arxiv.org/abs/2309.12815) mirror augmentation significantly improves imitation learning generalization.

### Noise Injection

Add small Gaussian noise to continuous values:

```elixir
def add_noise(frame, scale \\ 0.01) do
  # Add noise to positions
  player = %{player |
    x: player.x + gaussian_noise(scale),
    y: player.y + gaussian_noise(scale),
    percent: max(0.0, player.percent + gaussian_noise(scale * 10))
  }
end
```

**Benefits**:
- Improves robustness to state estimation errors
- Reduces overfitting
- Scale ~0.01 for positions, ~0.1 for percent

### Frame Delay Augmentation

For online play (18+ frame delay), augment training with simulated delay:

```elixir
def add_frame_delay(sequence, delay \\ 18) do
  # Actions are executed `delay` frames after observation
  observations = Enum.drop(sequence.states, -delay)
  actions = Enum.drop(sequence.actions, delay)
  %{states: observations, actions: actions}
end
```

This teaches the model to predict actions based on delayed observations.

### What NOT to Augment

- **Discrete values**: Action states, stock counts
- **Binary flags**: on_ground, invulnerable
- **Character IDs**: Never change character
- **Stage**: Could flip for symmetric stages only

---

## Storage and Pipeline

### Recommended Structure

```
replays/
├── raw/                    # Original .slp files
│   ├── ranked_2024_01/
│   ├── tournament_genesis/
│   └── ...
├── filtered/               # Character-filtered
│   ├── mewtwo/
│   ├── ganondorf/
│   └── ...
├── parsed/                 # Peppi-processed
│   ├── mewtwo.parquet
│   └── ...
└── metadata/
    └── stats.json          # Dataset statistics
```

### Cloud Storage

See `docs/REPLAY_STORAGE.md` for detailed setup.

| Provider | Storage | Egress | Best For |
|----------|---------|--------|----------|
| Backblaze B2 | $0.005/GB | $0.01/GB | Simple, cheap |
| Cloudflare R2 | $0.015/GB | **Free** | Frequent pulls |
| Google Drive | Free | Free | Small datasets |

### Processing Pipeline

```
1. Obtain replays
   └── Slippi Discord / Tournament archives / Personal

2. Filter by character
   └── scripts/cloud_filter_replays.py
   └── Output: filtered/{character}/*.slp

3. Parse with Peppi
   └── peppi-py / peppi-slp
   └── Output: parsed/{character}.parquet

4. Compute statistics
   └── Character distribution
   └── Skill estimation
   └── Frame counts

5. Upload to cloud
   └── rclone sync to B2/R2

6. Training pulls from cloud
   └── fetch_replays.sh on RunPod
```

---

## Dataset Statistics

Track these metrics for your dataset:

### Per-Character Stats

```json
{
  "mewtwo": {
    "games": 1523,
    "frames": 8234567,
    "unique_opponents": 847,
    "matchup_distribution": {
      "fox": 0.34,
      "marth": 0.18,
      "falco": 0.15
    },
    "avg_game_length_frames": 5407,
    "estimated_skill_distribution": {
      "high": 0.23,
      "medium": 0.54,
      "low": 0.23
    }
  }
}
```

### Quality Indicators

| Metric | Good | Concerning |
|--------|------|------------|
| L-cancel rate | > 70% | < 50% |
| Avg game length | 3000-8000 | < 2000 (stomps) |
| Unique actions | > 40 | < 20 |
| SD rate | < 5% | > 15% |

### Computing Statistics

```elixir
defmodule ExPhil.Data.Statistics do
  def compute(replays) do
    %{
      total_games: length(replays),
      total_frames: Enum.sum(Enum.map(replays, & &1.frame_count)),
      character_distribution: compute_char_dist(replays),
      avg_l_cancel_rate: compute_avg_l_cancel(replays),
      skill_estimates: compute_skill_estimates(replays)
    }
  end
end
```

---

## Legal and Ethical Considerations

### Replay Data Ownership

**Current understanding** (not legal advice):
- Replay files contain player inputs, not copyrighted content
- Anonymized data removes personal identifiers
- Used for research/education is generally acceptable

### Privacy

- **Always anonymize** player tags before sharing
- Don't publish data that could identify individuals
- Respect opt-out requests

### Fair Use for AI Training

- Training AI on gameplay data is generally considered transformative
- Commercial use may have different considerations
- Nintendo's stance on derivative works varies

### Best Practices

1. **Use anonymized sources** (Fizzi's collections)
2. **Don't redistribute** raw player data
3. **Credit data sources** in publications
4. **Be transparent** about training data composition
5. **Respect community norms** of the Slippi community

---

## Practical Recommendations

### For ExPhil Low-Tier Training

1. **Start with Fizzi's collections** - Best quality/quantity ratio
2. **Filter aggressively for target characters** - Use RunPod filtering
3. **Apply 50% mirror augmentation** - Standard for all training
4. **Weight by estimated skill** - Prioritize cleaner data
5. **Track statistics** - Know your dataset composition

### Minimum Viable Dataset

| Character | Min Games | Min Frames | Notes |
|-----------|-----------|------------|-------|
| Mewtwo | 500 | 2.5M | Primary target |
| Ganondorf | 300 | 1.5M | Simpler, less data ok |
| Link | 400 | 2M | Projectiles need variety |
| G&W | 200 | 1M | Unique frame data |
| Ice Climbers | 400 | 2M | Nana handling critical |

### Scaling Up

Once baseline works:
1. Add more tournament data
2. Include adjacent skill levels
3. Expand matchup coverage
4. Consider pro player replay requests

---

## Tools Reference

### Parsing

| Tool | Language | Speed | Features |
|------|----------|-------|----------|
| [peppi](https://github.com/hohav/peppi) | Rust | Fastest | Arrow output |
| [peppi-py](https://github.com/hohav/peppi-py) | Python | Fast | Zero-copy |
| [slippi-js](https://github.com/project-slippi/slippi-js) | JavaScript | Moderate | Official |
| [py-slippi](https://github.com/hohav/py-slippi) | Python | Moderate | Easy to use |

### Filtering

| Tool | Purpose |
|------|---------|
| `cloud_filter_replays.py` | ExPhil character filter |
| [Slippipedia](https://github.com/cbartsch/Slippipedia) | GUI filtering |
| [slip.py](https://github.com/pcrain/slip.py) | Search engine |

### Analysis

| Tool | Purpose |
|------|---------|
| [slippi-stats](https://github.com/vinceau/slippi-stats) | Browser stats |
| [Chart.slp](https://chartslp.com/global) | Global statistics |
| [Lucky Stats](https://luckystats.gg/) | Player ratings |

---

## References

### Data Sources
- [Slippi Discord](https://discord.gg/slippi) - Anonymized ranked collections
- [ThePlayerDatabase](https://github.com/smashdata/ThePlayerDatabase) - Tournament metadata
- [Liquipedia Statistics](https://liquipedia.net/smash/Portal:Statistics/Melee/2024) - Character usage

### Research
- [Improving Generalization with Data Augmentation](https://arxiv.org/abs/2309.12815) - State augmentation for IL
- [Guided Data Augmentation for Offline RL](https://arxiv.org/abs/2310.18247) - Expert-quality augmentation
- [slippi-ai](https://github.com/vladfi1/slippi-ai) - Reference implementation

### ExPhil Docs
- [REPLAY_STORAGE.md](../REPLAY_STORAGE.md) - Cloud storage setup
- [RUNPOD_FILTER.md](../RUNPOD_FILTER.md) - Character filtering on cloud
- [RCLONE_GDRIVE.md](../RCLONE_GDRIVE.md) - Google Drive downloads
