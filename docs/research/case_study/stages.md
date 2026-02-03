# Stage Considerations for Melee AI

This document covers stage-specific considerations for Melee AI training, including dynamic elements, asymmetries, blast zones, and embedding strategies.

## Why Stages Matter for AI

Stages significantly affect gameplay:
- **Blast zones** determine kill percentages
- **Platform layouts** enable different combo routes
- **Dynamic elements** create unpredictable situations
- **Asymmetries** can favor certain positions/directions

An AI that ignores stage context will:
- Miss kills due to incorrect knockback estimation
- Drop combos when platforms move
- Get cheesed by Randall or Whispy
- Fail to exploit stage advantages

---

## Tournament Legal Stages

### Stage List

| Stage | ID | Type | Key Feature |
|-------|-----|------|-------------|
| Battlefield | 0x1F (31) | Starter | Tri-plat, ledge issues |
| Final Destination | 0x20 (32) | Starter | Flat, no platforms |
| Fountain of Dreams | 0x02 (2) | Starter | Moving platforms |
| Yoshi's Story | 0x08 (8) | Starter | Randall, small blast zones |
| Dream Land | 0x1C (28) | Counterpick | Whispy, large blast zones |
| Pokemon Stadium | 0x12 (18) | Counterpick | Transformations (frozen optional) |

### Blast Zone Comparison

From [SmashWiki debug data](https://www.ssbwiki.com/Debug_menu_(SSBM)/Stage_data):

| Stage | Ceiling | Side | Bottom | Notes |
|-------|---------|------|--------|-------|
| Yoshi's Story | 168 | ±175.7 | -91 | **Smallest** - early kills |
| Battlefield | 200 | ±224 | -108.8 | Average |
| Final Destination | 188 | ±246 | -140 | Wide, deep |
| Fountain of Dreams | 202.5 | ±198.75 | -146.25 | High ceiling |
| Pokemon Stadium | 180 | ±230 | -111 | Low ceiling |
| Dream Land | 200 | ±255 | -123 | **Largest** - live longer |

**AI implications**:
- On Yoshi's, kills happen ~15-20% earlier
- On Dream Land, need higher percent for kills
- Must adjust kill thresholds per stage

---

## Dynamic Stage Elements

### Fountain of Dreams - Moving Platforms

**Overview**: Two side platforms move independently up and down, while center platform stays fixed.

**Platform Behavior** ([SmashWiki](https://www.ssbwiki.com/Fountain_of_Dreams)):
- Platforms move in **no particular correspondence** (independent)
- Can rise to nearly center platform height
- Can sink through the stage (temporarily unusable)
- Movement is **not random** - follows a pattern, but patterns are complex

**Height Range**:
- Highest: Just under center platform (~54 units)
- Lowest: Below stage surface (disappear)

**AI Challenges**:
1. **Combo routing**: Platform height affects follow-ups
2. **Landing options**: Platform may not be there
3. **Edge guarding**: Low platforms affect recovery paths

**Embedding Strategy**:

```elixir
# Option 1: Explicit platform heights (recommended)
defstruct [
  # ... other fields
  fod_left_platform_height: float(),   # Y position
  fod_right_platform_height: float()   # Y position
]

# Embed as normalized floats
def embed_fod_platforms(left_y, right_y) do
  # Normalize: typical range 0-54, cap at 1.0
  left_norm = min(max(left_y / 54.0, 0.0), 1.0)
  right_norm = min(max(right_y / 54.0, 0.0), 1.0)
  Nx.tensor([left_norm, right_norm], type: :f32)
end

# Option 2: Platform existence flags
def embed_fod_platforms_binary(left_y, right_y) do
  # Just track if platform is usable (above threshold)
  left_usable = if left_y > 5.0, do: 1.0, else: 0.0
  right_usable = if right_y > 5.0, do: 1.0, else: 0.0
  Nx.tensor([left_usable, right_usable], type: :f32)
end
```

**slippi-ai approach**: Includes `fod_platforms` (2 dims) for left/right height.

---

### Yoshi's Story - Randall the Cloud

**Overview**: A cloud platform that circles the stage on a fixed timer.

**Timing** ([SmashWiki](https://www.ssbwiki.com/Yoshi's_Story), [Smashboards](https://smashboards.com/threads/randall-the-cloud-yoshis-story-timings.355920/)):
- **Cycle**: Exactly 20 seconds total
- **Pattern**: Counter-clockwise
- **Breakdown**:
  - 5 seconds: Hidden on right side
  - 5 seconds: Traversing stage (right → left)
  - 5 seconds: Hidden on left side
  - 5 seconds: Traversing stage (left → right)

**Start Position**: Does not appear for first 10 seconds, then emerges on right.

**Timer-Based Position** (for timed matches):
```
Looking at timer X:YZ:
- Y is even → Randall on left side
- Y is odd → Randall on right side
- Z = 4 → Randall emerges
- Z = 0 → Randall disappears
```

**AI Implications**:
1. **Recovery**: Randall can save or deny recoveries
2. **Edge guarding**: Standing on Randall extends edge guard range
3. **Stalling**: Can be used for timeout strats
4. **Positioning**: Know when Randall will appear

**Embedding Strategy**:

```elixir
# Option 1: Explicit position (recommended)
defstruct [
  randall_x: float(),  # X position (-80 to +80, outside = hidden)
  randall_y: float()   # Y position (fixed when visible)
]

def embed_randall(x, y) do
  # Normalize x to [-1, 1], y is relatively fixed
  x_norm = x / 80.0
  visible = if abs(x) <= 80.0, do: 1.0, else: 0.0
  Nx.tensor([visible, x_norm], type: :f32)
end

# Option 2: Cycle phase (0-1)
def embed_randall_phase(game_frame) do
  # 20 second cycle = 1200 frames
  # First 10 seconds (600 frames) Randall is hidden
  adjusted_frame = max(0, game_frame - 600)
  phase = rem(adjusted_frame, 1200) / 1200.0
  Nx.tensor([phase], type: :f32)
end

# Option 3: Discretized regions
def embed_randall_region(x) do
  region = cond do
    x < -80 -> 0  # Hidden left
    x < -40 -> 1  # Emerging left
    x < 0 -> 2    # Center-left
    x < 40 -> 3   # Center-right
    x < 80 -> 4   # Emerging right
    true -> 5     # Hidden right
  end
  Primitives.one_hot(region, size: 6)
end
```

**slippi-ai approach**: Includes `randall` (2 dims) for x, y position.

---

### Yoshi's Story - Shy Guys

**Behavior** ([SmashWiki](https://www.ssbwiki.com/Yoshi's_Island:_Yoshi's_Story)):
- Propeller Shy Guys fly across stage carrying fruit
- Can be hit, causing hitlag
- Do NOT drop food when items are off

**AI Implications**:
1. **Hitlag disruption**: Accidentally hitting Shy Guys disrupts combo timing
2. **Minor obstacle**: Can block projectiles briefly

**Embedding Strategy**: Generally **ignore** - Shy Guys are minor and unpredictable. The hitlag disruption is rare enough that embedding their position isn't worth the complexity.

---

### Dream Land - Whispy Woods

**Behavior** ([SmashWiki](https://www.ssbwiki.com/Whispy_Woods)):
- Blows wind gusts every 10-20 seconds (random timing)
- Wind direction: Faces side with more fighters
- Wind duration: 4 seconds per gust
- Effect: Pushes grounded and aerial players

**Port Priority**: In VS mode, prefers to blow toward higher controller port (P1 > P2 > P3 > P4).

**Wind Properties**:
- Slight upward angle (negligible ~0.04 units/frame)
- Stops exactly at ledge (offstage players unaffected)
- Does NOT push players off stage when grounded

**AI Implications**:
1. **Recovery**: Wind can help or hinder recovery
2. **Neutral**: Affects movement speed and spacing
3. **Combos**: Can extend or break combos
4. **Shield pressure**: Wind affects out-of-shield options

**Embedding Strategy**:

```elixir
# Option 1: Binary wind state
defstruct [
  whispy_blowing: boolean(),
  whispy_direction: integer()  # -1 left, 0 none, 1 right
]

def embed_whispy(blowing, direction) do
  blowing_val = if blowing, do: 1.0, else: 0.0
  direction_val = direction * 1.0  # -1, 0, or 1
  Nx.tensor([blowing_val, direction_val], type: :f32)
end

# Option 2: Wind strength + direction as single value
def embed_whispy_simple(blowing, direction) do
  # Positive = blowing right, negative = blowing left, 0 = no wind
  wind = if blowing, do: direction * 1.0, else: 0.0
  Nx.tensor([wind], type: :f32)
end
```

**Note**: Whispy timing is random, so predicting when wind will occur is difficult. Focus on reacting to current wind state rather than predicting.

---

### Pokemon Stadium - Transformations

**Overview**: Stage transforms between 5 configurations every ~30 seconds.

**Configurations** ([SmashWiki](https://www.ssbwiki.com/Pok%C3%A9mon_Stadium)):

| Type | Features | Gameplay Effect |
|------|----------|-----------------|
| **Normal** | Flat with 2 platforms | Standard neutral |
| **Fire** | Elevated sides, tree | Camping potential |
| **Water** | Water on sides, windmill | Windmill enables wall infinites |
| **Rock** | Uneven terrain, rock pillar | Cave of life on rock |
| **Grass** | Bush, tree platform | Tree blocks projectiles |

**Transformation Timing**:
- ~15-20 seconds between transformations
- Warning animation before change
- Some time frozen during transition

**Frozen Stadium** ([Inven Global](https://www.invenglobal.com/articles/15576/the-return-of-unfrozen-pokemon-stadium-was-awful-and-awesome)):
- Modded version that disables transformations
- Was standard for online play 2020-2023
- Most majors have reverted to unfrozen due to Nintendo legal concerns

**AI Implications**:
1. **Water windmill**: Fox/Falco can infinite opponents into wall
2. **Rock cave**: Creates "cave of life" where knockback is absorbed
3. **Fire/Grass**: Encourages camping due to obstructed approaches
4. **Transformation timing**: Skilled players stall for favorable layouts

**Embedding Strategy**:

```elixir
# For Frozen Stadium: No special handling needed

# For Unfrozen Stadium:
defstruct [
  stadium_transformation: integer()  # 0=normal, 1=fire, 2=water, 3=rock, 4=grass
]

def embed_stadium(transformation) do
  Primitives.one_hot(transformation, size: 5)
end

# Or include transformation + time until next
def embed_stadium_full(transformation, time_in_transform) do
  transform_emb = Primitives.one_hot(transformation, size: 5)
  # Normalize time: typical transform lasts ~900 frames (15 sec)
  time_norm = min(time_in_transform / 900.0, 1.0)
  Nx.concatenate([transform_emb, Nx.tensor([time_norm])])
end
```

**Recommendation**: If using unfrozen Stadium, embed the current transformation. For ExPhil, consider training primarily on Frozen Stadium for simplicity, with optional unfrozen handling.

---

## Stage Asymmetries

### Battlefield Ledge Issues

**The Problem** ([SmashWiki](https://www.ssbwiki.com/Battlefield_(SSBM))):
- Ledge sweetspot doesn't match visual position
- Angled walls cause characters to slip underneath
- Known as "getting Battlefielded" or "butterledge"

**Affected Recoveries**:
- Fox/Falco side-B can clip under and die
- Peach float can drift past sweetspot
- Any recovery that approaches at specific angles

**Left vs Right Ledge**:
- Technically symmetric, but players report different "feel"
- Angle calculations for ledgedash differ: "only for left ledge, right ledge would be mirrored"

**AI Implications**:
1. **Recovery angle**: Must be precise, stage-specific
2. **Mirror augmentation**: Left/right ledge handling differs slightly
3. **Edge guard positioning**: Account for ledge grab failures

**Embedding Strategy**: No special embedding needed - the ledge geometry is fixed. However, **training should include examples of failed recoveries** so the agent learns the sweetspot.

---

### Stage Coordinate Systems

**Coordinate Origin**: Stage center is (0, 0)

**Standard Positions** ([Smashboards debug data](https://smashboards.com/threads/stage-blast-zones-via-debug-mode.319898/)):

| Stage | Left Edge | Right Edge | Main Platform Y |
|-------|-----------|------------|-----------------|
| Battlefield | -68.4 | +68.4 | 0 |
| Final Destination | -85.57 | +85.57 | 0 |
| Fountain of Dreams | -63.35 | +63.35 | 0.62 |
| Yoshi's Story | -56 | +56 | 0 (angled) |
| Dream Land | -77.27 | +77.27 | 0.01 |
| Pokemon Stadium | -87.75 | +87.75 | 0 |

**Player Start Positions** (Battlefield example):
- Player 1: (-42.00, 26.60)
- Player 2: (+42.00, 28.00)

**Note**: Start positions are slightly asymmetric! P1 starts slightly lower than P2.

---

## Embedding Strategy Summary

### Current ExPhil Approach

```elixir
# From lib/exphil/embeddings/primitives.ex
@stage_size 64

def stage_embed(stage) do
  one_hot(stage, size: @stage_size, clamp: true)
end
```

**Dimensions**: 64 (one-hot)
**Dynamic features**: None currently

### Recommended Enhancement

Add stage-specific dynamic features:

```elixir
defmodule ExPhil.Embeddings.StageFeatures do
  @moduledoc """
  Dynamic stage element embeddings.
  """

  @doc """
  Embed stage-specific dynamic features.

  Returns different sized tensors depending on stage:
  - FoD: 2 dims (left/right platform heights)
  - Yoshi's: 2 dims (Randall x, visibility)
  - Dream Land: 2 dims (Whispy blowing, direction)
  - Pokemon Stadium: 5 dims (transformation one-hot)
  - Others: 0 dims
  """
  def embed_dynamic(game_state) do
    stage = game_state.stage || 0

    case stage do
      2 -> embed_fod(game_state)       # Fountain of Dreams
      8 -> embed_yoshis(game_state)    # Yoshi's Story
      28 -> embed_dreamland(game_state) # Dream Land
      18 -> embed_stadium(game_state)   # Pokemon Stadium
      _ -> Nx.tensor([], type: :f32)    # No dynamic features
    end
  end

  defp embed_fod(gs) do
    left = gs.fod_left_platform || 30.0
    right = gs.fod_right_platform || 30.0
    Nx.tensor([left / 54.0, right / 54.0], type: :f32)
  end

  defp embed_yoshis(gs) do
    randall_x = gs.randall_x || 0.0
    visible = if abs(randall_x) <= 80.0, do: 1.0, else: 0.0
    Nx.tensor([visible, randall_x / 80.0], type: :f32)
  end

  defp embed_dreamland(gs) do
    blowing = if gs.whispy_blowing, do: 1.0, else: 0.0
    direction = (gs.whispy_direction || 0) * 1.0
    Nx.tensor([blowing, direction], type: :f32)
  end

  defp embed_stadium(gs) do
    transformation = gs.stadium_transformation || 0
    Primitives.one_hot(transformation, size: 5)
  end
end
```

### Fixed vs Variable Feature Size

**Option A: Fixed size with padding**
```elixir
# Always 8 dims, pad unused with zeros
def embed_dynamic_fixed(game_state) do
  base = embed_dynamic(game_state)
  pad_size = 8 - Nx.size(base)
  if pad_size > 0 do
    Nx.concatenate([base, Nx.broadcast(0.0, {pad_size})])
  else
    base
  end
end
```

**Option B: Stage-conditioned MLP**
```elixir
# Let network learn to handle variable features
# Stage one-hot tells network which features to expect
```

**Recommendation**: Option A (fixed size) is simpler and works well. 8 dims for stage dynamics is negligible overhead.

---

## Training Considerations

### Mirror Augmentation and Stages

Most stages are horizontally symmetric, so mirror augmentation works:
- ✅ Battlefield (symmetric)
- ✅ Final Destination (symmetric)
- ✅ Fountain of Dreams (symmetric)
- ✅ Yoshi's Story (symmetric)
- ✅ Dream Land (symmetric)
- ⚠️ Pokemon Stadium (mostly symmetric, but transformations may have slight asymmetries)

When mirroring:
```elixir
def mirror_stage_features(game_state) do
  # FoD: swap left/right platform
  # Yoshi's: negate Randall x
  # Dream Land: flip Whispy direction
  # Stadium: no change needed
end
```

### Stage-Specific Training

**Option 1: All stages together**
- Pro: Single model handles everything
- Con: May underfit stage-specific behaviors

**Option 2: Stage-specific fine-tuning**
- Train general model on all stages
- Fine-tune on specific stages for deployment

**Option 3: Stage-weighted sampling**
```elixir
# Weight replays by stage importance
stage_weights = %{
  battlefield: 1.0,
  fd: 1.0,
  fod: 1.2,    # Extra weight for platform dynamics
  yoshis: 1.2,  # Extra weight for Randall
  dreamland: 1.0,
  stadium: 0.8  # Less weight (transformation complexity)
}
```

### Blast Zone Awareness

Include stage-specific kill thresholds in value function:
```elixir
def estimate_kill_percent(character, stage, direction) do
  base_kill = character_base_kill_percent(character)

  stage_modifier = case stage do
    8 -> 0.85   # Yoshi's - kills 15% earlier
    28 -> 1.15  # Dream Land - kills 15% later
    _ -> 1.0
  end

  base_kill * stage_modifier
end
```

---

## Data Availability

### Stage Distribution in Replays

Typical ranked replay distribution:
| Stage | Frequency |
|-------|-----------|
| Battlefield | ~25% |
| Final Destination | ~20% |
| Yoshi's Story | ~18% |
| Fountain of Dreams | ~15% |
| Dream Land | ~12% |
| Pokemon Stadium | ~10% |

**Implication**: Some stages have less data. Consider upsampling rare stages if training stage-specific behaviors.

### Getting Stage Dynamic Data

**Challenge**: Standard Slippi replays include stage ID but not always dynamic element positions.

**Solutions**:
1. **Compute from frame count**: Randall position is deterministic
2. **libmelee**: Provides Randall position in `gamestate.stage.randall`
3. **Slippi Parser extensions**: Some parsers extract stage elements

```python
# libmelee example
randall_x = gamestate.stage.randall_position.x
randall_y = gamestate.stage.randall_position.y

# FoD platforms
left_plat_y = gamestate.stage.left_platform_height
right_plat_y = gamestate.stage.right_platform_height
```

---

## Practical Recommendations

### Phase 1: Basic Stage Handling

1. **Stage one-hot** (current): Already implemented, 64 dims
2. **Blast zone awareness**: Implicitly learned from kill data
3. **No dynamic features**: Simpler model, still competitive

### Phase 2: Enhanced Stage Features

1. **Add FoD platforms** (+2 dims): Significant for combo routing
2. **Add Randall** (+2 dims): Important for recovery/edge guard
3. **Add Whispy** (+2 dims): Affects neutral significantly

### Phase 3: Full Stage Modeling

1. **Pokemon Stadium transformations**: If using unfrozen
2. **Predictive Randall**: Predict Randall position N frames ahead
3. **Stage-specific reward shaping**: Bonus for using stage elements

### ExPhil-Specific Notes

For low-tier characters:
- **Mewtwo**: Benefits from high ceiling stages (FoD, Dream Land)
- **Ganondorf**: Prefers small stages (Yoshi's) for early kills
- **Link**: Struggles on small stages, projectiles work better on FD
- **G&W**: Neutral on most stages
- **Ice Climbers**: Prefer flat stages (FD) for wobbling

Consider stage-specific training weighting based on character.

---

## References

### Technical Data
- [SmashWiki Debug Stage Data](https://www.ssbwiki.com/Debug_menu_(SSBM)/Stage_data)
- [Smashboards Blast Zones](https://smashboards.com/threads/stage-blast-zones-via-debug-mode.319898/)
- [Liquipedia Stage Pages](https://liquipedia.net/smash/Battlefield)

### Stage-Specific
- [Fountain of Dreams](https://www.ssbwiki.com/Fountain_of_Dreams) - SmashWiki
- [Yoshi's Story](https://www.ssbwiki.com/Yoshi's_Story) - SmashWiki
- [Randall Timings](https://smashboards.com/threads/randall-the-cloud-yoshis-story-timings.355920/) - Smashboards
- [Whispy Woods](https://www.ssbwiki.com/Whispy_Woods) - SmashWiki
- [Pokemon Stadium](https://www.ssbwiki.com/Pok%C3%A9mon_Stadium) - SmashWiki
- [Battlefield](https://www.ssbwiki.com/Battlefield_(SSBM)) - SmashWiki

### Research
- [Dignitas Stage Guide](https://dignitas.gg/articles/the-ins-and-outs-of-every-competitive-stage-in-melee)
- [Frozen Stadium Discussion](https://www.invenglobal.com/articles/15576/the-return-of-unfrozen-pokemon-stadium-was-awful-and-awesome)

### ExPhil Code
- `lib/exphil/embeddings/primitives.ex` - Current stage embedding
- `lib/exphil/embeddings/game.ex` - Game state embedding
