# Character Considerations for Melee AI

This document covers character-specific considerations for training Melee AI, with particular focus on the low-tier characters targeted by ExPhil.

## Character Tiers Overview

### Top Tier (Most Studied)

| Character | Key Strengths | AI Challenges |
|-----------|---------------|---------------|
| **Fox** | Speed, frame data, shine | Frame-perfect tech required |
| **Falco** | Lasers, shine, combos | Similar to Fox |
| **Marth** | Range, edgeguarding | Spacing precision |
| **Sheik** | Reliable, safe options | Transform mechanic |
| **Jigglypuff** | Rest punishes, air game | Unique movement |

**Existing AI work**: slippi-ai, Phillip primarily focused here

### Mid Tier (Limited Study)

| Character | Key Strengths | AI Challenges |
|-----------|---------------|---------------|
| **Peach** | Float, turnips | RNG items |
| **Captain Falcon** | Speed, combo game | Execution demands |
| **Ice Climbers** | Wobbling, desyncs | Dual character control |
| **Pikachu** | Recovery, speed | Small size, unique physics |

### Low Tier (ExPhil Targets)

| Character | Key Strengths | AI Challenges |
|-----------|---------------|---------------|
| **Mewtwo** | Teleport, disable | Long recovery windows |
| **Ganondorf** | Power, punishes | Slow, exploitable |
| **Link** | Projectiles, range | Item management |
| **Game & Watch** | Unique moves | No L-cancel, RNG |
| **Zelda** | Kick, transform | Transform timing |

## ExPhil Target Characters

### Mewtwo

**Character Profile**:
- Weight: 85 (light)
- Fall speed: 1.5 (floaty)
- Air speed: 1.2 (good)
- Unique: Second jump with huge tail hitbox

**Technical Requirements**:

| Technique | Frames | Difficulty | Notes |
|-----------|--------|------------|-------|
| Teleport recovery | Variable | Medium | 8 angles |
| DJC (Double Jump Cancel) | 5 | Hard | Critical for neutral |
| Confusion | 17 | Easy | Reflector + command grab |
| Disable | 28 | Medium | Stun move, requires setup |

**AI Considerations**:

1. **Recovery**: Teleport angles must be precise
   - Need all 8 cardinal + diagonal angles
   - Timing window for sweetspot
   - Stall options

2. **Neutral Game**: DJC aerials are core
   - Need to learn rising vs falling aerials
   - Spacing with tail hitbox
   - Confusion reflects and grabs

3. **Edgeguarding**: Long range options
   - Fair/Bair spacing
   - Teleport to intercept
   - Stage spike with dtilt

**Embedding needs**:
- Teleport charge state
- Disable stun tracking
- Shadow Ball charge level

### Ganondorf

**Character Profile**:
- Weight: 109 (heavy)
- Fall speed: 2.0 (fast faller)
- Air speed: 0.83 (poor)
- Unique: Highest damage output, slowest character

**Technical Requirements**:

| Technique | Frames | Difficulty | Notes |
|-----------|--------|------------|-------|
| Stomp | 16 startup | Easy | Spike, main combo tool |
| Flame Choke | 20 | Easy | Command grab |
| Wizard's Foot | 15 | Easy | Mobility option |
| Waveland | 10 | Medium | Essential for movement |

**AI Considerations**:

1. **Punish Game**: Core strength
   - Stomp → knee/fair/uair followups
   - Flame choke setups
   - Hard reads for kills

2. **Neutral Game**: Baiting and reading
   - Can't approach safely
   - Must anti-air and whiff punish
   - Platform camping options

3. **Recovery**: Major weakness
   - Up-B is short and predictable
   - Must learn drift and mix angles
   - Wizard's Foot stall

**Embedding needs**:
- Standard is sufficient
- Focus on spacing and timing

### Link

**Character Profile**:
- Weight: 104 (heavy)
- Fall speed: 2.13 (fast faller)
- Air speed: 1.0 (average)
- Unique: Three projectiles, tether grab

**Technical Requirements**:

| Technique | Frames | Difficulty | Notes |
|-----------|--------|------------|-------|
| Bomb pull | 40 | Easy | Item generation |
| Boomerang | 27 | Easy | Angled projectile |
| Arrow | 17 | Easy | Fast projectile |
| Z-air | 12 | Medium | Spacing tool |

**AI Considerations**:

1. **Projectile Game**: Core identity
   - Bomb routing (throw, catch, z-drop)
   - Boomerang angles and catches
   - Arrow pressure and edgeguard

2. **Item Management**: Unique challenge
   - Track bomb timer
   - Catch boomerang
   - Z-drop techs

3. **Recovery**: Tether adds options
   - Bomb jump
   - Tether angles
   - Boomerang return timing

**Embedding needs**:
- Projectile state (type, position, velocity, owner)
- Bomb timer
- Boomerang return tracking
- Item hold state

### Game & Watch

**Character Profile**:
- Weight: 60 (lightest)
- Fall speed: 1.7 (floaty)
- Air speed: 1.1 (good)
- Unique: No L-cancel, bucket, RNG hammer

**Technical Requirements**:

| Technique | Frames | Difficulty | Notes |
|-----------|--------|------------|-------|
| Bucket | Variable | Medium | Absorbs projectiles |
| Judgment | 23 | Easy | RNG 1-9 |
| Bacon | 17 | Easy | Projectile spam |
| Bucket braking | Variable | Medium | Recovery option |

**AI Considerations**:

1. **No L-Cancel**: Different timing model
   - Landing lag is fixed
   - Aerial timing more forgiving
   - Different combo windows

2. **RNG Elements**: Bucket and hammer
   - Bucket absorption tracking
   - Hammer outcome (can't predict)
   - Must play around variance

3. **Survivability**: Light but floaty
   - Good recovery mix-ups
   - Bucket braking
   - Fire stall

**Embedding needs**:
- Bucket charge level (0-3)
- No L-cancel timing adjustment
- Standard otherwise

### Zelda

**Character Profile**:
- Weight: 90 (light)
- Fall speed: 1.4 (floaty)
- Air speed: 0.91 (poor)
- Unique: Transform to Sheik

**Technical Requirements**:

| Technique | Frames | Difficulty | Notes |
|-----------|--------|------------|-------|
| Lightning Kick | 1 frame sweetspot | Hard | Critical for kills |
| Transform | 55 | Easy | Character swap |
| Din's Fire | 60 | Easy | Projectile |
| Nayru's Love | 20 | Easy | Reflector |

**AI Considerations**:

1. **Sweetspot Precision**: Lightning kicks
   - 1 frame sweetspot
   - Spacing critical
   - Must learn exact hitboxes

2. **Transform Strategy**: When to be Zelda vs Sheik
   - Kill setups as Zelda
   - Neutral as Sheik
   - Transform punish windows

3. **Matchup Dependent**: Some matchups unplayable as pure Zelda
   - Must incorporate transform
   - Adds state complexity

**Embedding needs**:
- Transform availability/cooldown
- Consider Sheik state if transform included

### Ice Climbers

**Character Profile**:
- Weight: 88 (Popo), 95 (Nana)
- Fall speed: 1.6
- Air speed: 0.7 (poor)
- Unique: Dual character, desyncs, wobbling

**Technical Requirements**:

| Technique | Frames | Difficulty | Notes |
|-----------|--------|------------|-------|
| Wobble | Infinite | Hard | Frame-perfect inputs |
| Handoff | Variable | Hard | Regrab technique |
| Desync | Variable | Very Hard | Separate Nana control |
| Blizzard wall | 49 | Medium | Zoning tool |

**AI Considerations**:

1. **Nana Control**: Unique challenge
   - Nana follows Popo with delay
   - Desync separates their actions
   - Must track both characters

2. **Wobbling**: Infinite if grabbed
   - Frame-perfect A presses
   - Requires grab and Nana alive
   - Controversial but effective

3. **Recovery**: Popo + Nana
   - Belay requires both
   - Nana can be separated
   - Complex edgeguard scenarios

**Embedding needs**:
- Full Nana state (position, action, damage)
- Desync state tracking
- Grab state for wobble
- Belay sync state

## Embedding Dimensions by Character

| Character | Base | Special | Total Notes |
|-----------|------|---------|-------------|
| Standard | 484 | 0 | Default player embedding |
| Mewtwo | 484 | +2 | Teleport charge, shadow ball |
| Link | 484 | +15 | Projectile slots |
| G&W | 484 | +1 | Bucket level |
| Zelda | 484 | +1 | Transform state |
| ICs | 484 | +449 | Full Nana with `nana_mode: :full` |

## Character-Specific Rewards

### Mewtwo

```elixir
rewards = %{
  recovery_success: 0.1,      # Made it back
  teleport_sweetspot: 0.05,   # Grabbed ledge cleanly
  djc_aerial: 0.02,           # Used DJC
  confusion_reflect: 0.05,    # Reflected projectile
  disable_stun: 0.1,          # Successfully disabled
}
```

### Ganondorf

```elixir
rewards = %{
  stomp_hit: 0.05,           # Landing stomps
  punish_damage: 0.01,       # High damage combos
  spacing_neutral: 0.02,     # Not getting hit in neutral
  recovery_survival: 0.1,    # Made it back despite weakness
}
```

### Link

```elixir
rewards = %{
  projectile_hit: 0.02,      # Boomerang/arrow hits
  bomb_tech: 0.05,           # Bomb jump, z-drop tech
  item_catch: 0.03,          # Caught returning projectile
  edgeguard_projectile: 0.1, # Killed with projectile offstage
}
```

### Ice Climbers

```elixir
rewards = %{
  wobble_damage: 0.001,      # Per % during wobble
  handoff_success: 0.1,      # Completed handoff
  desync_tech: 0.05,         # Desync'd and hit
  nana_survival: 0.05,       # Kept Nana alive
  nana_death: -0.2,          # Lost Nana
}
```

## Data Availability

### Replay Abundance (Estimated)

| Character | Ranked Games | Quality |
|-----------|--------------|---------|
| Fox | 100K+ | High (many pros) |
| Falco | 80K+ | High |
| Marth | 70K+ | High |
| Sheik | 50K+ | High |
| Mewtwo | 5K | Medium |
| Ganondorf | 8K | Medium |
| Link | 3K | Low |
| G&W | 2K | Low |
| Zelda | 1K | Very Low |
| ICs | 10K | Medium (technical) |

### Data Augmentation Strategies

**Mirror augmentation**: Doubles data for symmetric characters
- Works for: Mewtwo, Ganondorf, G&W
- Requires care for: Link (item hand), ICs (Nana side)

**Multi-character training**: Use general model, fine-tune per character
- Eric Gu's finding: All-character > single-character
- May help low-data characters

## References

- [SmashWiki Character Data](https://www.ssbwiki.com/Category:Characters_(SSBM))
- [IKneeData Frame Data](https://ikneedata.com/)
- [Melee Library](https://www.meleelibrary.com/) - Tech skill guides
