# Embedding Dimension Design Document

## Problem Space

Neural networks for Melee need to encode game state into fixed-size vectors. The key tradeoffs:

| Consideration | Larger Embeddings | Smaller Embeddings |
|--------------|-------------------|-------------------|
| **Information** | Preserves all state details | May lose critical info |
| **GPU efficiency** | More memory, slower matmuls | Better kernel utilization |
| **Generalization** | Risk of memorization | Forces learning abstractions |
| **Training speed** | Slower per step | Faster per step |
| **IC tech** | Can learn precise setups | May miss timing windows |

### Target: 512 Dimensions

Power-of-2 dimensions (256, 512, 1024, 2048) optimize GPU tensor core utilization. 512 is our target because:
- Small enough for fast inference (60 FPS requirement)
- Large enough for complex game state
- Sweet spot for CUDA kernel efficiency

---

## Current Architecture (1204 dims)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Game Embedding: 1204 dims                    │
├─────────────────────────────────────────────────────────────────┤
│  Player 1 (488 dims)                                            │
│  ├─ Base: 440 dims                                              │
│  │   ├─ percent: 1                                              │
│  │   ├─ facing: 1                                               │
│  │   ├─ x, y: 2                                                 │
│  │   ├─ action: 399 (one-hot)  ← LARGEST COMPONENT              │
│  │   ├─ character: 33 (one-hot)                                 │
│  │   ├─ invulnerable: 1                                         │
│  │   ├─ jumps_left: 1 (normalized)                              │
│  │   ├─ shield: 1                                               │
│  │   └─ on_ground: 1                                            │
│  ├─ Speeds: 5 dims (air_x, ground_x, y, attack_x, attack_y)     │
│  ├─ Frame info: 2 dims (hitstun_frames, action_frame)           │
│  ├─ Stock: 1 dim                                                │
│  ├─ Ledge distance: 1 dim                                       │
│  └─ Nana (compact): 39 dims                                     │
│      ├─ exists: 1                                               │
│      ├─ position: 2 (x, y)                                      │
│      ├─ facing: 1                                               │
│      ├─ on_ground: 1                                            │
│      ├─ percent: 1                                              │
│      ├─ stock: 1                                                │
│      ├─ hitstun_frames: 1 (currently hardcoded 0)               │
│      ├─ action_frame: 1 (currently hardcoded 0)                 │
│      ├─ invulnerable: 1                                         │
│      ├─ action_category: 25 (simplified one-hot)                │
│      └─ IC flags: 4 (attacking, grabbing, can_act, synced)      │
├─────────────────────────────────────────────────────────────────┤
│  Player 2 (488 dims) - Same structure                           │
├─────────────────────────────────────────────────────────────────┤
│  Stage: 64 dims (one-hot, most unused)                          │
├─────────────────────────────────────────────────────────────────┤
│  Player names: 112 dims (one-hot, currently unused)             │
├─────────────────────────────────────────────────────────────────┤
│  Spatial features: 4 dims                                       │
│  ├─ distance: 1                                                 │
│  ├─ relative_pos: 2 (x, y)                                      │
│  └─ frame_count: 1                                              │
├─────────────────────────────────────────────────────────────────┤
│  Projectiles: 35 dims (5 slots × 7 dims)                        │
│  └─ Per projectile: exists, owner, x, y, type, speed_x, speed_y │
├─────────────────────────────────────────────────────────────────┤
│  Controller (prev action): 13 dims                              │
│  ├─ buttons: 8                                                  │
│  ├─ main_stick: 2                                               │
│  ├─ c_stick: 2                                                  │
│  └─ shoulder: 1                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dimension Reduction Strategies

### 1. Learned Action Embedding (IMPLEMENTED)

**Problem:** Action one-hot is 399 dims × 2 players = 798 dims (66% of total!)

**Solution:** Replace one-hot with trainable embedding lookup.

```
Before: action_id → 399-dim one-hot → network
After:  action_id → embedding_table[action_id] → 64-dim vector → network
```

**Benefits:**
- 798 → 128 dims (-670 dims, -84%)
- Network learns action similarities (aerials cluster together)
- Faster training (smaller first layer)

**Implementation:**
```elixir
# In PlayerEmbed config
config = %PlayerEmbed{action_mode: :learned}

# In Policy network
model = Policy.build(embed_size: total_size, action_embed_size: 64)
```

**Tradeoff:** Embedding must be trained; can't use pretrained weights as easily.

### 2. Normalized Jumps (IMPLEMENTED)

**Problem:** Jumps one-hot is 7 dims (0-6 jumps for Puff/Kirby).

**Solution:** Normalize to single float: `jumps / 6.0`

**Benefits:**
- 7 → 1 dim (-6 dims per player, -12 total)
- Preserves ordinal information (2 jumps > 1 jump)
- Float captures "more jumps = better recovery"

**Implementation:**
```elixir
config = %PlayerEmbed{jumps_normalized: true}  # Default
```

### 3. Compact Nana (IMPLEMENTED)

**Problem:** Full Nana embedding is 449 dims (same as player).

**Solution:** Reduced representation preserving IC tech essentials.

**Current compact (39 dims):**
- Position, facing, on_ground, percent, stock
- Action category (25 categories vs 399 actions)
- IC-specific flags (attacking, grabbing, can_act, synced)

**Limitation:** Loses action specificity needed for precise setups.

### 4. Enhanced Compact Nana (IMPLEMENTED)

**Problem:** Compact Nana uses 25-dim action category, losing precise action info.

**Solution:** Replace action category with action ID for learned embedding.

**Enhanced compact (14 dims + 1 action ID per Nana):**
- Core state: exists, x, y, facing, on_ground, percent, stock (7 dims)
- Frame info: hitstun_frames, action_frame, invulnerable (3 dims)
- IC flags: is_attacking, is_grabbing, can_act, is_synced (4 dims)
- Nana action ID: Separate tensor, embedded by network (1 ID → 64 dims)

**Benefits:**
- 39 → 14 dims (-25 dims per player, -50 total)
- Preserves exact action via learned embedding (dair vs fair vs nair)
- Enables 4 action IDs total (2 players + 2 Nanas)

**Implementation:**
```elixir
# Enhanced Nana with learned actions (optimal for IC tech)
player_config = %PlayerEmbed{
  action_mode: :learned,
  nana_mode: :enhanced,
  with_nana: true
}
game_config = %GameEmbed{player: player_config}

# Check dimensions
GameEmbed.embedding_size(game_config)      # Total size including action IDs
GameEmbed.continuous_embedding_size(game_config)  # Continuous features only
GameEmbed.num_action_ids(game_config)      # 4 (2 players + 2 Nanas)

# Build policy network with 4 action IDs
model = Policy.build(
  embed_size: GameEmbed.embedding_size(game_config),
  action_embed_size: 64,
  num_action_ids: 4
)
```

**Tradeoff:** Requires training Nana action embeddings from scratch.

### 4. Stage Embedding (TODO)

**Problem:** 64-dim one-hot but only 6 competitive stages used.

**Proposed:** Competitive stage encoding + transformation state

```
Current:  64 dims (one-hot, 58 wasted)
Proposed: 11 dims total
  - Competitive stages: 6 dims (BF, FD, DL, YS, FoD, PS)
  - PS transformation: 4 dims (fire, water, grass, rock)
  - Other flag: 1 dim
```

**Savings:** 64 → 11 dims (-53 dims)

### 5. Player Names (TODO)

**Problem:** 112 dims allocated but unused.

**Options:**
- Remove entirely: -112 dims
- Small learned embedding: 112 → 16-32 dims (when implemented)
- Style tokens: Map player → learned style vector

**Current recommendation:** Remove (set `num_player_names: 0`) until player identification is implemented.

---

## Path to 512 Dimensions

### Option A: Minimal Changes (Learned Actions Only)

```
Component               One-Hot    Learned Actions
─────────────────────────────────────────────────
Player 1 base           440        41 (no action one-hot)
Player 1 speeds         5          5
Player 1 frame_info     2          2
Player 1 stock          1          1
Player 1 ledge_dist     1          1
Player 1 compact Nana   39         39
─────────────────────────────────────────────────
Player 1 total          488        89

Player 2 total          488        89

Stage                   64         64
Player names            112        112
Spatial features        4          4
Projectiles             35         35
Controller              13         13
Action IDs              0          2
─────────────────────────────────────────────────
TOTAL                   1204       408
```

**Result:** 408 dims + 2 action IDs = 410 continuous dims

Already under 512! But loses IC precision with compact Nana.

### Option B: Enhanced Compact Nana for IC Tech

**Goal:** Keep compact-ish Nana but preserve action specificity.

**Enhanced Compact Nana (proposed: ~50-60 dims):**

```
┌─────────────────────────────────────────────────────────────────┐
│  Enhanced Compact Nana: ~55 dims                                │
├─────────────────────────────────────────────────────────────────┤
│  Core state: 10 dims                                            │
│  ├─ exists: 1                                                   │
│  ├─ position: 2 (x, y)                                          │
│  ├─ facing: 1                                                   │
│  ├─ on_ground: 1                                                │
│  ├─ percent: 1                                                  │
│  ├─ stock: 1                                                    │
│  ├─ hitstun_frames: 1 (REAL data from opponent or estimate)     │
│  ├─ action_frame: 1 (REAL data - critical for timing)           │
│  └─ invulnerable: 1                                             │
├─────────────────────────────────────────────────────────────────┤
│  Action representation (choose one):                            │
│  ├─ Option 1: Action ID for learned embedding: 1 dim            │
│  │   (Network learns Nana action embedding separately)          │
│  ├─ Option 2: Full action one-hot: 399 dims (too big)           │
│  └─ Option 3: IC-relevant action subset: ~40 dims               │
│       (Only actions that matter for IC tech)                    │
├─────────────────────────────────────────────────────────────────┤
│  IC tech flags: 4 dims                                          │
│  ├─ is_attacking: 1                                             │
│  ├─ is_grabbing: 1                                              │
│  ├─ can_act: 1                                                  │
│  └─ is_synced: 1                                                │
├─────────────────────────────────────────────────────────────────┤
│  Grab-specific state (for regrab setups): ~4 dims               │
│  ├─ popo_grabbing: 1 (is Popo currently grabbing?)              │
│  ├─ grab_timer: 1 (normalized pummel/throw timing)              │
│  ├─ opponent_hitstun: 1 (can opponent escape?)                  │
│  └─ regrab_window: 1 (is Nana in position to regrab?)           │
└─────────────────────────────────────────────────────────────────┘
```

**With learned Nana action:** 10 + 1 + 4 + 4 = 19 dims + 1 Nana action ID
**With IC action subset:** 10 + 40 + 4 + 4 = 58 dims

### Option C: Full 512 Build with All Optimizations

```
Component                    Dims    Notes
───────────────────────────────────────────────────────────────
Player 1 (learned actions)   41      Base without action one-hot
Player 1 speeds              5
Player 1 frame_info          2
Player 1 stock               1
Player 1 ledge_dist          1
Player 1 enhanced Nana       20      With learned Nana action
───────────────────────────────────────────────────────────────
Player 1 subtotal            70

Player 2 subtotal            70
───────────────────────────────────────────────────────────────
Stage (competitive)          11      6 stages + 4 PS transform + 1 other
Player names                 0       Removed (unused)
Spatial features             4
Projectiles                  35
Controller                   13
───────────────────────────────────────────────────────────────
Continuous total             203

Action IDs                   4       2 player actions + 2 Nana actions
───────────────────────────────────────────────────────────────
TOTAL                        207 continuous + 4 action IDs
```

**With 64-dim embeddings for all 4 actions:** 207 + (4 × 64) = 463 dims

**Room to spare!** Could add:
- More IC tech features (+20 dims)
- Momentum/trajectory prediction (+10 dims)
- Platform state for FoD/YS (+10 dims)
- Padding to exactly 512 (+29 dims)

---

## IC Tech Requirements Analysis

### Guaranteed Regrab (Dthrow → Nana Dair → Regrab)

**Required state:**
1. Popo's current action (grab/throw state)
2. Throw type (which throw was used)
3. Nana's action (dair specifically, not just "aerial")
4. Nana's action_frame (timing within dair)
5. Opponent's hitstun_frames (can they act?)
6. Opponent's position relative to Nana
7. Opponent's DI/SDI (affects regrab window)

**Current compact Nana misses:** #3 (action specificity), #4 (action_frame is 0)

### Desync Setups (Nana Ftilt While Grabbing)

**Required state:**
1. Popo grabbing (boolean)
2. Nana's exact action (ftilt vs dtilt vs jab matters)
3. Nana's action_frame (when will hitbox come out?)
4. Nana's position relative to grabbed opponent
5. Opponent percent (affects hitstun of Nana's hit)

**Current compact Nana misses:** #2 (has category, not exact action), #3

### Wobbling Timing

**Required state:**
1. Popo grab timer (pummel timing)
2. Nana's action (pummel sync)
3. Opponent mash state (can they escape?)
4. Percent thresholds for kill confirms

**Current compact has:** Popo grabbing flag, Nana grabbing flag
**Missing:** Precise timing, mash state

---

## Recommendations

### For General Training
Use learned action embeddings with current compact Nana:
- 408 dims + 2 action IDs
- Fast training, good generalization
- Works for most characters

### For IC-Specific Training
Use enhanced compact Nana with Nana action ID:
- ~220 dims + 4 action IDs
- Preserves precise action info via learned embedding
- Add IC-specific features (grab timer, regrab window)

### For Maximum IC Tech
Use full Nana mode:
- ~900 dims with one-hot (not recommended)
- OR ~460 dims with learned actions (viable)
- Every detail preserved but slower training

---

## Implementation Status

| Optimization | Status | Savings | Config |
|-------------|--------|---------|--------|
| Learned action embedding | ✅ Done | -670 dims | `action_mode: :learned` |
| Normalized jumps | ✅ Done | -12 dims | `jumps_normalized: true` |
| Compact Nana | ✅ Done | -410 dims | `nana_mode: :compact` |
| Enhanced compact Nana | ✅ Done | -25 dims (vs compact) | `nana_mode: :enhanced` |
| 4 action IDs (player + Nana) | ✅ Done | Enables Nana action embedding | `nana_mode: :enhanced, with_nana: true` |
| Policy network 4 action IDs | ✅ Done | N/A | `num_action_ids: 4` |
| Competitive stage embed | 🔲 TODO | -53 dims | `stage_mode: :competitive` |
| Remove player names | 🔲 TODO | -112 dims | `num_player_names: 0` |
| IC Tech Feature Block | 🔲 TODO | +32 dims | Dedicated IC features |

---

## Concrete 512-Dim Implementation Plan

### Enhanced Compact Nana (15 dims + 1 action ID)

Replace current 39-dim compact Nana with action ID for learned embedding:

```elixir
# Enhanced Compact Nana structure
defp enhanced_compact_nana_embedding_size do
  1 +   # exists
  2 +   # x, y position (scaled)
  1 +   # facing
  1 +   # on_ground
  1 +   # percent (scaled)
  1 +   # stock (normalized)
  1 +   # hitstun_frames (normalized, REAL data)
  1 +   # action_frame (normalized, REAL data)
  1 +   # invulnerable
  # IC tech flags (keep these, they're useful)
  1 +   # is_attacking
  1 +   # is_grabbing
  1 +   # can_act
  1     # is_synced
  # = 14 dims continuous + 1 action ID
end
```

**Key change:** Remove 25-dim action_category, add 1-dim action_id for learned embedding.

### IC Tech Features Block (50 dims)

New dedicated block for Ice Climbers tech:

```elixir
# IC Tech Feature Block
defp ic_tech_embedding_size do
  # Grab state (8 dims)
  1 +   # popo_grabbing (bool)
  1 +   # popo_grab_timer (normalized 0-1, for pummel timing)
  1 +   # popo_throw_type (0=none, then one-hot for uthrow/dthrow/fthrow/bthrow)
  4 +   # throw_type_onehot (4 dims)
  1 +   # frames_since_throw (normalized, for regrab timing)

  # Opponent state in grab (8 dims)
  1 +   # opponent_hitstun_remaining (normalized)
  1 +   # opponent_percent (for combo/kill threshold)
  1 +   # opponent_di_x (if available from prev frame)
  1 +   # opponent_di_y
  1 +   # opponent_can_escape (bool - mash threshold)
  1 +   # opponent_actionable_frames (when can they act?)
  2 +   # opponent_position_relative_to_nana (x, y)

  # Nana sync state (6 dims)
  1 +   # nana_action_frame (normalized - critical for hitbox timing!)
  1 +   # nana_distance_to_opponent (for regrab range)
  1 +   # nana_can_regrab (bool - in range and actionable?)
  1 +   # frames_until_nana_hitbox (for dair/ftilt timing)
  1 +   # nana_in_desync (bool - different action category than popo)
  1 +   # nana_facing_opponent (bool)

  # Combo/setup detection (10 dims)
  1 +   # is_guaranteed_regrab_window (bool)
  1 +   # is_tech_chase_situation (bool)
  1 +   # is_wobble_ready (bool - both in grab + percent threshold)
  1 +   # is_handoff_possible (bool)
  1 +   # frames_in_current_setup (normalized)
  2 +   # optimal_nana_position_offset (x, y - where should Nana be?)
  3     # padding for alignment

  # Total: 8 + 8 + 6 + 10 = 32 dims (leaves 18 for future)
end
```

### Competitive Stage Embedding (11 dims)

```elixir
defp competitive_stage_embedding do
  # Competitive stages one-hot (6 dims)
  # BF=31, FD=2, DL64=28, YS=8, FoD=3, PS=18
  6 +

  # Pokemon Stadium transformation (4 dims, one-hot)
  # normal, fire, water, grass, rock
  4 +

  # Other stage flag (1 dim)
  1

  # Total: 11 dims
end
```

### Final 512-Dim Build

```
┌─────────────────────────────────────────────────────────────────┐
│                    512-DIM NETWORK INPUT                        │
├─────────────────────────────────────────────────────────────────┤
│  CONTINUOUS FEATURES (256 dims)                                 │
│  ├─ Player 1: 51 dims                                          │
│  │   ├─ Base (no action): 41                                   │
│  │   ├─ Speeds: 5                                              │
│  │   ├─ Frame info: 2                                          │
│  │   ├─ Stock: 1                                               │
│  │   ├─ Ledge dist: 1                                          │
│  │   └─ Enhanced Nana: 14 (continuous only)                    │
│  │       ├─ exists, x, y, facing, ground: 5                    │
│  │       ├─ percent, stock, hitstun, action_frame: 4           │
│  │       ├─ invulnerable: 1                                    │
│  │       └─ IC flags: 4                                        │
│  ├─ Player 2: 51 dims (same structure)                         │
│  ├─ Stage (competitive): 11 dims                               │
│  ├─ Spatial features: 4 dims                                   │
│  ├─ Projectiles: 35 dims                                       │
│  ├─ Controller: 13 dims                                        │
│  ├─ IC Tech Block: 32 dims                                     │
│  └─ Padding: 59 dims (for future features)                     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ACTION IDS → LEARNED EMBEDDINGS (256 dims)                     │
│  ├─ Player 1 action: 1 ID → 64-dim embedding                   │
│  ├─ Player 2 action: 1 ID → 64-dim embedding                   │
│  ├─ Player 1 Nana action: 1 ID → 64-dim embedding              │
│  └─ Player 2 Nana action: 1 ID → 64-dim embedding              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  TOTAL NETWORK INPUT: 256 + 256 = 512 dims                     │
└─────────────────────────────────────────────────────────────────┘
```

### What This Enables for IC Tech

**Guaranteed Regrab (Dthrow → Nana Dair → Regrab):**
- ✅ Popo throw type (dthrow specifically)
- ✅ Nana's exact action via action ID (dair, not just "aerial")
- ✅ Nana's action_frame (timing within dair animation)
- ✅ Opponent's hitstun_remaining (can they escape?)
- ✅ `is_guaranteed_regrab_window` flag
- ✅ Nana distance to opponent (in range?)

**Desync Setups (Nana Ftilt While Grabbing):**
- ✅ Popo grabbing flag + grab timer
- ✅ Nana's exact action via learned embedding (ftilt vs dtilt vs jab)
- ✅ Nana's action_frame (when will hitbox come out?)
- ✅ `nana_in_desync` flag
- ✅ `frames_until_nana_hitbox`

**Wobbling:**
- ✅ `is_wobble_ready` (both grabbing + percent threshold)
- ✅ Popo grab timer (pummel timing)
- ✅ Opponent mash state (`opponent_can_escape`)

**Handoffs:**
- ✅ `is_handoff_possible` flag
- ✅ Both Nana actions via learned embedding
- ✅ Position relative features

---

## Next Steps

1. ~~**Implement enhanced compact Nana**~~ ✅ **DONE**
   - ✅ Nana action ID (for learned embedding)
   - ⚠️ hitstun_frames (placeholder - need libmelee data)
   - ⚠️ action_frame (placeholder - need libmelee data)

2. ~~**Update GameEmbed to support 4 action IDs**~~ ✅ **DONE**
   - ✅ `num_action_ids/1` returns 0, 2, or 4 based on config
   - ✅ Nana action IDs appended when using enhanced mode
   - ✅ Policy network updated to accept configurable `num_action_ids`

3. **Implement IC Tech Feature Block** (TODO) with:
   - Grab state features (popo_grabbing, grab_timer, throw_type)
   - Regrab window detection (is_guaranteed_regrab_window)
   - Desync state tracking (nana_in_desync, frames_until_nana_hitbox)

4. **Implement competitive stage embedding** (TODO)
   - Replace 64-dim one-hot with 11-dim competitive encoding
   - Add PS transformation state (fire/water/grass/rock)

5. **Run benchmark script** to compare configurations:
   - `scripts/benchmark_embeddings.exs` (skeleton implemented)
   - Compare current_default, learned_actions, enhanced_nana, full_nana

6. **Test IC tech learning** with enhanced mode
   - Train on IC replays with enhanced Nana mode
   - Evaluate regrab/desync accuracy vs compact mode
