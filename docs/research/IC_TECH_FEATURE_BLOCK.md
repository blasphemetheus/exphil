# IC Tech Feature Block Design

**Status:** Planning
**Created:** 2026-01-29

This document outlines the design for dedicated Ice Climbers tech features in the game embedding.

---

## Overview

Ice Climbers require specialized features because their gameplay revolves around:
- **Grab game** - wobbles, handoffs, regrabs
- **Nana synchronization** - desync setups, coordinated attacks
- **Frame-perfect timing** - handoff windows are 1-3 frames

Without explicit features, the network must discover these patterns from raw action states, which is slower and less precise.

---

## Option A: Minimal Implementation (14 dims)

Core features that capture 80% of IC tech with minimal embedding overhead.

```elixir
defmodule ExPhil.Embeddings.ICTech do
  @moduledoc """
  Minimal IC tech features (14 dims).
  Only added when player character is Ice Climbers.
  """

  # Grab State (5 dims)
  # ─────────────────────
  # 1. popo_grabbing         - bool (1 dim)
  # 2. grab_timer_normalized - 0-1, frames held / max_grab_frames (1 dim)
  # 3. throw_type            - one-hot [none, uthrow, dthrow, fthrow, bthrow] (5 dims) - wait this is 5
  # Actually let's simplify:

  @spec embed(map(), map()) :: Nx.Tensor.t()
  def embed(popo_state, nana_state) do
    # --- Grab State (4 dims) ---
    popo_grabbing = is_grab_action?(popo_state.action_state)
    grab_timer = if popo_grabbing, do: normalize_grab_timer(popo_state), else: 0.0
    opponent_in_grab = is_grabbed_action?(get_opponent(popo_state).action_state)
    frames_since_throw = normalize_throw_recovery(popo_state)

    # --- Nana Sync (4 dims) ---
    nana_distance = distance(nana_state.position, get_opponent(popo_state).position)
    nana_actionable = is_actionable?(nana_state)
    nana_in_desync = different_action_category?(popo_state, nana_state)
    nana_facing_opponent = facing_toward?(nana_state, get_opponent(popo_state))

    # --- Situation Flags (4 dims) ---
    regrab_possible = opponent_in_grab and nana_actionable and nana_distance < @regrab_range
    handoff_window = detect_handoff_window(popo_state, nana_state)
    wobble_ready = popo_grabbing and not nana_in_desync and get_opponent(popo_state).percent > 0
    tech_chase = is_tech_chase_situation?(popo_state, nana_state)

    # --- Timing (2 dims) ---
    nana_action_frame = normalize_action_frame(nana_state)
    opponent_hitstun = normalize_hitstun(get_opponent(popo_state))

    Nx.tensor([
      # Grab state
      bool_to_float(popo_grabbing),
      grab_timer,
      bool_to_float(opponent_in_grab),
      frames_since_throw,
      # Nana sync
      normalize_distance(nana_distance),
      bool_to_float(nana_actionable),
      bool_to_float(nana_in_desync),
      bool_to_float(nana_facing_opponent),
      # Situations
      bool_to_float(regrab_possible),
      bool_to_float(handoff_window),
      bool_to_float(wobble_ready),
      bool_to_float(tech_chase),
      # Timing
      nana_action_frame,
      opponent_hitstun
    ], type: :f32)
  end
end
```

### Minimal Feature Breakdown

| Category | Feature | Dims | Why It Matters |
|----------|---------|------|----------------|
| **Grab** | popo_grabbing | 1 | Core state detection |
| | grab_timer | 1 | Pummel/throw timing |
| | opponent_in_grab | 1 | Confirms grab success |
| | frames_since_throw | 1 | Regrab timing |
| **Nana** | nana_distance | 1 | Regrab/handoff range |
| | nana_actionable | 1 | Can Nana act? |
| | nana_in_desync | 1 | Desync setup active |
| | nana_facing | 1 | Correct orientation |
| **Situation** | regrab_possible | 1 | Compound detection |
| | handoff_window | 1 | Frame-perfect window |
| | wobble_ready | 1 | Infinite possible |
| | tech_chase | 1 | Ground tech situation |
| **Timing** | nana_action_frame | 1 | Hitbox timing |
| | opponent_hitstun | 1 | Escape window |
| **Total** | | **14** | |

---

## Option B: Full Implementation (32 dims)

Complete IC tech representation for maximum precision.

```elixir
defmodule ExPhil.Embeddings.ICTechFull do
  @moduledoc """
  Full IC tech features (32 dims).
  Comprehensive representation for IC specialists.
  """

  @spec embed(map(), map(), map()) :: Nx.Tensor.t()
  def embed(popo_state, nana_state, opponent_state) do
    Nx.concatenate([
      grab_features(popo_state, opponent_state),      # 8 dims
      opponent_grab_features(opponent_state),          # 8 dims
      nana_sync_features(popo_state, nana_state, opponent_state), # 8 dims
      situation_features(popo_state, nana_state, opponent_state)  # 8 dims
    ])
  end

  # --- Grab State (8 dims) ---
  defp grab_features(popo, opponent) do
    Nx.tensor([
      bool_to_float(is_grab_action?(popo.action_state)),      # grabbing?
      normalize_grab_timer(popo),                              # grab duration
      bool_to_float(is_pummel_action?(popo.action_state)),    # pummeling?
      encode_throw_type(popo.action_state),                    # 0=none, 1=u, 2=d, 3=f, 4=b (normalized)
      normalize_throw_recovery(popo),                          # frames since throw
      bool_to_float(can_regrab?(popo)),                       # regrab actionable?
      popo.position.x - opponent.position.x |> normalize_pos(), # relative X
      popo.position.y - opponent.position.y |> normalize_pos()  # relative Y
    ])
  end

  # --- Opponent in Grab (8 dims) ---
  defp opponent_grab_features(opponent) do
    Nx.tensor([
      bool_to_float(is_grabbed_action?(opponent.action_state)), # in grab?
      normalize_hitstun(opponent),                               # hitstun remaining
      opponent.percent / 300.0,                                  # percent (for kill confirms)
      opponent.di_x || 0.0,                                      # DI X (if available)
      opponent.di_y || 0.0,                                      # DI Y
      bool_to_float(can_mash_out?(opponent)),                   # escape possible?
      normalize_actionable_frames(opponent),                     # frames until actionable
      bool_to_float(is_on_ground?(opponent))                    # grounded?
    ])
  end

  # --- Nana Synchronization (8 dims) ---
  defp nana_sync_features(popo, nana, opponent) do
    nana_to_opp = distance(nana.position, opponent.position)

    Nx.tensor([
      normalize_action_frame(nana),                              # action frame (hitbox timing!)
      normalize_distance(nana_to_opp),                           # distance to opponent
      bool_to_float(in_regrab_range?(nana_to_opp)),             # can regrab?
      frames_until_hitbox(nana) |> normalize_frames(),           # dair/ftilt timing
      bool_to_float(is_desynced?(popo, nana)),                  # desync active?
      bool_to_float(facing_toward?(nana, opponent)),            # facing right way?
      bool_to_float(is_actionable?(nana)),                      # can act?
      nana.hitstun_remaining |> normalize_frames()               # nana stunned?
    ])
  end

  # --- Situation Detection (8 dims) ---
  defp situation_features(popo, nana, opponent) do
    Nx.tensor([
      bool_to_float(is_guaranteed_regrab?(popo, nana, opponent)), # guaranteed regrab
      bool_to_float(is_tech_chase?(popo, nana, opponent)),        # tech chase
      bool_to_float(is_wobble_ready?(popo, nana, opponent)),      # wobble possible
      bool_to_float(is_handoff_possible?(popo, nana, opponent)),  # handoff window
      normalize_setup_duration(popo),                              # frames in current setup
      optimal_nana_offset_x(popo, nana, opponent),                # where should Nana be? X
      optimal_nana_offset_y(popo, nana, opponent),                # where should Nana be? Y
      0.0  # padding for alignment
    ])
  end
end
```

### Full Feature Breakdown

| Block | Features | Dims | Purpose |
|-------|----------|------|---------|
| **Grab State** | grabbing, timer, pummel, throw_type, recovery, can_regrab, rel_pos | 8 | Complete grab info |
| **Opponent** | in_grab, hitstun, percent, DI, mash, actionable, grounded | 8 | Opponent state in grab |
| **Nana Sync** | action_frame, distance, range, hitbox_frames, desync, facing, actionable, stunned | 8 | Nana coordination |
| **Situations** | regrab, tech_chase, wobble, handoff, duration, optimal_pos | 8 | High-level detection |
| **Total** | | **32** | |

---

## Implementation Plan

### Phase 1: Core Infrastructure

1. **Create `lib/exphil/embeddings/ic_tech.ex`**
   - Implement minimal 14-dim version first
   - Helper functions for action state detection

2. **Action State Constants**
   ```elixir
   # Grab-related action states (from libmelee)
   @grab_actions [0xD4, 0xD5, 0xD6, ...]  # CATCH, CATCH_PULL, CATCH_WAIT
   @grabbed_actions [0xDF, 0xE0, ...]      # CAPTURE_PULLED, CAPTURE_WAIT
   @throw_actions %{
     up: 0xE4,    # THROW_UP
     down: 0xE5,  # THROW_DOWN
     forward: 0xE6,
     back: 0xE7
   }
   ```

3. **Integrate into GameEmbed**
   ```elixir
   def embed(game_state, opts \\ []) do
     base = embed_base(game_state, opts)

     # Conditionally add IC features
     if opts[:ic_tech] and is_ice_climbers?(game_state.player1) do
       ic_features = ICTech.embed(game_state.player1, game_state.player1.nana)
       Nx.concatenate([base, ic_features])
     else
       base
     end
   end
   ```

### Phase 2: Training Integration

4. **CLI flag: `--ic-tech`**
   - Enables IC tech features when training on IC replays
   - Auto-detected from character if `--train-character ice_climbers`

5. **Conditional embedding size**
   - Base embedding + 14/32 dims when IC
   - Policy network must handle variable input size OR pad non-IC

### Phase 3: Validation

6. **Unit tests**
   - Test each situation detector
   - Test with known grab sequences from replays

7. **Integration test**
   - Train small model on IC replays with/without IC tech
   - Compare grab game accuracy

---

## Key Helper Functions Needed

```elixir
# Action state detection
def is_grab_action?(action_state)
def is_grabbed_action?(action_state)
def is_pummel_action?(action_state)
def is_throw_action?(action_state)

# Timing calculations
def normalize_grab_timer(player_state)
def frames_until_hitbox(nana_state)
def normalize_hitstun(player_state)

# Situation detection
def is_guaranteed_regrab?(popo, nana, opponent)
def is_handoff_possible?(popo, nana, opponent)
def is_wobble_ready?(popo, nana, opponent)
def is_desynced?(popo, nana)

# Geometric
def distance(pos1, pos2)
def facing_toward?(player, target)
def in_regrab_range?(distance)
```

---

## Decision Points

### 1. Character-Conditional vs Always-On

**Option A: Conditional (recommended)**
- Only add IC dims when player is IC
- Requires variable embedding size handling
- Most memory efficient

**Option B: Always-On with Zeros**
- Always include IC dims, zero when not IC
- Fixed embedding size (simpler)
- Wastes dims for non-IC

### 2. Minimal vs Full

**Minimal (14 dims):** Start here
- Covers core grab game and Nana sync
- Quick to implement and validate
- Can expand later

**Full (32 dims):** Future expansion
- Complete representation
- Includes opponent DI, optimal positioning
- For IC specialists

### 3. Where to Compute

**Option A: Embedding time**
- Computed once per frame during preprocessing
- Cached with other embeddings
- Most efficient

**Option B: Runtime**
- Computed during training/inference
- Allows dynamic thresholds
- More flexible but slower

---

## References

- [IC Handbook](https://smashboards.com/threads/ice-climbers-handbook.298862/) - Frame data, techniques
- [Wobbling Frame Data](https://www.ssbwiki.com/Wobbling) - Timing windows
- [libmelee Action States](https://github.com/altf4/libmelee/blob/master/melee/enums.py) - Action state IDs

---

## Next Steps

1. [ ] Review this design doc
2. [ ] Decide: Minimal (14) or Full (32)?
3. [ ] Decide: Conditional or Always-On?
4. [ ] Implement core helper functions
5. [ ] Add to GameEmbed with `--ic-tech` flag
6. [ ] Write tests
7. [ ] Train comparison model
