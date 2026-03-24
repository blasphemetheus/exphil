# HuggingFace Dataset → ExPhil Embedding Mapping

Mapping between erickfm's processed Melee datasets and ExPhil's embedding pipeline.
Schema verified against actual `norm_stats.json` and `cat_maps.json` from mimic-melee-subset.

**Source datasets:**
- Raw .slp: [slippi-public-dataset-v3.7](https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7) (~95k replays, 200GB)
- Pre-processed PyTorch shards: [mimic-melee](https://huggingface.co/datasets/erickfm/mimic-melee) (1.81B frames, 2.59TB)
- Subset: [mimic-melee-subset](https://huggingface.co/datasets/erickfm/mimic-melee-subset) (18.7M frames, 26.7GB)
- Extractor tool: [slippi-frame-extractor](https://github.com/erickfm/slippi-frame-extractor)

**Note:** The Discord message called these "frame-melee" but the actual HF repos are named "mimic-melee".

## Format Summary

| Dataset | Format | Preprocessing | Best Use |
|---------|--------|---------------|----------|
| slippi-public-dataset-v3.7 | Raw .slp | None | Feed into Peppi → existing pipeline |
| mimic-melee | PyTorch .pt shards | Z-score normalized, K-means sticks, categoricals encoded | Viable second path (see below) |
| slippi-frame-extractor output | Per-game parquet | Raw values, no normalization | Third option: run extractor ourselves |

## mimic-melee .pt Shard Structure

```python
{
    "states": {feature_name: Tensor},   # game-state features (z-score normalized)
    "targets": {head_name: Tensor},     # controller-input targets (K-means discretized)
    "offsets": [int, ...],              # game boundary indices along time axis
    "n_games": int,                     # number of games in this shard
}
```

## Complete Feature Schema (from norm_stats.json)

Verified from actual metadata. Format: `[mean, std]` for z-score normalization.
Fields with `[0.0, 1.0]` are boolean/sparse (not z-scored).

### Stage & Environment (19 continuous features)

| Feature | mean | std | ExPhil Equivalent | Notes |
|---------|------|-----|-------------------|-------|
| `frame` | 5059 | 3327 | `GameState.frame` | Normalized ÷28800 in ExPhil |
| `distance` | 47.8 | 41.6 | `GameState.distance` | ExPhil: ÷200, clamp [0,1.5] |
| `stage_edge_left` | -71.7 | 1.85 | — | **New**: stage edge geometry |
| `stage_edge_right` | 71.7 | 1.85 | — | **New** |
| `blastzone_bottom` | -109.4 | 2.96 | — | **New**: kill boundaries |
| `blastzone_left` | -225.4 | 6.47 | — | **New** |
| `blastzone_right` | 225.4 | 6.47 | — | **New** |
| `blastzone_top` | 202.3 | 10.4 | — | **New** |
| `left_platform_height` | 27.3 | 0.61 | — | **New**: platform geometry |
| `left_platform_left` | -57.8 | 0.79 | — | **New** |
| `left_platform_right` | -20.5 | 2.45 | — | **New** |
| `right_platform_height` | 27.3 | 0.63 | — | **New** |
| `right_platform_left` | 20.5 | 2.44 | — | **New** |
| `right_platform_right` | 57.8 | 1.14 | — | **New** |
| `top_platform_height` | 54.3 | 0.62 | — | **New** |
| `top_platform_left` | -18.8 | 0.05 | — | **New** |
| `top_platform_right` | 18.8 | 0.05 | — | **New** |
| `randall_height` | 0.0 | 1.0 | — | **New**: Yoshi's moving cloud |
| `randall_left` | 0.0 | 1.0 | — | **New** |
| `randall_right` | 0.0 | 1.0 | — | **New** |

### Per Player (self_* and opp_* — 2 × 25 = 50 continuous features)

| Feature | mean | std | ExPhil Equivalent | Match |
|---------|------|-----|-------------------|-------|
| `pos_x` | 1.13 | 55.0 | `Player.x` | exact |
| `pos_y` | 13.6 | 32.4 | `Player.y` | exact |
| `percent` | 52.6 | 42.4 | `Player.percent` | exact |
| `stock` | 2.74 | 1.07 | `Player.stock` | exact |
| `shield_strength` | 58.7 | 3.81 | `Player.shield_strength` | exact |
| `jumps_left` | 1.42 | 0.97 | `Player.jumps_left` | exact |
| `action_frame` | 12.6 | 16.8 | `Player.action_frame` | exact |
| `hitstun_left` | 6.31 | 22.0 | `Player.hitstun_frames_left` | exact |
| `speed_air_x_self` | 0.0 | 1.0 | `Player.speed_air_x_self` | **exact** |
| `speed_ground_x_self` | 0.0 | 1.0 | `Player.speed_ground_x_self` | **exact** |
| `speed_y_self` | 0.0 | 1.0 | `Player.speed_y_self` | **exact** |
| `speed_x_attack` | 0.0 | 1.0 | `Player.speed_x_attack` | **exact** |
| `speed_y_attack` | 0.0 | 1.0 | `Player.speed_y_attack` | **exact** |
| `main_x` | 0.50 | 0.32 | `Controller.main_stick.x` | range: [0,1] center=0.5, same as ExPhil |
| `main_y` | 0.46 | 0.21 | `Controller.main_stick.y` | same range |
| `l_shldr` | 0.17 | 0.37 | `Controller.l_shoulder` | exact |
| `r_shldr` | 0.17 | 0.37 | `Controller.r_shoulder` | ExPhil stores but doesn't embed |
| `hitlag_left` | 0.0 | 1.0 | — | **New**: ExPhil doesn't extract |
| `invuln_left` | 0.0 | 1.0 | — | **New**: ExPhil only has boolean |
| ECBs (8 fields) | 0.0 | 1.0 | — | **New**: collision box x/y × 4 |

### Per Player Nana (self_nana_* and opp_nana_* — 2 × 25 = 50 features)

Same field set as player state, prefixed with `nana_`. ExPhil uses compact Nana (39 dims)
which computes derived features (is_attacking, is_grabbing, can_act, is_synced) not present here.

### Projectiles (proj0-7 — 8 × 5 = 40 continuous features)

| Feature | ExPhil Equivalent | Notes |
|---------|-------------------|-------|
| `proj{N}_frame` | — | Projectile lifetime (new for ExPhil) |
| `proj{N}_pos_x` | `Projectile.x` | exact |
| `proj{N}_pos_y` | `Projectile.y` | exact |
| `proj{N}_speed_x` | `Projectile.speed_x` | exact |
| `proj{N}_speed_y` | `Projectile.speed_y` | exact |

### Categorical Features (from cat_maps.json)

| Feature | Categories | ExPhil Equivalent | Notes |
|---------|-----------|-------------------|-------|
| `self_port` | {0, -1} | — | Not used in ExPhil embeddings |
| `opp_port` | {0, -1} | — | Not used |
| `self_costume` | {0-5} | — | Cosmetic, not useful |
| `opp_costume` | {0-5} | — | Cosmetic |
| `proj{N}_owner` | {0, -1} | `Projectile.owner` | Direct |
| `proj{N}_subtype` | {0, -1} | `Projectile.subtype` | Direct |

**Missing from cat_maps but likely in states dict:** `character`, `action`, `stage` —
these are probably separate tensor keys (not z-scored, not in norm_stats).
Need to inspect actual .pt shard to confirm.

### Target Heads (controller outputs)

| Head | Encoding | ExPhil Equivalent |
|------|----------|-------------------|
| Main stick | 60 K-means clusters (from stick_clusters.json) | 17-bucket uniform discretization |
| C-stick | 5-way cardinal (neutral/up/down/left/right) | 17-bucket uniform discretization |
| L/R triggers | 4 bins (from stick_clusters.json) | 5-bucket uniform discretization |
| Buttons | Per-button binary | Per-button binary (same) |

Note: MIMIC uses 60 K-means stick clusters (stick_centers in [0,1]² space);
ExPhil uses 17 uniform buckets per axis. The [0,1] range with 0.5=center is the same
convention in both — **no range shift needed**.

## Key Corrections from Schema Inspection

### Speed decomposition: NO GAP

The initial analysis incorrectly claimed the dataset only had 3 generic speed fields.
The actual `norm_stats.json` confirms **all 5 decomposed speed fields are present**:
- `speed_air_x_self`, `speed_ground_x_self`, `speed_y_self`, `speed_x_attack`, `speed_y_attack`

These match ExPhil's Peppi output exactly.

### Stick ranges: NO SHIFT NEEDED

The stick values use [0, 1] with 0.5 = center (visible from `main_x` mean ≈ 0.50),
which is the same convention ExPhil uses. No range transformation required.

### mimic-melee shards are more usable than initially assessed

Since the shards contain:
- All 5 decomposed speeds (matching ExPhil)
- Stick values in the same [0,1] range
- Game boundaries via `offsets` array
- Both player perspectives

The main adaptation needed is:
1. Un-normalize continuous features (multiply by std, add mean, using norm_stats.json)
2. Decode categoricals (using cat_maps.json)
3. Re-encode with ExPhil's embedding pipeline (learned vs one-hot, ExPhil's own scaling)
4. Recover self-controller inputs for target labels (excluded from states — this is the real blocker)

## Summary: What's Directly Usable

### Fields that map 1:1 (no transform needed after un-normalization)
- Position (x, y), percent, stock, shield_strength, jumps_left
- All 5 speed components (air_x, ground_x, y_self, x_attack, y_attack)
- action_frame, hitstun_left
- Stick inputs (main_x/y), shoulder (l_shldr)
- Projectile positions and velocities
- Distance, frame

### Fields available as enrichment (ExPhil doesn't currently use)
- Stage geometry: blastzones, edges, platforms (19 dims)
- ECBs: collision box coordinates (16 dims per player)
- hitlag_left, invuln_left (frame counts vs ExPhil's booleans)
- Randall position (Yoshi's Story)
- Full Nana state (vs ExPhil's compact 39-dim)

### Fields missing from mimic-melee (ExPhil has)
- `facing`, `on_ground`, `invulnerable` — boolean flags, may be in states dict as separate tensors (not in norm_stats since they're not continuous)
- `character`, `action`, `stage` — likely separate categorical tensors
- Player name/tag — not available (MIMIC doesn't use this)
- Relative position (dx, dy) — trivially computed from pos_x/pos_y
- Ledge distance — trivially computed
- Previous action — shift by 1 frame in targets

## Recommended Integration Path (Updated)

### Phase 1: Raw .slp (immediate, no new code)
Download .slp files → feed into existing Peppi pipeline. ~95k replays.

### Phase 2: mimic-melee .pt shards (moderate effort)
Now viable since speeds match and sticks don't need shifting. Steps:
1. Write PyTorch .pt reader (Python script or Rust NIF to extract tensors → Nx)
2. Un-normalize with norm_stats.json
3. Map to ExPhil embedding format
4. **Blocker:** Self-controller inputs excluded from states — need targets dict
   for buttons, or fall back to .slp for controller ground truth

### Phase 3: Run slippi-frame-extractor ourselves (cleanest second path)
Run extractor on .slp files → get raw parquet with ALL fields (no normalization, no exclusions).
Then load with Explorer. This gives full control and no un-normalization needed.
