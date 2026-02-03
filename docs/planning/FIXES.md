# ExPhil Technical Debt & Improvements

This document tracks identified issues and planned improvements from the January 2026 codebase review.

## Priority Legend

- **P0**: Critical - fix immediately (security, data loss risk)
- **P1**: High - fix soon (maintainability blockers)
- **P2**: Medium - fix when possible (quality improvements)
- **P3**: Low - nice to have (polish)

---

## 1. Architecture & Module Organization

### 1.1 Giant Modules Need Decomposition [P1]

| Module | Lines | Issue |
|--------|-------|-------|
| `config.ex` | 3,216 | Mixing parsing, validation, presets, YAML, smart inference |
| `policy.ex` | 2,060 | Backbone selection, sampling, loss computation mixed |
| `imitation.ex` | 1,801 | Training loop, metrics, checkpointing mixed |
| `game.ex` (embedding) | 1,156 | All embedding logic in one file |
| `player.ex` (embedding) | 1,290 | All embedding logic in one file |

**config.ex Split Plan:**
```
lib/exphil/training/config/
├── parser.ex          # CLI argument parsing
├── validator.ex       # Validation rules and error collection
├── presets.ex         # Training presets (quick, standard, production, etc.)
├── defaults.ex        # Default values and env var handling
├── yaml.ex            # YAML loading and conversion
├── inference.ex       # Smart flag inference (infer_smart_defaults)
├── json_builder.ex    # Config JSON for model provenance
└── config.ex          # Main module, delegates to submodules
```

**policy.ex Split Plan:**
```
lib/exphil/networks/policy/
├── backbone.ex        # Backbone selection and building
├── heads.ex           # Controller output heads (buttons, sticks, shoulder)
├── sampling.ex        # Action sampling logic
├── loss.ex            # Loss computation (cross-entropy, focal)
├── embeddings.ex      # Action/character embedding handling
└── policy.ex          # Main module, orchestrates above
```

**imitation.ex Split Plan:**
```
lib/exphil/training/imitation/
├── train_loop.ex      # Main epoch/batch loop
├── metrics.ex         # Loss tracking, accuracy computation
├── checkpoint.ex      # Saving/loading/pruning
├── validation.ex      # Validation pass logic
├── progress.ex        # Progress display and logging
└── imitation.ex       # Main module, orchestrates above
```

### 1.2 Multiple Mamba Implementations [P2]

Five separate Mamba variants with no shared abstraction:
- `mamba.ex` - Standard Mamba
- `mamba_ssd.ex` - SSD variant
- `mamba_nif.ex` - NIF-accelerated
- `mamba_cumsum.ex` - Cumsum-based scan
- `mamba_hillis_steele.ex` - Parallel scan variant

**Plan:** Create `MambaBehaviour` module defining common interface, extract shared logic into helpers.

### 1.3 Embedding Logic Duplication [P2]

Three Nana embedding functions share ~60% code:
- `embed_batch_nana_compact/2`
- `embed_batch_nana_enhanced/2`
- `embed_batch_nana_full/2`

**Plan:** Extract common `extract_nana_values/1` and `compute_nana_flags/1` helpers.

---

## 2. Code Quality & Patterns

### 2.1 Unsafe `String.to_atom` Usage [P0] - FIXED

Multiple locations create atoms from untrusted input before validation:

| File | Line | Context |
|------|------|---------|
| `config.ex` | 537 | `maybe_env_preset` - atom created before validation |
| `config.ex` | 757 | `normalize_key` - YAML key conversion |
| `config.ex` | 763-776 | `convert_value` - YAML value conversion |
| `config.ex` | 1332 | `preset/1` string clause |
| `config.ex` | 2228, 2247, 2266, 2285 | Mode parsing fallbacks |
| `config.ex` | 2687 | `parse_atom_arg` |

**Fix:** Created `ExPhil.Training.Config.AtomSafety` module with:
- `safe_to_atom/2` - converts with allowlist validation
- `safe_to_existing_atom/1` - uses `String.to_existing_atom` with fallback

### 2.2 Magic Numbers Scattered [P1] - FIXED

| Value | Meaning | Occurrences |
|-------|---------|-------------|
| `28800` | 8 minutes in frames (60fps × 60s × 8min) | 5 places |
| `399` | Number of Melee action states | 3 places |
| `120` | Max hitstun frames | 2 places |
| `60` | Frames per second / action duration | 2+ places |
| `180` | Link bomb timer | 1 place |

**Fix:** Created `ExPhil.Constants` module with all game constants.

### 2.3 Inconsistent Defaults [P1] - TRACKING

```elixir
# config.ex line 23
@default_hidden_sizes [512, 256]

# policy.ex line 77
@default_hidden_sizes [512, 512]  # Different!

# value.ex line 45
@default_hidden_sizes [512, 512]  # Same as policy, different from config
```

**Resolution:** `config.ex` is authoritative for training CLI. Policy/Value use their own defaults for direct API usage. Document this distinction.

### 2.4 Silent Failures & Missing Error Context [P2]

- `ReplayParser.load_parsed` uses `File.read!()` in try block
- Config YAML parsing errors return generic `{:error, reason}`
- Directory listing failures return empty list instead of error

**Plan:** Add structured error types with context.

---

## 3. Testing Gaps

### 3.1 Missing Negative Path Tests [P1] - IN PROGRESS

| Area | Missing Tests |
|------|---------------|
| Config | Invalid YAML, malformed presets, bad env vars |
| Config | Interdependent flag validation (batch_size > num_frames) |
| Data | Empty datasets, single-frame files, corrupted replays |
| Embeddings | NaN/infinity detection and handling |

**Added Tests:**
- [ ] `config_validation_test.exs` - Invalid inputs, edge cases
- [ ] Config YAML malformed input
- [ ] Config invalid preset names
- [ ] Config bad environment variable values

### 3.2 Skipped Integration Tests [P2]

- `dolphin_self_play_test.exs` is marked `:skip`
- No comparative tests between Nana embedding modes
- No comparative tests between Mamba variants

### 3.3 Property-Based Testing Underutilized [P3]

`stream_data` dependency exists but underutilized. Good candidates:
- Embedding shape invariants across batch sizes
- Action ID round-trip encoding/decoding
- Config parsing determinism

---

## 4. Performance Issues

### 4.1 TODOs Left in Production Code [P2]

| File | Line | TODO |
|------|------|------|
| `mamba_ssd.ex` | 286 | "Optimize by using precomputed chunk_a_products" |
| `player.ex` | 1067 | "Get real action_frame from libmelee if available" |
| `policy.ex` | 714 | "Use true Mamba when implemented" |

**Plan:** Create GitHub issues to track these.

### 4.2 List Access in Hot Paths [P3]

`data.ex:897-898` uses `Enum.at` which is O(n) for small datasets. The optimization path exists for large datasets but threshold may be too high.

---

## 5. Documentation Drift

### 5.1 Outdated References [P2]

- Docs reference "py-slippi" but implementation uses Peppi (Rust)
- README says "Stage 2: RL Coming Soon" but PPO/self-play code exists
- GOTCHAS.md is 81KB - needs pruning

### 5.2 Missing Documentation [P2]

- No "how to add a new CLI flag" guide
- No clear interface documentation for Mamba variants
- Minimal API docs for `Data.ex`

---

## 6. Configuration Issues

### 6.1 YAML Conversion Incomplete [P2]

`convert_yaml_map` only handles 6 value types. Unknown types silently pass through.

**Plan:** Add exhaustive type handling with warnings for unknown types.

### 6.2 Environment Variable Coupling [P3]

Scripts mutate `XLA_FLAGS`, `EXLA_TARGET` at startup without conflict detection.

---

## 7. Script Inconsistencies

### 7.1 Different Patterns Across 33 Scripts [P2]

| Issue | Scripts Affected |
|-------|------------------|
| Different arg parsing (`Config` vs `CLI` vs `OptionParser`) | ~10 |
| Inline format functions instead of `Training.Output` | 3 |
| Inconsistent `Logger.configure` timing | ~5 |

**Plan:** Create shared script template in `lib/exphil/script_utils.ex`.

### 7.2 Duplicate Formatting Functions [P2]

`train_from_replays.exs` defines:
- `format_time_per_it/1`
- `_format_float/2`
- `_format_fixed/2`

These should be in `Training.Output`.

---

## 8. Dependency Concerns

### 8.1 GitHub Branch Override [P2]

```elixir
{:axon_onnx, github: "blasphemetheus/axon_onnx", branch: "runtime-fixes", override: true}
```

Tracking branch without version pin. Add comment with upstream PR link.

### 8.2 Optional Dependencies Not Guarded [P3]

`{:pythonx, "~> 0.3", optional: true}` - code doesn't check if loaded.

### 8.3 Flash Attention NIF Warnings [P3]

The Rust NIF in `native/flash_attention_nif/` has unused code warnings:
- `build.rs`: Unused imports (std::env, std::path::PathBuf, std::process::Command)
- `src/lib.rs`: Unused CUDA_AVAILABLE and CUDA_CHECKED statics

**Fix:** Run `cargo fix --lib -p flash_attention_nif` or remove unused code.

---

## Implementation Progress

### Completed (2026-01-31)

- [x] Created `ExPhil.Constants` module for magic numbers (`lib/exphil/constants.ex`)
  - Frame timing: `fps()`, `max_game_frames()`, `online_frame_delay()`
  - Actions: `num_actions()`, character/stage counts
  - Combat: hitstun, shieldstun, action frame constants
  - Network defaults: `default_hidden_sizes()`, `tensor_alignment()`
  - Normalization helpers: `normalize_frame/1`, `normalize_hitstun/1`

- [x] Created `ExPhil.Training.Config.AtomSafety` for safe atom conversion
  - `safe_to_atom/2` - converts with allowlist validation
  - `safe_to_atom_downcase/2` - case-insensitive conversion
  - `safe_to_existing_atom/1` - uses existing atoms only
  - `validate/2` - validates both atoms and strings

- [x] Added negative path tests for config validation (60+ new test cases)
  - YAML parsing: malformed YAML, empty, non-map, scalar inputs
  - Arg validation: typo detection, suggestions
  - Frame delay range validation
  - Optimizer validation
  - Regularization parameter bounds
  - Cosine restart validation
  - Gradient clipping validation
  - Streaming options validation
  - Smart defaults inference tests

- [x] Created comprehensive FIXES.md tracking document

### Completed (2026-02-03)

- [x] Integrated `ExPhil.Constants` module across entire codebase
  - `player.ex` - Replaced magic numbers (399, 33, 120, 60) with Constants functions
  - `primitives.ex` - Module attributes now source from Constants (action_size, character_size, stage_size, max_jumps)
  - `policy.ex` - Already using Constants via module attributes (@num_actions, @num_characters)
  - `config.ex` - Frame delay defaults use `Constants.online_frame_delay()`
  - `data.ex` - Frame delay defaults use `Constants.online_frame_delay()`

- [x] AtomSafety already integrated into config.ex (verified via grep - uses allowlist validation)

- [x] Fixed Range.new/2 deprecation warning in `config.ex:3220` (rotate_backups function)
  - Changed `(count - 2)..0` to `(count - 2)..0//-1` for explicit step direction

### Completed (config.ex decomposition)

- [x] Split `config.ex` into submodules (**60% reduction: 3,270 → 1,297 lines**)
  - [x] `config/parser.ex` (652 lines) - CLI argument parsing with context pattern
  - [x] `config/presets.ex` (617 lines) - Training preset definitions
  - [x] `config/validator.ex` (475 lines) - Validation rules with context passing
  - [x] `config/yaml.ex` (258 lines) - YAML loading/saving with context map pattern
  - [x] `config/inference.ex` (199 lines) - Smart flag inference logic
  - [x] `config/atom_safety.ex` (188 lines) - Safe atom conversion with allowlists
  - [x] `config/checkpoint.ex` (172 lines) - Checkpoint safety and backup rotation
  - [x] `config/diff.ex` (120 lines) - Config diff display using function reference pattern

Total: 8 submodules extracted, main config.ex now serves as orchestration facade.

### Completed (policy.ex decomposition - 2026-02-03)

- [x] Split `policy.ex` into submodules (**74% reduction: 2,060 → 544 lines**)
  - [x] `policy/backbone.ex` (731 lines) - All temporal/non-temporal backbone builders
  - [x] `policy/loss.ex` (429 lines) - BCE, CE, focal, weighted loss functions
  - [x] `policy/embeddings.ex` (450 lines) - Action/character embedding preprocessing
  - [x] `policy/heads.ex` (297 lines) - Controller output heads
  - [x] `policy/sampling.ex` (198 lines) - Action sampling functions

Total: 5 submodules extracted, main policy.ex delegates to submodules.

### Planned

**imitation.ex maintainability plan**:
1. Extract epoch/batch loop to `imitation/train_loop.ex`
2. Extract metrics tracking to `imitation/metrics.ex`
3. Extract checkpoint logic to `imitation/checkpoint.ex`
4. Extract validation pass to `imitation/validation.ex`
5. Extract progress display to `imitation/progress.ex`
6. Keep main `imitation.ex` with `train/3` as entry point

**Other planned work**:
- [ ] Unify Mamba implementations with shared behavior
- [ ] Extract Nana embedding helpers
- [ ] Add property-based tests
- [ ] Update outdated documentation
- [ ] Create shared script template

---

## Additional Ideas for Future Consideration

### Architecture Improvements

1. **Typed Config Struct** - Replace keyword list opts with typed struct for compile-time safety
2. **Config Schema DSL** - Declarative flag definitions instead of manual parsing
3. **Plugin System for Backbones** - Dynamic backbone registration
4. **Telemetry Integration** - Replace ad-hoc timing with `:telemetry` events

### Testing Improvements

1. **Mutation Testing** - Use `muzak` or similar for test quality
2. **Benchmark Regression CI** - Track performance across commits
3. **Contract Tests** - Define and test module interfaces

### Developer Experience

1. **Interactive Config Validator** - Web UI for config exploration
2. **Training Dashboard** - Real-time metrics beyond wandb
3. **Model Diff Tool** - Compare trained models architecturally

### Performance

1. **ONNX Graph Optimization** - Fuse ops for faster inference
2. **Batched Embedding Cache** - Pre-batch cached embeddings
3. **Async Checkpoint Compression** - Compress checkpoints in background

---

*Last updated: 2026-02-03*
*Review triggered by: Codebase critical analysis*
*config.ex decomposition completed: 2026-02-03*
