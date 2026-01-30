# Hex Publishing Readiness

This document tracks progress toward publishing ExPhil on Hex.pm.

## Completed

### @spec Coverage (Jan 2026)
- [x] Bridge modules (MeleePort, AsyncRunner) - Added types and specs
- [x] Agents modules (Agent, Supervisor) - Full spec coverage
- [x] Training.Config - Key public API specs
- [x] Training.Imitation - Already had 100% coverage
- [x] Embeddings modules - Already had 100% coverage

### ExDoc Configuration (Jan 2026)
- [x] Added `docs/0` function to mix.exs
- [x] Grouped extras by category (Getting Started, Core Guides, etc.)
- [x] Grouped modules by domain (Training, Embeddings, Networks, etc.)
- [x] Enabled `nest_modules_by_prefix` for cleaner sidebar

### Top-Level Module (Jan 2026)
- [x] Expanded `ExPhil` moduledoc with comprehensive overview
- [x] Installation instructions
- [x] Quick start examples
- [x] Architecture diagram
- [x] Module overview with links
- [x] Convenience functions with @doc and @spec

## Remaining Tasks

### High Priority

#### 1. Add @doc to Undocumented Functions
Some modules have @spec but sparse @doc. Priority modules:

```bash
# Find functions with @spec but no @doc
grep -B2 "@spec" lib/**/*.ex | grep -v "@doc" | head -30
```

Key modules to document:
- `ExPhil.Embeddings.Primitives` - Low-level embedding utilities
- `ExPhil.Networks.FusedOps` - Custom CUDA operations
- `ExPhil.Training.Streaming` - Memory-efficient data loading

#### 2. Add Doctests to Key Modules
Doctests serve as both documentation and tests. Priority:

```elixir
# Example doctest format
@doc """
Embed a game state to a tensor.

## Examples

    iex> game_state = ExPhil.Bridge.GameState.dummy()
    iex> tensor = ExPhil.Embeddings.Game.embed(game_state, nil, 1)
    iex> Nx.shape(tensor)
    {288}

"""
```

Modules to add doctests:
- `ExPhil.Embeddings.Game` - embed/4, embed_batch/2
- `ExPhil.Networks.Policy` - build/1, sample/4
- `ExPhil.Training.Config` - parse_args/1, preset/1

#### 3. Fix ExDoc Warnings
Current `mix docs` produces warnings about:
- Missing `ExPhil.Data.Peppi.*` type references
- References to non-existent guide files in GOALS.md

### Medium Priority

#### 4. Create Cheatsheet
ExDoc supports `.cheatmd` files that render as quick-reference cards.

Create `docs/cheatsheets/training.cheatmd`:
```markdown
# Training Cheatsheet

## Presets
{: .col-2}

### Quick Test
```bash
mix run scripts/train_from_replays.exs --preset quick
```

### Production
```bash
mix run scripts/train_from_replays.exs --preset production --wandb
```

## Common Flags
{: .col-2}

| Flag | Default | Description |
|------|---------|-------------|
| --epochs | 10 | Training epochs |
| --batch-size | 64 | Batch size |
| --backbone | mlp | Network type |
```

#### 5. Add Typedocs to Custom Types
Some types lack `@typedoc`:

```elixir
# Before
@type config :: %__MODULE__{...}

# After
@typedoc "Configuration for game state embedding"
@type config :: %__MODULE__{...}
```

#### 6. Improve README for Hex
Update README.md with:
- [ ] Hex.pm badge: `[![Hex.pm](https://img.shields.io/hexpm/v/exphil.svg)](https://hex.pm/packages/exphil)`
- [ ] Documentation badge: `[![Documentation](https://img.shields.io/badge/docs-hexdocs-blue.svg)](https://hexdocs.pm/exphil)`
- [ ] Minimal installation and usage example
- [ ] Link to full documentation

### Low Priority

#### 7. Add Module Diagrams
Use Mermaid diagrams in moduledocs:

```elixir
@moduledoc """
## Data Flow

```mermaid
graph LR
    A[Replays] --> B[Parser]
    B --> C[Embeddings]
    C --> D[Batches]
    D --> E[Training]
```
"""
```

#### 8. Add Livebook Notebooks
Create interactive notebooks in `notebooks/`:
- `getting_started.livemd` - Installation and first training
- `architecture_tour.livemd` - Interactive exploration
- `custom_embeddings.livemd` - How to modify embeddings

## Publishing Checklist

Before running `mix hex.publish`:

- [ ] All @moduledoc present (check with `mix docs`)
- [ ] @spec on all public functions (check with Dialyzer)
- [ ] No ExDoc warnings about missing references
- [ ] README has installation instructions
- [ ] LICENSE file present
- [ ] Version bumped in mix.exs
- [ ] CHANGELOG.md updated
- [ ] `mix test` passes
- [ ] `mix dialyzer` passes
- [ ] Documentation reviewed: `mix docs && open doc/index.html`

## Metrics

Track documentation quality:

```bash
# Count modules with @moduledoc
grep -r "@moduledoc" lib --include="*.ex" | wc -l

# Count @spec annotations
grep -r "@spec" lib --include="*.ex" | wc -l

# Count @doc annotations
grep -r "@doc" lib --include="*.ex" | wc -l

# Run ExDoc and count warnings
mix docs 2>&1 | grep "warning:" | wc -l
```

Current stats (Jan 2026):
- Modules with @moduledoc: 100%
- Functions with @spec: ~85%
- ExDoc warnings: ~20 (mostly missing Peppi types)
