# Architecture League System

The Architecture League is a competition framework where different neural network architectures compete against each other in Melee matches. It enables systematic comparison of architectures (MLP, LSTM, GRU, Mamba, Attention, Jamba) through Elo-rated matches and iterative self-play training.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Architecture League Flow                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Imitation Pretraining                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  All architectures train on same replay data to target loss        │ │
│  │  (Ensures fair starting point for competition)                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  Phase 2: Tournament Competition                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Round-robin matches between all architectures                     │ │
│  │  Elo ratings updated after each match                              │ │
│  │  Experiences collected for training                                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  Phase 3: Self-Play Evolution                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Train each architecture on collected match experiences (PPO)      │ │
│  │  Advance generation, repeat tournament                             │ │
│  │  Architectures improve through competition                         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Full league with all architectures
mix run scripts/run_league.exs \
  --replays ~/replays/mewtwo \
  --target-loss 1.0 \
  --generations 10

# Quick test with 2 architectures
mix run scripts/run_league.exs \
  --replays test/fixtures/replays \
  --architectures mlp,mamba \
  --target-loss 2.0 \
  --generations 2 \
  --matches-per-pair 5

# Generate report from existing checkpoints
mix run scripts/league_report.exs \
  --checkpoint-dir checkpoints/league
```

## Core Components

### League GenServer (`lib/exphil/league/league.ex`)

The main orchestrator that manages architecture registration, match execution, and state tracking.

```elixir
# Start a league
{:ok, _pid} = League.start_link(name: MyLeague, game_type: :mock)

# Register architectures
{:ok, entry} = ArchitectureEntry.new(id: :mamba_mewtwo, architecture: :mamba)
League.register_entry(MyLeague, entry)

# Run a tournament
{:ok, results} = League.run_tournament(MyLeague, matches_per_pair: 10)

# Get leaderboard
leaderboard = League.get_leaderboard(MyLeague)
```

**Key Functions:**

| Function | Description |
|----------|-------------|
| `start_link/1` | Start league GenServer |
| `register_entry/2` | Register an architecture |
| `run_tournament/2` | Run round-robin matches |
| `run_match/3` | Run single match between architectures |
| `get_leaderboard/2` | Get Elo-sorted rankings |
| `get_experiences/1` | Get collected match experiences |
| `advance_generation/1` | Increment generation counter |

### Architecture Entry (`lib/exphil/league/architecture_entry.ex`)

Struct representing a competing architecture with its model, parameters, and statistics.

```elixir
%ArchitectureEntry{
  id: :mamba_mewtwo,
  architecture: :mamba,        # :mlp, :lstm, :gru, :mamba, :attention, :jamba
  character: :mewtwo,
  model: %Axon{},              # Compiled Axon model
  params: %{},                 # Current weights
  generation: 5,
  elo: 1450.0,
  config: %{hidden_size: 256, state_size: 16},
  lineage: ["mamba_mewtwo_v0", "mamba_mewtwo_v1", ...],
  stats: %{wins: 42, losses: 38, draws: 5, total_frames: 180000}
}
```

**Supported Architectures:**

| Type | Description | Default Config |
|------|-------------|----------------|
| `:mlp` | Multi-layer perceptron | `hidden_sizes: [256, 256]` |
| `:lstm` | Long Short-Term Memory | `hidden_size: 256, num_layers: 2` |
| `:gru` | Gated Recurrent Unit | `hidden_size: 256, num_layers: 2` |
| `:mamba` | State Space Model | `hidden_size: 256, state_size: 16` |
| `:attention` | Transformer attention | `num_heads: 4, head_dim: 64` |
| `:jamba` | Hybrid Mamba + Attention | `attention_every: 3, num_layers: 4` |

### Match Scheduler (`lib/exphil/league/match_scheduler.ex`)

Generates tournament schedules with various strategies.

```elixir
# Round-robin (all pairs)
schedule = MatchScheduler.round_robin([:a, :b, :c, :d], matches_per_pair: 5)

# Skill-based (prefer similar Elo)
schedule = MatchScheduler.skill_based(architectures, num_matches: 100, elo_range: 100)

# Swiss rounds
rounds = MatchScheduler.swiss_rounds(architectures, num_rounds: 3)

# Single-elimination bracket
bracket = MatchScheduler.bracket(architectures, shuffle_seeds: true)

# PFSP (Prioritized Fictitious Self-Play)
opponents = MatchScheduler.pfsp(:target, candidates, num_matches: 20)
```

**Scheduling Strategies:**

| Strategy | Use Case |
|----------|----------|
| `round_robin/2` | Fair comparison, all architectures play each other |
| `skill_based/2` | Efficient matchmaking, similar skill opponents |
| `swiss_rounds/2` | Tournament format, winners play winners |
| `bracket/2` | Single elimination, quick champion determination |
| `pfsp/3` | Training optimization, target weak opponents |
| `diverse/2` | Variety in matchups for experience diversity |

### Pretraining (`lib/exphil/league/pretraining.ex`)

Imitation learning to initialize all architectures from replay data.

```elixir
# Train all architectures to target loss
{:ok, trained} = Pretraining.train_all(
  architectures,
  dataset,
  target_loss: 1.0,
  max_epochs: 50,
  batch_size: 64,
  learning_rate: 1.0e-4
)

# Build a specific model
model = Pretraining.build_model(:mamba, embed_config, arch_config)
```

### Evolution (`lib/exphil/league/evolution.ex`)

Self-play training loop that improves architectures through competition.

```elixir
# Run single evolution iteration
{:ok, metrics} = Evolution.evolve(league,
  matches_per_pair: 10,
  ppo_epochs: 4,
  verbose: true
)

# Run multiple generations
{:ok, final} = Evolution.run(league,
  generations: 10,
  checkpoint_dir: "checkpoints/league",
  checkpoint_every: 5
)

# Run with pruning of weak architectures
{:ok, final} = Evolution.run_with_pruning(league,
  generations: 20,
  prune_every: 5,
  min_elo: 800,
  keep_min: 3
)
```

**Evolution Cycle:**

1. **Tournament Phase**: Run matches, collect experiences, update Elo
2. **Training Phase**: PPO update on collected experiences
3. **Evolution Phase**: Advance generation, optional pruning, save checkpoints

## Scripts

### run_league.exs

Main script to run the full league pipeline.

```bash
mix run scripts/run_league.exs \
  --replays ~/replays/mewtwo \
  --architectures mlp,lstm,mamba \
  --target-loss 1.0 \
  --generations 10 \
  --matches-per-pair 20 \
  --checkpoint-dir checkpoints/league \
  --report-path results.html
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--replays` | (required) | Path to replay directory |
| `--architectures` | all 6 | Comma-separated list |
| `--target-loss` | 1.0 | Pretraining target loss |
| `--generations` | 10 | Evolution generations |
| `--matches-per-pair` | 20 | Matches per architecture pair |
| `--ppo-epochs` | 4 | PPO training epochs |
| `--checkpoint-dir` | checkpoints/league | Save directory |
| `--report-path` | league_results.html | HTML report path |

### league_report.exs

Generate reports from saved checkpoints or league state.

```bash
# HTML report with charts
mix run scripts/league_report.exs \
  --checkpoint-dir checkpoints/league

# Dark theme
mix run scripts/league_report.exs \
  --checkpoint-dir checkpoints/league \
  --theme dark

# JSON for external tools
mix run scripts/league_report.exs \
  --checkpoint-dir checkpoints/league \
  --format json \
  --output results.json

# Quick terminal summary
mix run scripts/league_report.exs \
  --league-state checkpoints/league/final/league_state.json \
  --format terminal
```

**Output Formats:**

- **HTML**: Interactive report with Chart.js visualizations, Elo distribution, win rate charts, progression over generations
- **JSON**: Machine-readable data for integration with other tools
- **Terminal**: Quick text summary for command-line review

## Elo System

The league uses a standard Elo rating system:

- **Initial Rating**: 1000
- **K-Factor**: 32 (dynamic, decreases with more games)
- **Expected Score**: `E = 1 / (1 + 10^((R_opponent - R_self) / 400))`
- **Rating Update**: `R_new = R_old + K * (S - E)`

Elo changes are symmetric: winner gains what loser loses.

## Testing

```bash
# Run all league tests
mix test test/exphil/league/

# Run specific test file
mix test test/exphil/league/match_scheduler_test.exs

# Run with verbose output
mix test test/exphil/league/ --trace
```

**Test Coverage:**

| Test File | Coverage |
|-----------|----------|
| `architecture_entry_test.exs` | Struct creation, Elo updates, serialization |
| `match_scheduler_test.exs` | All scheduling algorithms |
| `league_test.exs` | GenServer lifecycle, registration, matches |
| `evolution_test.exs` | Module structure, evolution loop |
| `pretraining_test.exs` | Model building for all architectures |

## File Structure

```
lib/exphil/league/
├── league.ex              # Main GenServer
├── architecture_entry.ex  # Architecture struct
├── match_scheduler.ex     # Tournament scheduling
├── pretraining.ex         # Imitation learning
└── evolution.ex           # Self-play training

scripts/
├── run_league.exs         # Main orchestration
└── league_report.exs      # Report generation

test/exphil/league/
├── architecture_entry_test.exs
├── match_scheduler_test.exs
├── league_test.exs
├── evolution_test.exs
└── pretraining_test.exs
```

## Example Output

### Leaderboard

```
FINAL RESULTS
============================================================

🥇 1. mamba_mewtwo
      Elo: 1287.3 | Win Rate: 68.5% | Games: 180 (123W/52L/5D)
🥈 2. attention_mewtwo
      Elo: 1156.8 | Win Rate: 55.2% | Games: 180 (99W/75L/6D)
🥉 3. jamba_mewtwo
      Elo: 1089.4 | Win Rate: 48.9% | Games: 180 (88W/89L/3D)
   4. lstm_mewtwo
      Elo: 1002.1 | Win Rate: 44.1% | Games: 180 (79W/96L/5D)
   5. gru_mewtwo
      Elo: 956.7 | Win Rate: 41.8% | Games: 180 (75W/101L/4D)
   6. mlp_mewtwo
      Elo: 907.7 | Win Rate: 36.5% | Games: 180 (66W/107L/7D)

Total matches played: 540
Generations completed: 10
```

## Future Extensions

- **Cross-character matchups**: IC LSTM vs Ganon Jamba
- **Population-based training**: Parallel architecture populations
- **Curriculum learning**: Progressive opponent difficulty
- **Meta-learning**: Learn optimal training hyperparameters
