# BEAM Concurrency Self-Play Architecture

## Overview

This document outlines the architecture for self-play reinforcement learning using
Elixir/BEAM concurrency. The key insight is that BEAM's actor model is ideal for
running multiple parallel games, each with its own Dolphin instance.

## Goals

1. **Parallel Game Execution** - Run N games simultaneously on multiple cores
2. **Population-Based Training** - Avoid policy collapse through diversity
3. **Historical Sampling** - Play against past policy versions
4. **Efficient Experience Collection** - Batch experiences from all games
5. **Hot Policy Swaps** - Update policies without restarting games

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              Application                                    │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │                        SelfPlaySupervisor                               │ │
│ │                                                                         │ │
│ │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐│ │
│ │  │PopulationManager│  │   Matchmaker    │  │   ExperienceCollector   ││ │
│ │  │                 │  │                 │  │                          ││ │
│ │  │ • policies[]    │←→│ • Elo ratings   │→→│ • experience_buffer     ││ │
│ │  │ • history[]     │  │ • matchmaking   │  │ • batch_size: 2048      ││ │
│ │  │ • add/remove    │  │ • league tiers  │  │ • to_trainer()          ││ │
│ │  └─────────────────┘  └─────────────────┘  └──────────────────────────┘│ │
│ │                              │                            ▲             │ │
│ │                              ▼                            │             │ │
│ │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│ │  │                    GamePoolSupervisor                             │  │ │
│ │  │  (DynamicSupervisor - one_for_one)                               │  │ │
│ │  │                                                                   │  │ │
│ │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │  │ │
│ │  │  │GameRunner │  │GameRunner │  │GameRunner │  │GameRunner │ ... │  │ │
│ │  │  │   #1      │  │   #2      │  │   #3      │  │   #N      │     │  │ │
│ │  │  │           │  │           │  │           │  │           │     │  │ │
│ │  │  │ p1_agent──│──│ p1_agent──│──│ p1_agent──│──│ p1_agent  │     │  │ │
│ │  │  │ p2_agent──│──│ p2_agent──│──│ p2_agent──│──│ p2_agent  │     │  │ │
│ │  │  │ dolphin   │  │ dolphin   │  │ dolphin   │  │ dolphin   │     │  │ │
│ │  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘     │  │ │
│ │  └──────────────────────────────────────────────────────────────────┘  │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          Trainer                                      │  │
│  │  • PPO optimizer                                                      │  │
│  │  • Gradient updates                                                   │  │
│  │  • Pushes new policies to PopulationManager                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. SelfPlaySupervisor

Top-level supervisor that manages all self-play components.

```elixir
defmodule ExPhil.SelfPlay.Supervisor do
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(opts) do
    children = [
      {ExPhil.SelfPlay.PopulationManager, opts},
      {ExPhil.SelfPlay.Matchmaker, opts},
      {ExPhil.SelfPlay.ExperienceCollector, opts},
      {ExPhil.SelfPlay.GamePoolSupervisor, opts}
    ]
    Supervisor.init(children, strategy: :rest_for_one)
  end
end
```

### 2. PopulationManager

Manages the population of policies, including current and historical versions.

```elixir
defmodule ExPhil.SelfPlay.PopulationManager do
  use GenServer

  defstruct [
    :current_policy,      # Latest trained policy
    :historical_policies, # Past versions for sampling
    :population,          # Multiple diverse policies
    :max_history_size,    # Max historical policies to keep
    :history_sample_prob  # Probability of sampling historical opponent
  ]

  # API
  def add_policy(manager, policy, generation)
  def sample_opponent(manager, opts \\ [])
  def get_current(manager)
  def get_population(manager)
end
```

**Key Design Decisions:**
- Keep last N policy checkpoints for historical sampling
- Sample historical policies with probability p (e.g., 0.25)
- Support multiple concurrent policies for population diversity

### 3. Matchmaker

Pairs policies for games and tracks Elo ratings.

```elixir
defmodule ExPhil.SelfPlay.Matchmaker do
  use GenServer

  defstruct [
    :elo_ratings,      # Map of policy_id -> Elo rating
    :pending_matches,  # Queue of matches to be played
    :active_matches    # Currently running matches
  ]

  # API
  def request_match(matchmaker, game_runner)
  def report_result(matchmaker, match_id, winner, game_stats)
  def get_ratings(matchmaker)
end
```

**Matchmaking Strategies:**
1. **Self-play** - Latest policy vs itself
2. **Historical** - Latest vs random historical policy
3. **Population** - Random pair from population pool
4. **League** - Skill-based matching using Elo

### 4. ExperienceCollector

Collects experiences from all running games and batches for training.

```elixir
defmodule ExPhil.SelfPlay.ExperienceCollector do
  use GenServer

  defstruct [
    :buffer,          # List of experiences
    :batch_size,      # Target batch size for training
    :max_buffer_size, # Max experiences to keep
    :ready_callback   # Function to call when batch is ready
  ]

  # API
  def submit(collector, experience)
  def get_batch(collector, size)
  def size(collector)
  def flush(collector)

  # Experience format
  @type experience :: %{
    state: Nx.Tensor.t(),
    action: map(),
    reward: float(),
    next_state: Nx.Tensor.t(),
    done: boolean(),
    policy_id: String.t()
  }
end
```

### 5. GamePoolSupervisor

DynamicSupervisor that manages multiple GameRunner processes.

```elixir
defmodule ExPhil.SelfPlay.GamePoolSupervisor do
  use DynamicSupervisor

  def start_link(opts)
  def start_game(pool, game_opts)
  def stop_game(pool, game_id)
  def list_games(pool)
  def count_games(pool)
end
```

### 6. GameRunner

GenServer that manages a single game instance (Dolphin + two agents).

```elixir
defmodule ExPhil.SelfPlay.GameRunner do
  use GenServer

  defstruct [
    :game_id,
    :dolphin_pid,     # Dolphin process
    :p1_policy_id,
    :p2_policy_id,
    :frame_count,
    :episode_reward,
    :match_id,
    :status           # :waiting, :playing, :finished
  ]

  # API
  def start_link(opts)
  def start_game(runner)
  def get_status(runner)
  def swap_policy(runner, port, policy_id)
end
```

## Data Flow

### Experience Collection Flow

```
GameRunner#1 ──────────┐
                       │
GameRunner#2 ─────────├──→ ExperienceCollector ──→ Trainer
                       │           │
GameRunner#N ─────────┘           │
                                   ▼
                            PopulationManager
                                   │
                                   ▼
                            All GameRunners
                            (policy update)
```

### Policy Update Flow

```
Trainer
    │
    ├─── Compute gradients from experience batch
    │
    ├─── Update policy parameters
    │
    └─── Push new policy to PopulationManager
              │
              ├─── Add to historical policies
              │
              └─── Broadcast to GameRunners
                        │
                        └─── Hot-swap Agent policies
```

## Configuration

```elixir
config :exphil, :self_play,
  # Game pool
  num_games: System.schedulers_online(),
  max_games: 16,

  # Population
  population_size: 4,
  history_size: 20,
  history_sample_prob: 0.25,

  # Experience collection
  batch_size: 2048,
  max_buffer_size: 10_000,

  # Training
  update_interval: :batch_ready,  # or {:frames, 10_000}
  checkpoint_interval: {:episodes, 100}
```

## Implementation Phases

### Phase 1: Core Infrastructure (MVP)
- [ ] GameRunner GenServer with Dolphin integration
- [ ] GamePoolSupervisor for managing game instances
- [ ] ExperienceCollector for batching experiences
- [ ] Basic self-play (latest vs latest)

### Phase 2: Population & History
- [ ] PopulationManager with historical sampling
- [ ] Matchmaker with Elo ratings
- [ ] League-based matchmaking

### Phase 3: Training Integration
- [ ] PPO integration with experience batches
- [ ] Policy hot-swapping during games
- [ ] Gradient accumulation across games

### Phase 4: Scaling & Optimization
- [ ] Distributed games across nodes
- [ ] GPU batch inference
- [ ] Async policy updates

## Key BEAM Advantages

1. **Process Isolation** - Each game runs in isolated process, crashes don't affect others
2. **Lightweight Processes** - Can run hundreds of games with minimal overhead
3. **Message Passing** - Clean communication between components
4. **Supervision Trees** - Automatic restart on failures
5. **Hot Code Loading** - Update policies without stopping games
6. **Distribution** - Scale to multiple machines with minimal code changes

## Anti-Patterns to Avoid

1. **Single-opponent collapse** - Always use historical/population sampling
2. **Shared mutable state** - Use message passing, not shared memory
3. **Synchronous bottlenecks** - Keep experience collection async
4. **Monolithic processes** - Keep components small and focused

## References

- slippi-ai: https://github.com/vladfi1/slippi-ai
- OpenAI Five: https://arxiv.org/abs/1912.06680
- AlphaStar: https://www.nature.com/articles/s41586-019-1724-z
- Population Based Training: https://arxiv.org/abs/1711.09846
