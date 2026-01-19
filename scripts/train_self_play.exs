#!/usr/bin/env elixir
# Self-play reinforcement learning training script
#
# Usage:
#   mix run scripts/train_self_play.exs [options]
#
# Prerequisites:
#   - Pretrained policy from imitation learning (or start from scratch)
#   - Dolphin/Slippi setup (for :dolphin mode) or use :mock for testing
#
# Options:
#   --pretrained PATH   - Load pretrained policy (optional, recommended)
#   --mode MODE         - Training mode: simple_mix or league (default: simple_mix)
#   --game-type TYPE    - Game type: mock or dolphin (default: mock)
#   --num-envs N        - Number of parallel games (default: 2)
#   --timesteps N       - Total training timesteps (default: 100000)
#   --rollout-length N  - Steps per rollout collection (default: 128)
#   --checkpoint PATH   - Checkpoint save path (default: checkpoints/self_play.axon)
#   --save-interval N   - Save checkpoint every N iterations (default: 10)
#
# Dolphin Options (when --game-type dolphin):
#   --dolphin PATH      - Path to Dolphin/Slippi
#   --iso PATH          - Path to Melee ISO
#   --character NAME    - Character to play (default: mewtwo)
#   --stage NAME        - Stage (default: final_destination)
#
# PPO Options:
#   --ppo-epochs N      - PPO epochs per update (default: 4)
#   --ppo-clip F        - PPO clip epsilon (default: 0.2)
#   --learning-rate F   - Learning rate (default: 3e-4)
#   --gamma F           - Discount factor (default: 0.99)
#   --gae-lambda F      - GAE lambda (default: 0.95)

# Force line-buffered output for progress visibility
:io.setopts(:standard_io, [:binary, {:encoding, :unicode}])

defmodule Progress do
  def puts(line), do: IO.puts(line) |> tap(fn _ -> :erlang.yield() end)
  def stderr(line), do: IO.puts(:stderr, line)
end

require Logger

alias ExPhil.Training.{PPO}
alias ExPhil.Training.SelfPlay.{OpponentPool, ParallelCollector, LeagueTrainer}

# ============================================================================
# Parse Arguments
# ============================================================================

args = System.argv()

get_arg = fn name, default ->
  case Enum.find_index(args, &(&1 == name)) do
    nil -> default
    idx -> Enum.at(args, idx + 1, default)
  end
end

has_flag = fn name -> name in args end

opts = %{
  pretrained: get_arg.("--pretrained", nil),
  mode: String.to_atom(get_arg.("--mode", "simple_mix")),
  game_type: String.to_atom(get_arg.("--game-type", "mock")),
  num_envs: String.to_integer(get_arg.("--num-envs", "2")),
  timesteps: String.to_integer(get_arg.("--timesteps", "100000")),
  rollout_length: String.to_integer(get_arg.("--rollout-length", "128")),
  checkpoint: get_arg.("--checkpoint", "checkpoints/self_play.axon"),
  save_interval: String.to_integer(get_arg.("--save-interval", "10")),
  # Dolphin options
  dolphin_path: get_arg.("--dolphin", nil),
  iso_path: get_arg.("--iso", nil),
  character: get_arg.("--character", "mewtwo"),
  stage: get_arg.("--stage", "final_destination"),
  # PPO options
  ppo_epochs: String.to_integer(get_arg.("--ppo-epochs", "4")),
  ppo_clip: String.to_float(get_arg.("--ppo-clip", "0.2")),
  learning_rate: String.to_float(get_arg.("--learning-rate", "3.0e-4")),
  gamma: String.to_float(get_arg.("--gamma", "0.99")),
  gae_lambda: String.to_float(get_arg.("--gae-lambda", "0.95")),
  # Flags
  help: has_flag.("--help") or has_flag.("-h")
}

if opts.help do
  IO.puts("""

  Self-Play RL Training Script
  ============================

  Usage: mix run scripts/train_self_play.exs [options]

  Options:
    --pretrained PATH   Load pretrained policy (recommended)
    --mode MODE         Training mode: simple_mix or league (default: simple_mix)
    --game-type TYPE    Game type: mock or dolphin (default: mock)
    --num-envs N        Number of parallel games (default: 2)
    --timesteps N       Total training timesteps (default: 100000)
    --rollout-length N  Steps per rollout collection (default: 128)
    --checkpoint PATH   Checkpoint save path
    --save-interval N   Save every N iterations (default: 10)

  Dolphin Options (when --game-type dolphin):
    --dolphin PATH      Path to Dolphin/Slippi
    --iso PATH          Path to Melee ISO
    --character NAME    Character (default: mewtwo)
    --stage NAME        Stage (default: final_destination)

  PPO Options:
    --ppo-epochs N      PPO epochs per update (default: 4)
    --ppo-clip F        PPO clip epsilon (default: 0.2)
    --learning-rate F   Learning rate (default: 3e-4)
    --gamma F           Discount factor (default: 0.99)
    --gae-lambda F      GAE lambda (default: 0.95)

  Examples:
    # Quick test with mock environment
    mix run scripts/train_self_play.exs --game-type mock --timesteps 1000

    # Train with pretrained policy
    mix run scripts/train_self_play.exs \\
      --pretrained checkpoints/imitation_latest_policy.bin \\
      --mode simple_mix \\
      --timesteps 100000

    # League training with Dolphin
    mix run scripts/train_self_play.exs \\
      --pretrained checkpoints/imitation_latest_policy.bin \\
      --mode league \\
      --game-type dolphin \\
      --dolphin ~/.config/Slippi\\ Launcher/netplay \\
      --iso ~/melee.iso \\
      --timesteps 500000
  """)
  System.halt(0)
end

# ============================================================================
# Display Configuration
# ============================================================================

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║             ExPhil Self-Play RL Training                       ║
╚════════════════════════════════════════════════════════════════╝

Configuration:
  Mode:           #{opts.mode}
  Game Type:      #{opts.game_type}
  Num Envs:       #{opts.num_envs}
  Timesteps:      #{opts.timesteps}
  Rollout Length: #{opts.rollout_length}
  Checkpoint:     #{opts.checkpoint}
  Save Interval:  #{opts.save_interval} iterations
  Pretrained:     #{opts.pretrained || "none (random init)"}

PPO Config:
  Epochs:         #{opts.ppo_epochs}
  Clip Epsilon:   #{opts.ppo_clip}
  Learning Rate:  #{opts.learning_rate}
  Gamma:          #{opts.gamma}
  GAE Lambda:     #{opts.gae_lambda}
""")

if opts.game_type == :dolphin do
  IO.puts("""
Dolphin Config:
  Path:           #{opts.dolphin_path}
  ISO:            #{opts.iso_path}
  Character:      #{opts.character}
  Stage:          #{opts.stage}
""")
end

# ============================================================================
# Step 1: Load or Initialize Policy
# ============================================================================

Progress.stderr("\nStep 1: Initializing policy...")

{policy_model, policy_params, embed_size} = if opts.pretrained do
  Progress.stderr("  Loading pretrained policy from #{opts.pretrained}...")

  case ExPhil.Training.load_policy(opts.pretrained) do
    {:ok, policy} ->
      embed_size = policy.config[:embed_size] || 1991
      Progress.stderr("  ✓ Loaded policy (embed_size=#{embed_size})")
      {policy.model, policy.params, embed_size}

    {:error, reason} ->
      IO.puts(:stderr, "  ✗ Failed to load policy: #{inspect(reason)}")
      System.halt(1)
  end
else
  Progress.stderr("  Creating random policy...")
  embed_size = 1991
  hidden_sizes = [256, 256]

  model = ExPhil.Networks.Policy.build(
    embed_size: embed_size,
    hidden_sizes: hidden_sizes,
    dropout: 0.0
  )

  {init_fn, _} = Axon.build(model)
  template = Nx.template({1, embed_size}, :f32)
  params = init_fn.(template, %{})

  Progress.stderr("  ✓ Created random policy (embed_size=#{embed_size})")
  {model, params, embed_size}
end

policy = {policy_model, policy_params}

# ============================================================================
# Step 2: Initialize Opponent Pool
# ============================================================================

Progress.stderr("\nStep 2: Setting up opponent pool...")

{:ok, opponent_pool} = OpponentPool.new(
  config: %{
    current: 0.4,      # 40% current self-play
    historical: 0.3,   # 30% recent historical
    cpu: 0.2,          # 20% CPU opponents
    random: 0.1        # 10% random from history
  },
  cpu_levels: [5, 6, 7, 8, 9],
  max_historical: 10
)

# Set current policy as available opponent
opponent_pool = OpponentPool.set_current(opponent_pool, policy_params)
Progress.stderr("  ✓ Opponent pool initialized")

# ============================================================================
# Step 3: Initialize PPO Trainer
# ============================================================================

Progress.stderr("\nStep 3: Initializing PPO trainer...")

ppo_config = [
  gamma: opts.gamma,
  gae_lambda: opts.gae_lambda,
  clip_epsilon: opts.ppo_clip,
  ppo_epochs: opts.ppo_epochs,
  learning_rate: opts.learning_rate,
  embed_size: embed_size
]

{:ok, ppo_trainer} = PPO.new(policy_model, policy_params, ppo_config)
Progress.stderr("  ✓ PPO trainer initialized")

# ============================================================================
# Step 4: Initialize Parallel Collector
# ============================================================================

Progress.stderr("\nStep 4: Setting up #{opts.num_envs} parallel game environments...")

dolphin_config = if opts.game_type == :dolphin do
  %{
    path: opts.dolphin_path,
    iso: opts.iso_path,
    character: opts.character,
    stage: opts.stage
  }
else
  %{}
end

{:ok, collector} = ParallelCollector.new(
  num_envs: opts.num_envs,
  policy: policy,
  opponent_pool: opponent_pool,
  rollout_length: opts.rollout_length,
  game_type: opts.game_type,
  dolphin_config: dolphin_config
)

Progress.stderr("  ✓ #{opts.num_envs} environments ready")

# ============================================================================
# Step 5: Training Loop
# ============================================================================

Progress.stderr("\nStep 5: Starting self-play training...")
Progress.stderr("─" |> String.duplicate(60))

total_iterations = div(opts.timesteps, opts.rollout_length * opts.num_envs)
Progress.stderr("  Total iterations: #{total_iterations}")
Progress.stderr("  Steps per iteration: #{opts.rollout_length * opts.num_envs}")

start_time = System.monotonic_time(:second)
iteration = 0
total_steps = 0

# Training state
state = %{
  ppo_trainer: ppo_trainer,
  collector: collector,
  opponent_pool: opponent_pool,
  iteration: 0,
  total_steps: 0,
  total_episodes: 0,
  metrics_history: []
}

final_state = Enum.reduce_while(1..total_iterations, state, fn iter, state ->
  iter_start = System.monotonic_time(:millisecond)

  # Collect rollouts from all environments
  {:ok, updated_collector, rollouts} = ParallelCollector.collect_rollouts(state.collector)

  collect_time = System.monotonic_time(:millisecond) - iter_start

  # Check if we have data
  rollout_size = Nx.axis_size(rollouts.states, 0)

  if rollout_size == 0 do
    Progress.stderr("  ⚠ No data collected in iteration #{iter}, skipping...")
    {:cont, %{state | collector: updated_collector, iteration: iter}}
  else
    # Update PPO with collected experience
    update_start = System.monotonic_time(:millisecond)
    {updated_ppo, ppo_metrics} = PPO.update(state.ppo_trainer, rollouts)
    update_time = System.monotonic_time(:millisecond) - update_start

    # Sync new policy to collector
    {:ok, synced_collector} = ParallelCollector.update_policy(
      updated_collector,
      updated_ppo.policy_params
    )

    # Update opponent pool with new policy
    updated_pool = OpponentPool.set_current(state.opponent_pool, updated_ppo.policy_params)

    # Snapshot to historical every 10 iterations
    updated_pool = if rem(iter, 10) == 0 do
      OpponentPool.snapshot(updated_pool, "iter_#{iter}")
    else
      updated_pool
    end

    # Resample opponents periodically
    {:ok, resampled_collector} = if rem(iter, 5) == 0 do
      ParallelCollector.resample_opponents(synced_collector)
    else
      {:ok, synced_collector}
    end

    # Calculate stats
    steps_this_iter = rollout_size
    new_total_steps = state.total_steps + steps_this_iter
    collector_stats = ParallelCollector.get_stats(resampled_collector)

    # Progress output
    elapsed = System.monotonic_time(:second) - start_time
    steps_per_sec = if elapsed > 0, do: Float.round(new_total_steps / elapsed, 1), else: 0.0
    pct = round(iter / total_iterations * 100)

    # Progress bar
    bar_width = 20
    filled = round(pct / 100 * bar_width)
    bar = String.duplicate("█", filled) <> String.duplicate("░", bar_width - filled)

    progress_line = "  #{bar} #{pct}% | iter #{iter}/#{total_iterations} | " <>
                   "steps: #{new_total_steps} | " <>
                   "policy_loss: #{Float.round(ppo_metrics.policy_loss, 4)} | " <>
                   "value_loss: #{Float.round(ppo_metrics.value_loss, 4)} | " <>
                   "#{steps_per_sec} steps/s"

    Progress.stderr(progress_line)

    # Save checkpoint periodically
    if rem(iter, opts.save_interval) == 0 do
      checkpoint = %{
        ppo_trainer: updated_ppo,
        opponent_pool: updated_pool,
        iteration: iter,
        total_steps: new_total_steps,
        config: opts
      }

      checkpoint_path = String.replace(opts.checkpoint, ".axon", "_iter#{iter}.axon")
      case File.write(checkpoint_path, :erlang.term_to_binary(checkpoint)) do
        :ok -> Progress.stderr("  ✓ Checkpoint saved: #{checkpoint_path}")
        {:error, reason} -> Progress.stderr("  ⚠ Checkpoint failed: #{inspect(reason)}")
      end
    end

    new_state = %{state |
      ppo_trainer: updated_ppo,
      collector: resampled_collector,
      opponent_pool: updated_pool,
      iteration: iter,
      total_steps: new_total_steps,
      total_episodes: collector_stats.total_episodes,
      metrics_history: [ppo_metrics | state.metrics_history]
    }

    # Check if we've reached target timesteps
    if new_total_steps >= opts.timesteps do
      {:halt, new_state}
    else
      {:cont, new_state}
    end
  end
end)

# ============================================================================
# Step 6: Save Final Checkpoint
# ============================================================================

total_time = System.monotonic_time(:second) - start_time
total_min = div(total_time, 60)
total_sec = rem(total_time, 60)

Progress.stderr("")
Progress.stderr("─" |> String.duplicate(60))
Progress.stderr("✓ Training complete in #{total_min}m #{total_sec}s")
Progress.stderr("─" |> String.duplicate(60))

Progress.stderr("\nStep 6: Saving final checkpoint...")

final_checkpoint = %{
  ppo_trainer: final_state.ppo_trainer,
  opponent_pool: final_state.opponent_pool,
  iteration: final_state.iteration,
  total_steps: final_state.total_steps,
  config: opts
}

case File.write(opts.checkpoint, :erlang.term_to_binary(final_checkpoint)) do
  :ok -> Progress.stderr("  ✓ Final checkpoint saved: #{opts.checkpoint}")
  {:error, reason} -> Progress.stderr("  ✗ Failed: #{inspect(reason)}")
end

# Also export policy for inference
policy_path = String.replace(opts.checkpoint, ".axon", "_policy.bin")
policy_export = %{
  model: policy_model,
  params: final_state.ppo_trainer.policy_params,
  config: %{embed_size: embed_size}
}

case File.write(policy_path, :erlang.term_to_binary(policy_export)) do
  :ok -> Progress.stderr("  ✓ Policy exported: #{policy_path}")
  {:error, reason} -> Progress.stderr("  ✗ Policy export failed: #{inspect(reason)}")
end

# Shutdown environments
ParallelCollector.shutdown(final_state.collector)

Progress.stderr("""

Training Summary:
  Total Steps:    #{final_state.total_steps}
  Total Episodes: #{final_state.total_episodes}
  Iterations:     #{final_state.iteration}
  Time:           #{total_min}m #{total_sec}s

Next steps:
  1. Evaluate: mix run scripts/eval_model.exs --policy #{policy_path}
  2. Play: mix run scripts/play_dolphin.exs --policy #{policy_path} ...
  3. Continue training: mix run scripts/train_self_play.exs --pretrained #{policy_path} ...
""")
