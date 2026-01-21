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
#   --num-games N       - Number of parallel games (default: 4)
#   --timesteps N       - Total training timesteps (default: 100000)
#   --rollout-length N  - Steps per rollout collection (default: 128)
#   --batch-size N      - Batch size for PPO updates (default: 2048)
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

defmodule SelfPlayHelpers do
  alias ExPhil.Training.PPO

  def experiences_to_rollout(experiences) when length(experiences) == 0 do
    %{
      states: Nx.tensor([], type: :f32),
      actions: %{},
      rewards: Nx.tensor([], type: :f32),
      values: Nx.tensor([], type: :f32),
      log_probs: Nx.tensor([], type: :f32),
      dones: Nx.tensor([], type: :f32)
    }
  end

  def experiences_to_rollout(experiences) do
    # Stack states
    states = experiences
    |> Enum.map(& &1.state)
    |> Enum.map(fn t ->
      case Nx.shape(t) do
        {1, _n} -> Nx.squeeze(t, axes: [0])
        _ -> t
      end
    end)
    |> Nx.stack()

    # Stack scalar values
    rewards = experiences
    |> Enum.map(& &1.reward)
    |> Nx.tensor(type: :f32)

    dones = experiences
    |> Enum.map(& if(&1.done, do: 1.0, else: 0.0))
    |> Nx.tensor(type: :f32)

    values = experiences
    |> Enum.map(fn exp ->
      case exp.value do
        %Nx.Tensor{} = t -> Nx.to_number(t)
        n when is_number(n) -> n
      end
    end)
    |> Kernel.++([0.0])  # Bootstrap value
    |> Nx.tensor(type: :f32)

    log_probs = experiences
    |> Enum.map(fn exp ->
      case exp.log_prob do
        %Nx.Tensor{} = t -> Nx.to_number(t)
        n when is_number(n) -> n
      end
    end)
    |> Nx.tensor(type: :f32)

    # Stack actions (map of tensors)
    actions = stack_actions(Enum.map(experiences, & &1.action))

    %{
      states: states,
      actions: actions,
      rewards: rewards,
      values: values,
      log_probs: log_probs,
      dones: dones
    }
  end

  defp stack_actions(actions_list) when length(actions_list) == 0, do: %{}

  defp stack_actions(actions_list) do
    keys = Map.keys(hd(actions_list))

    Map.new(keys, fn key ->
      values = Enum.map(actions_list, fn action ->
        case Map.get(action, key) do
          %Nx.Tensor{} = t -> t
          n when is_number(n) -> Nx.tensor(n)
          other -> Nx.tensor(other)
        end
      end)

      {key, Nx.stack(values)}
    end)
  end

  def save_checkpoint(ppo_trainer, policy_model, path, iteration, total_steps, opts) do
    checkpoint = %{
      params: PPO.to_binary_backend(ppo_trainer.params),
      model: policy_model,
      iteration: iteration,
      total_steps: total_steps,
      config: opts
    }

    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    case File.write(path, :erlang.term_to_binary(checkpoint)) do
      :ok -> Progress.stderr("  ✓ Checkpoint saved: #{path}")
      {:error, reason} -> Progress.stderr("  ⚠ Checkpoint failed: #{inspect(reason)}")
    end
  end
end

require Logger

alias ExPhil.SelfPlay.{Supervisor, ExperienceCollector}
alias ExPhil.Training.PPO

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
  num_games: String.to_integer(get_arg.("--num-games", "4")),
  timesteps: String.to_integer(get_arg.("--timesteps", "100000")),
  rollout_length: String.to_integer(get_arg.("--rollout-length", "128")),
  batch_size: String.to_integer(get_arg.("--batch-size", "2048")),
  checkpoint: get_arg.("--checkpoint", "checkpoints/self_play.axon"),
  save_interval: String.to_integer(get_arg.("--save-interval", "10")),
  snapshot_interval: String.to_integer(get_arg.("--snapshot-interval", "5")),
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
  help: has_flag.("--help") or has_flag.("-h"),
  track_elo: has_flag.("--track-elo")
}

if opts.help do
  IO.puts("""

  Self-Play RL Training Script (GenServer Architecture)
  =====================================================

  Usage: mix run scripts/train_self_play.exs [options]

  Options:
    --pretrained PATH   Load pretrained policy (recommended)
    --mode MODE         Training mode: simple_mix or league (default: simple_mix)
    --game-type TYPE    Game type: mock or dolphin (default: mock)
    --num-games N       Number of parallel games (default: 4)
    --timesteps N       Total training timesteps (default: 100000)
    --rollout-length N  Steps per rollout (default: 128)
    --batch-size N      PPO batch size (default: 2048)
    --checkpoint PATH   Checkpoint save path
    --save-interval N   Save every N iterations (default: 10)
    --snapshot-interval N Snapshot policy every N iterations (default: 5)
    --track-elo         Enable Elo rating tracking

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

    # Full training with Elo tracking
    mix run scripts/train_self_play.exs \\
      --pretrained checkpoints/imitation_latest_policy.bin \\
      --num-games 8 \\
      --batch-size 4096 \\
      --timesteps 500000 \\
      --track-elo
  """)
  System.halt(0)
end

# ============================================================================
# Display Configuration
# ============================================================================

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║      ExPhil Self-Play RL Training (GenServer Architecture)     ║
╚════════════════════════════════════════════════════════════════╝

Configuration:
  Mode:             #{opts.mode}
  Game Type:        #{opts.game_type}
  Num Games:        #{opts.num_games}
  Timesteps:        #{opts.timesteps}
  Rollout Length:   #{opts.rollout_length}
  Batch Size:       #{opts.batch_size}
  Checkpoint:       #{opts.checkpoint}
  Save Interval:    #{opts.save_interval} iterations
  Snapshot Interval:#{opts.snapshot_interval} iterations
  Pretrained:       #{opts.pretrained || "none (random init)"}
  Elo Tracking:     #{opts.track_elo}

PPO Config:
  Epochs:           #{opts.ppo_epochs}
  Clip Epsilon:     #{opts.ppo_clip}
  Learning Rate:    #{opts.learning_rate}
  Gamma:            #{opts.gamma}
  GAE Lambda:       #{opts.gae_lambda}
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
  Progress.stderr("  Creating random actor-critic policy...")
  embed_size = 1991
  hidden_sizes = [256, 256]

  # Use ActorCritic to get both policy and value heads
  model = ExPhil.Networks.ActorCritic.build_combined(
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

# ============================================================================
# Step 2: Start Self-Play Supervisor
# ============================================================================

Progress.stderr("\nStep 2: Starting self-play infrastructure...")

{:ok, _supervisor} = Supervisor.start_link(
  batch_size: opts.batch_size,
  max_history_size: 20,
  start_matchmaker: opts.track_elo
)

Progress.stderr("  ✓ Self-play supervisor started")

# Set the current policy
:ok = Supervisor.set_policy(policy_model, policy_params)
Progress.stderr("  ✓ Policy registered with population manager")

# ============================================================================
# Step 3: Initialize PPO Trainer
# ============================================================================

Progress.stderr("\nStep 3: Initializing PPO trainer...")

ppo_trainer = PPO.new(
  embed_size: embed_size,
  pretrained_path: opts.pretrained,
  gamma: opts.gamma,
  gae_lambda: opts.gae_lambda,
  clip_range: opts.ppo_clip,
  num_epochs: opts.ppo_epochs,
  learning_rate: opts.learning_rate,
  batch_size: opts.batch_size
)

Progress.stderr("  ✓ PPO trainer initialized")

# ============================================================================
# Step 4: Start Parallel Games
# ============================================================================

Progress.stderr("\nStep 4: Starting #{opts.num_games} parallel games...")

game_results = Supervisor.start_games(opts.num_games, game_type: opts.game_type)

started_count = Enum.count(game_results, fn {:ok, _} -> true; _ -> false end)
Progress.stderr("  ✓ #{started_count}/#{opts.num_games} games started")

if started_count < opts.num_games do
  failed = Enum.filter(game_results, fn {:ok, _} -> false; _ -> true end)
  Progress.stderr("  ⚠ Some games failed to start: #{inspect(Enum.take(failed, 3))}")
end

# ============================================================================
# Step 5: Training Loop
# ============================================================================

Progress.stderr("\nStep 5: Starting self-play training...")
Progress.stderr("─" |> String.duplicate(60))

steps_per_iteration = opts.rollout_length * opts.num_games
total_iterations = div(opts.timesteps, steps_per_iteration)
Progress.stderr("  Total iterations: #{total_iterations}")
Progress.stderr("  Steps per iteration: #{steps_per_iteration}")

start_time = System.monotonic_time(:second)

# Training state
state = %{
  ppo_trainer: ppo_trainer,
  iteration: 0,
  total_steps: 0,
  total_episodes: 0,
  metrics_history: [],
  policy_model: policy_model
}

final_state = Enum.reduce_while(1..total_iterations, state, fn iter, state ->
  iter_start = System.monotonic_time(:millisecond)

  # Collect experience from all games
  experiences = Supervisor.collect_steps(opts.rollout_length)
  _collect_time = System.monotonic_time(:millisecond) - iter_start

  steps_collected = length(experiences)

  if steps_collected == 0 do
    Progress.stderr("  ⚠ No experience collected in iteration #{iter}, skipping...")
    {:cont, %{state | iteration: iter}}
  else
    # Convert experiences to rollout format for PPO
    rollouts = SelfPlayHelpers.experiences_to_rollout(experiences)

    # Update PPO with collected experience
    update_start = System.monotonic_time(:millisecond)
    {updated_ppo, ppo_metrics} = PPO.update(state.ppo_trainer, rollouts)
    _update_time = System.monotonic_time(:millisecond) - update_start

    # Update policy in population manager and sync to all games
    Supervisor.update_policy_params(updated_ppo.params)

    # Snapshot to historical periodically
    if rem(iter, opts.snapshot_interval) == 0 do
      Supervisor.snapshot_policy()
    end

    # Resample opponents periodically
    if rem(iter, 5) == 0 do
      Supervisor.resample_all_opponents()
    end

    # Calculate stats
    new_total_steps = state.total_steps + steps_collected
    collector_stats = ExperienceCollector.get_stats(Supervisor.experience_collector())

    # Progress output
    elapsed = System.monotonic_time(:second) - start_time
    steps_per_sec = if elapsed > 0, do: Float.round(new_total_steps / elapsed, 1), else: 0.0
    pct = round(iter / total_iterations * 100)

    # Progress bar
    bar_width = 20
    filled = round(pct / 100 * bar_width)
    bar = String.duplicate("█", filled) <> String.duplicate("░", bar_width - filled)

    policy_loss = Map.get(ppo_metrics, :policy_loss, 0.0)
    value_loss = Map.get(ppo_metrics, :value_loss, 0.0)

    progress_line = "  #{bar} #{pct}% | iter #{iter}/#{total_iterations} | " <>
                   "steps: #{new_total_steps} | " <>
                   "policy_loss: #{Float.round(policy_loss, 4)} | " <>
                   "value_loss: #{Float.round(value_loss, 4)} | " <>
                   "#{steps_per_sec} steps/s"

    Progress.stderr(progress_line)

    # Save checkpoint periodically
    if rem(iter, opts.save_interval) == 0 do
      checkpoint_path = String.replace(opts.checkpoint, ".axon", "_iter#{iter}.axon")
      SelfPlayHelpers.save_checkpoint(updated_ppo, state.policy_model, checkpoint_path, iter, new_total_steps, opts)
    end

    new_state = %{state |
      ppo_trainer: updated_ppo,
      iteration: iter,
      total_steps: new_total_steps,
      total_episodes: collector_stats.batches_produced,
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

SelfPlayHelpers.save_checkpoint(final_state.ppo_trainer, final_state.policy_model, opts.checkpoint, final_state.iteration, final_state.total_steps, opts)

# Also export policy for inference
policy_path = String.replace(opts.checkpoint, ".axon", "_policy.bin")
PPO.export_policy(final_state.ppo_trainer, policy_path)
Progress.stderr("  ✓ Policy exported: #{policy_path}")

# Print Elo leaderboard if tracking
if opts.track_elo do
  Progress.stderr("\nElo Leaderboard:")
  leaderboard = Supervisor.get_leaderboard(10)
  Enum.with_index(leaderboard, 1) |> Enum.each(fn {entry, rank} ->
    Progress.stderr("  #{rank}. #{entry.id}: #{Float.round(entry.rating, 1)} (#{entry.wins}W/#{entry.losses}L)")
  end)
end

# Shutdown
Supervisor.stop_all_games()

Progress.stderr("""

Training Summary:
  Total Steps:    #{final_state.total_steps}
  Iterations:     #{final_state.iteration}
  Time:           #{total_min}m #{total_sec}s

Next steps:
  1. Evaluate: mix run scripts/eval_model.exs --policy #{policy_path}
  2. Play: mix run scripts/play_dolphin.exs --policy #{policy_path} ...
  3. Continue training: mix run scripts/train_self_play.exs --pretrained #{policy_path} ...
""")

