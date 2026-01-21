#!/usr/bin/env elixir
# PPO Training Script: Fine-tune imitation policy with reinforcement learning
#
# Usage:
#   mix run scripts/train_ppo.exs --pretrained checkpoints/imitation_latest_policy.bin [options]
#
# Performance Tips:
#   XLA_FLAGS="--xla_cpu_multi_thread_eigen=true" mix run scripts/train_ppo.exs ...
#
# Options:
#   --pretrained PATH     - Path to pretrained imitation policy (recommended)
#   --timesteps N         - Total timesteps to train (default: 100000)
#   --rollout-length N    - Steps per rollout (default: 2048)
#   --num-epochs N        - PPO epochs per update (default: 10)
#   --batch-size N        - Minibatch size (default: 64)
#   --lr RATE             - Learning rate (default: 3e-4)
#   --checkpoint PATH     - Checkpoint save path (default: checkpoints/ppo_latest.axon)
#
# Environment Options:
#   --dolphin PATH        - Path to Slippi/Dolphin executable
#   --iso PATH            - Path to Melee 1.02 ISO
#   --character NAME      - Character to play (default: mewtwo)
#   --opponent TYPE       - Opponent: cpu1-9, self (default: cpu3)
#   --stage NAME          - Stage (default: final_destination)
#
# Mock/Test Mode:
#   --mock                - Use mock environment (no Dolphin required)
#   --mock-episodes N     - Episodes per rollout in mock mode (default: 4)

require Logger

alias ExPhil.Training.{PPO, Output}
alias ExPhil.Bridge.{MeleePort, GameState, Player, ControllerState}
alias ExPhil.{Embeddings, Rewards}
alias ExPhil.Agents.Agent

# Helper to parse hidden sizes
parse_hidden_sizes = fn
  nil -> nil
  str -> str |> String.split(",") |> Enum.map(&String.to_integer/1)
end

# Parse command line arguments
{opts, _, _} = OptionParser.parse(System.argv(),
  strict: [
    pretrained: :string,
    timesteps: :integer,
    rollout_length: :integer,
    num_epochs: :integer,
    batch_size: :integer,
    lr: :float,
    checkpoint: :string,
    dolphin: :string,
    iso: :string,
    character: :string,
    opponent: :string,
    stage: :string,
    mock: :boolean,
    mock_episodes: :integer,
    hidden_sizes: :string
  ]
)

# Configuration
config = %{
  pretrained: opts[:pretrained],
  timesteps: opts[:timesteps] || 100_000,
  rollout_length: opts[:rollout_length] || 2048,
  num_epochs: opts[:num_epochs] || 10,
  batch_size: opts[:batch_size] || 64,
  lr: opts[:lr] || 3.0e-4,
  checkpoint: opts[:checkpoint] || "checkpoints/ppo_latest.axon",
  dolphin: opts[:dolphin],
  iso: opts[:iso],
  character: opts[:character] || "mewtwo",
  opponent: opts[:opponent] || "cpu3",
  stage: opts[:stage] || "final_destination",
  mock: opts[:mock] || false,
  mock_episodes: opts[:mock_episodes] || 4,
  hidden_sizes: parse_hidden_sizes.(opts[:hidden_sizes])
}

defmodule PPOScript do
  @moduledoc false

  @doc """
  Collect a rollout from the environment.
  """
  def collect_rollout(env, agent, trainer, length) do
    embed_config = trainer.embed_config

    # Initialize buffers
    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []

    # Collect steps
    {final_env, final_states, final_actions, final_rewards, final_dones, final_values, final_log_probs} =
      Enum.reduce(1..length, {env, states, actions, rewards, dones, values, log_probs},
        fn _step, {env_acc, s_acc, a_acc, r_acc, d_acc, v_acc, lp_acc} ->
          # Get current game state from environment
          {:ok, game_state} = get_state(env_acc)

          # Embed state
          embedded = Embeddings.Game.embed(game_state, embed_config)

          # Get action and value from agent
          {:ok, action, action_log_prob, value} = get_action_with_value(agent, trainer, game_state, embedded)

          # Step environment
          {:ok, next_env, reward, done} = step_env(env_acc, action)

          # Accumulate
          {
            next_env,
            [embedded | s_acc],
            [action | a_acc],
            [reward | r_acc],
            [done | d_acc],
            [value | v_acc],
            [action_log_prob | lp_acc]
          }
        end)

    # Convert to tensors (reverse to get correct order)
    %{
      states: Nx.stack(Enum.reverse(final_states)),
      actions: stack_actions(Enum.reverse(final_actions)),
      rewards: Nx.tensor(Enum.reverse(final_rewards), type: :f32),
      dones: Nx.tensor(Enum.reverse(final_dones), type: :f32),
      values: Nx.stack(Enum.reverse(final_values)) |> Nx.squeeze(),
      log_probs: Nx.stack(Enum.reverse(final_log_probs)) |> Nx.squeeze(),
      env: final_env
    }
  end

  defp stack_actions(actions) do
    # Actions is a list of maps, need to convert to map of tensors
    keys = Map.keys(hd(actions))
    Map.new(keys, fn key ->
      values = Enum.map(actions, &Map.get(&1, key))
      {key, Nx.stack(values)}
    end)
  end

  defp get_action_with_value(agent, trainer, game_state, embedded) do
    # Use trainer's model to get action, log_prob, and value
    {_init_fn, predict_fn} = Axon.build(trainer.model)
    params = trainer.params

    # Forward pass
    %{policy: policy_logits, value: value} = predict_fn.(params, Nx.new_axis(embedded, 0))

    # Sample action
    action = sample_action(policy_logits)

    # Compute log prob of sampled action
    log_prob = ExPhil.Networks.ActorCritic.compute_log_probs(policy_logits, tensorize_action(action))

    {:ok, action, Nx.squeeze(log_prob), Nx.squeeze(value)}
  end

  defp sample_action(policy_logits) do
    # Sample from each head
    %{
      buttons: sample_buttons(policy_logits.buttons),
      main_x: sample_categorical(policy_logits.main_x),
      main_y: sample_categorical(policy_logits.main_y),
      c_x: sample_categorical(policy_logits.c_x),
      c_y: sample_categorical(policy_logits.c_y),
      shoulder: sample_categorical(policy_logits.shoulder)
    }
  end

  defp sample_buttons(logits) do
    # logits shape: [1, 8]
    probs = Nx.sigmoid(logits) |> Nx.squeeze()
    # Sample each button independently
    buttons = [:a, :b, :x, :y, :z, :l, :r, :d_up]
    Map.new(Enum.with_index(buttons), fn {btn, i} ->
      prob = probs[i] |> Nx.to_number()
      {btn, :rand.uniform() < prob}
    end)
  end

  defp sample_categorical(logits) do
    # logits shape: [1, num_classes]
    probs = Axon.Activations.softmax(logits) |> Nx.squeeze()
    # Sample from categorical distribution
    u = :rand.uniform()
    cumsum = Nx.cumulative_sum(probs)
    # Find first index where cumsum > u
    Enum.find_index(Nx.to_flat_list(cumsum), fn p -> p > u end) || 0
  end

  defp tensorize_action(action) do
    %{
      buttons: Nx.tensor([Enum.map([:a, :b, :x, :y, :z, :l, :r, :d_up], fn b -> if action.buttons[b], do: 1.0, else: 0.0 end)]),
      main_x: Nx.tensor([[action.main_x]]),
      main_y: Nx.tensor([[action.main_y]]),
      c_x: Nx.tensor([[action.c_x]]),
      c_y: Nx.tensor([[action.c_y]]),
      shoulder: Nx.tensor([[action.shoulder]])
    }
  end

  # ============================================================================
  # Environment Interface
  # ============================================================================

  def init_env(:mock, config) do
    Logger.info("Using mock environment")
    {:ok, %{type: :mock, step: 0, episode: 0, config: config}}
  end

  def init_env(:dolphin, config) do
    Logger.info("Initializing Dolphin environment...")

    {:ok, port} = MeleePort.start_link()
    :ok = MeleePort.init_console(port, %{
      dolphin_path: config.dolphin,
      iso_path: config.iso,
      character: config.character,
      stage: config.stage
    })

    {:ok, %{type: :dolphin, port: port, prev_state: nil}}
  end

  def get_state(%{type: :mock} = env) do
    # Generate random-ish game state for testing
    {:ok, mock_game_state(env)}
  end

  def get_state(%{type: :dolphin, port: port}) do
    MeleePort.step(port)
  end

  def step_env(%{type: :mock} = env, _action) do
    new_step = env.step + 1
    done = rem(new_step, 3600) == 0  # Episode ends every 60 seconds
    episode = if done, do: env.episode + 1, else: env.episode

    # Random reward for testing
    reward = (:rand.uniform() - 0.5) * 0.1

    new_env = %{env | step: new_step, episode: episode}
    {:ok, new_env, reward, if(done, do: 1.0, else: 0.0)}
  end

  def step_env(%{type: :dolphin, port: port, prev_state: prev} = env, action) do
    # Send action to Dolphin
    controller = action_to_controller(action)
    :ok = MeleePort.send_controller(port, controller)

    # Get next state
    {:ok, next_state} = MeleePort.step(port)

    # Compute reward
    reward = if prev do
      Rewards.compute_reward(prev, next_state, player_port: 1)
    else
      0.0
    end

    # Check if episode done (stock lost or game over)
    done = episode_done?(next_state)

    new_env = %{env | prev_state: next_state}
    {:ok, new_env, reward, if(done, do: 1.0, else: 0.0)}
  end

  defp action_to_controller(action) do
    %ControllerState{
      buttons: action.buttons,
      main_x: action.main_x / 16.0,  # Convert from bucket to 0-1
      main_y: action.main_y / 16.0,
      c_x: action.c_x / 16.0,
      c_y: action.c_y / 16.0,
      l_shoulder: if(action.shoulder > 0, do: action.shoulder / 4.0, else: 0.0),
      r_shoulder: 0.0
    }
  end

  defp episode_done?(game_state) do
    # Episode ends when player loses a stock or game ends
    player = game_state.players[1]
    player.stock == 0 or game_state.frame > 8 * 60 * 60  # 8 minute timeout
  end

  defp mock_game_state(env) do
    # Generate plausible game state for testing
    %GameState{
      frame: env.step,
      stage: 2,  # Final Destination
      players: %{
        1 => %Player{
          x: :rand.uniform() * 100 - 50,
          y: :rand.uniform() * 50,
          percent: :rand.uniform() * 100,
          stock: 4 - div(env.step, 3600),
          facing: Enum.random([1, -1]),
          character: 9,  # Mewtwo
          action: Enum.random([0, 14, 20, 30]),
          action_frame: rem(env.step, 30),
          invulnerable: false,
          jumps_left: Enum.random([0, 1, 2]),
          on_ground: :rand.uniform() > 0.3,
          shield_strength: 60.0
        },
        2 => %Player{
          x: :rand.uniform() * 100 - 50,
          y: :rand.uniform() * 50,
          percent: :rand.uniform() * 100,
          stock: 4,
          facing: Enum.random([1, -1]),
          character: 2,  # Fox
          action: Enum.random([0, 14, 20, 30]),
          action_frame: rem(env.step, 30),
          invulnerable: false,
          jumps_left: 2,
          on_ground: true,
          shield_strength: 60.0
        }
      }
    }
  end
end

# Import helper
import PPOScript

# ============================================================================
# Main Script
# ============================================================================

Output.banner("ExPhil PPO Training")
Output.config([
  {"Pretrained", config.pretrained || "none (random init)"},
  {"Total Steps", config.timesteps},
  {"Rollout Length", config.rollout_length},
  {"PPO Epochs", config.num_epochs},
  {"Batch Size", config.batch_size},
  {"Learning Rate", config.lr},
  {"Checkpoint", config.checkpoint},
  {"Environment", if(config.mock, do: "mock", else: "dolphin")},
  {"Character", config.character},
  {"Opponent", config.opponent}
])

# Validate required args for Dolphin mode
if not config.mock do
  unless config.dolphin && config.iso do
    Output.error("Dolphin mode requires --dolphin and --iso paths.")
    Output.puts("")
    Output.puts("Either provide Dolphin paths:")
    Output.puts("  mix run scripts/train_ppo.exs --dolphin /path/to/slippi --iso /path/to/melee.iso")
    Output.puts("")
    Output.puts("Or use mock mode for testing:")
    Output.puts("  mix run scripts/train_ppo.exs --mock --pretrained checkpoints/policy.bin")
    System.halt(1)
  end
end

Output.step(1, 3, "Initializing PPO trainer")

# Build trainer options
trainer_opts = [
  learning_rate: config.lr,
  batch_size: config.batch_size,
  num_epochs: config.num_epochs,
  rollout_length: config.rollout_length
]

trainer_opts = if config.pretrained do
  Keyword.put(trainer_opts, :pretrained_path, config.pretrained)
else
  trainer_opts
end

trainer_opts = if config.hidden_sizes do
  Keyword.put(trainer_opts, :hidden_sizes, config.hidden_sizes)
else
  trainer_opts
end

trainer = PPO.new(trainer_opts)
Output.puts("  PPO trainer initialized")
Output.puts("  Embed size: #{trainer.config.embed_size}")

Output.step(2, 3, "Initializing environment")

env_type = if config.mock, do: :mock, else: :dolphin
{:ok, env} = init_env(env_type, config)
Output.puts("  Environment ready")

Output.step(3, 3, "Training for #{config.timesteps} timesteps")
Output.divider()

# Training loop
num_updates = div(config.timesteps, config.rollout_length)
start_time = System.monotonic_time(:millisecond)

{final_trainer, final_env} = Enum.reduce(1..num_updates, {trainer, env}, fn update, {acc_trainer, acc_env} ->
  # Collect rollout
  rollout = collect_rollout(acc_env, nil, acc_trainer, config.rollout_length)

  # PPO update
  {new_trainer, metrics} = PPO.update(acc_trainer, rollout)

  # Log progress
  timesteps = update * config.rollout_length
  pct = Float.round(timesteps / config.timesteps * 100, 1)

  if rem(update, 10) == 0 or update == 1 do
    elapsed = (System.monotonic_time(:millisecond) - start_time) / 1000
    fps = timesteps / max(elapsed, 1)

    Output.puts("  Update #{update}/#{num_updates} (#{pct}%) | " <>
            "policy_loss: #{Float.round(metrics.policy_loss || 0.0, 4)} | " <>
            "value_loss: #{Float.round(metrics.value_loss || 0.0, 4)} | " <>
            "entropy: #{Float.round(metrics.entropy || 0.0, 4)} | " <>
            "#{Float.round(fps, 0)} steps/s")
  end

  # Save checkpoint periodically
  if rem(update, 50) == 0 do
    PPO.save_checkpoint(new_trainer, config.checkpoint)
    Logger.info("Saved checkpoint to #{config.checkpoint}")
  end

  {new_trainer, rollout.env}
end)

# Final save
:ok = PPO.save_checkpoint(final_trainer, config.checkpoint)
:ok = PPO.export_policy(final_trainer, String.replace(config.checkpoint, ".axon", "_policy.bin"))

elapsed = (System.monotonic_time(:millisecond) - start_time) / 1000

Output.divider()
Output.section("Training Complete!")
Output.puts("")
Output.training_summary(%{
  total_time_ms: elapsed * 1000,
  epochs_completed: num_updates,
  final_loss: 0.0,
  checkpoint_path: config.checkpoint
})
Output.puts("  Timesteps:      #{config.timesteps}")
Output.puts("  Policy:         #{String.replace(config.checkpoint, ".axon", "_policy.bin")}")
Output.puts("")
Output.puts("To evaluate:")
Output.puts("  mix run scripts/eval_model.exs --policy #{String.replace(config.checkpoint, ".axon", "_policy.bin")}")
Output.puts("")
Output.puts("To play in Dolphin:")
Output.puts("  mix run scripts/play_dolphin.exs --policy #{String.replace(config.checkpoint, ".axon", "_policy.bin")}")
