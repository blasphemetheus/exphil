defmodule ExPhil.Training.SelfPlay.SelfPlayEnv do
  @moduledoc """
  Self-play environment wrapper for training.

  Manages a game between two agents (P1 = learner, P2 = opponent) and
  collects experience from P1's perspective.

  ## Architecture

      ┌─────────────────────────────────────────────────────────┐
      │                    SelfPlayEnv                          │
      │                                                         │
      │  ┌─────────────┐              ┌─────────────┐          │
      │  │   Agent P1  │              │  Agent P2   │          │
      │  │  (learner)  │              │ (opponent)  │          │
      │  └──────┬──────┘              └──────┬──────┘          │
      │         │                            │                  │
      │         ▼                            ▼                  │
      │  ┌─────────────────────────────────────────────────┐   │
      │  │                 Game State                       │   │
      │  │   (Dolphin via MeleePort or Mock)               │   │
      │  └─────────────────────────────────────────────────┘   │
      │                         │                               │
      │                         ▼                               │
      │  ┌─────────────────────────────────────────────────┐   │
      │  │              Experience Buffer                   │   │
      │  │  (states, actions, rewards from P1 perspective) │   │
      │  └─────────────────────────────────────────────────┘   │
      │                                                         │
      └─────────────────────────────────────────────────────────┘

  ## Usage

      # Create environment with opponent
      {:ok, env} = SelfPlayEnv.new(
        p1_policy: current_policy,
        p2_policy: opponent_policy,  # or :cpu for CPU opponent
        p2_cpu_level: 7
      )

      # Step environment (both agents act)
      {:ok, env, experience} = SelfPlayEnv.step(env)
      # experience = %{state: ..., action: ..., reward: ..., done: ...}

      # Collect full episode
      {:ok, env, trajectory} = SelfPlayEnv.collect_episode(env)

  """

  alias ExPhil.Bridge.{MeleePort, GameState, ControllerState}
  alias ExPhil.{Embeddings, Rewards}

  require Logger

  defstruct [
    :game,                 # Game interface (:mock or MeleePort pid)
    :game_type,            # :mock or :dolphin
    :p1_policy,            # P1 policy (learner) - {model, params, predict_fn}
    :p2_policy,            # P2 policy (opponent) - {model, params, predict_fn} or :cpu
    :p2_cpu_level,         # CPU level if p2_policy is :cpu
    :embed_config,         # Embedding configuration
    :reward_config,        # Reward configuration
    :prev_state,           # Previous game state (for reward computation)
    :frame_count,          # Frames in current episode
    :episode_count,        # Total episodes
    :p1_port,              # P1 controller port (default: 1)
    :p2_port,              # P2 controller port (default: 2)
    :config                # Additional configuration
  ]

  @type t :: %__MODULE__{}

  @default_config %{
    max_episode_frames: 28800,   # 8 minutes at 60fps
    frame_skip: 1,               # Act every N frames
    deterministic_opponent: true # Opponent uses argmax vs sampling
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Create a new self-play environment.

  ## Options
    - `:p1_policy` - P1 policy as `{model, params}` (required)
    - `:p2_policy` - P2 policy as `{model, params}` or `:cpu` (required)
    - `:p2_cpu_level` - CPU level 1-9 if p2_policy is :cpu (default: 7)
    - `:game_type` - `:mock` or `:dolphin` (default: :mock)
    - `:dolphin_config` - Config for Dolphin (path, iso, character, stage)
    - `:embed_config` - Embedding config (default: standard)
    - `:reward_config` - Reward config (default: standard)
    - `:p1_port` - P1 controller port (default: 1)
    - `:p2_port` - P2 controller port (default: 2)
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, term()}
  def new(opts) do
    p1_policy = Keyword.fetch!(opts, :p1_policy)
    p2_policy = Keyword.fetch!(opts, :p2_policy)
    game_type = Keyword.get(opts, :game_type, :mock)

    embed_config = Keyword.get_lazy(opts, :embed_config, fn ->
      Embeddings.config([])
    end)

    reward_config = Keyword.get(opts, :reward_config, Rewards.default_config())
    config = Map.merge(@default_config, Map.new(Keyword.get(opts, :config, [])))

    # Build predict functions
    p1_compiled = compile_policy(p1_policy)
    p2_compiled = if p2_policy == :cpu, do: :cpu, else: compile_policy(p2_policy)

    # Initialize game
    game = case game_type do
      :mock ->
        init_mock_game()

      :dolphin ->
        dolphin_config = Keyword.fetch!(opts, :dolphin_config)
        init_dolphin_game(dolphin_config)
    end

    env = %__MODULE__{
      game: game,
      game_type: game_type,
      p1_policy: p1_compiled,
      p2_policy: p2_compiled,
      p2_cpu_level: Keyword.get(opts, :p2_cpu_level, 7),
      embed_config: embed_config,
      reward_config: reward_config,
      prev_state: nil,
      frame_count: 0,
      episode_count: 0,
      p1_port: Keyword.get(opts, :p1_port, 1),
      p2_port: Keyword.get(opts, :p2_port, 2),
      config: config
    }

    {:ok, env}
  end

  @doc """
  Step the environment once.

  Both agents observe the game state and produce actions.
  Returns experience from P1's perspective.
  """
  @spec step(t()) :: {:ok, t(), map()} | {:done, t(), map()}
  def step(%__MODULE__{} = env) do
    # Get current game state
    {:ok, game_state} = get_game_state(env)

    # Get P1 action (learner)
    {p1_action, p1_log_prob, p1_value} = get_p1_action(env, game_state)

    # Get P2 action (opponent)
    p2_action = get_p2_action(env, game_state)

    # Apply actions to game
    {:ok, new_game} = apply_actions(env, p1_action, p2_action)

    # Get next state and compute reward
    {:ok, next_state} = get_game_state(%{env | game: new_game})
    reward = compute_reward(env, game_state, next_state)

    # Check if episode is done
    done = is_episode_done(env, next_state)

    # Build experience
    embedded_state = Embeddings.Game.embed(game_state, env.embed_config)
    experience = %{
      state: embedded_state,
      action: p1_action,
      log_prob: p1_log_prob,
      value: p1_value,
      reward: reward,
      done: done
    }

    # Update environment
    new_env = %{env |
      game: new_game,
      prev_state: game_state,
      frame_count: env.frame_count + 1
    }

    if done do
      {:done, handle_episode_end(new_env), experience}
    else
      {:ok, new_env, experience}
    end
  end

  @doc """
  Collect a full episode of experience.
  """
  @spec collect_episode(t()) :: {:ok, t(), [map()]}
  def collect_episode(%__MODULE__{} = env) do
    collect_episode_loop(env, [])
  end

  @doc """
  Collect N steps of experience (may span multiple episodes).
  """
  @spec collect_steps(t(), non_neg_integer()) :: {:ok, t(), [map()]}
  def collect_steps(%__MODULE__{} = env, n) do
    collect_steps_loop(env, n, [])
  end

  @doc """
  Reset environment for new episode.
  """
  @spec reset(t()) :: {:ok, t()}
  def reset(%__MODULE__{} = env) do
    new_game = case env.game_type do
      :mock -> init_mock_game()
      :dolphin -> reset_dolphin_game(env.game)
    end

    {:ok, %{env |
      game: new_game,
      prev_state: nil,
      frame_count: 0,
      episode_count: env.episode_count + 1
    }}
  end

  @doc """
  Update P1 policy (after training step).
  """
  @spec update_p1_policy(t(), {Axon.t(), map()}) :: t()
  def update_p1_policy(%__MODULE__{} = env, policy) do
    %{env | p1_policy: compile_policy(policy)}
  end

  @doc """
  Update P2 opponent policy.
  """
  @spec update_p2_policy(t(), {Axon.t(), map()} | :cpu) :: t()
  def update_p2_policy(%__MODULE__{} = env, :cpu) do
    %{env | p2_policy: :cpu}
  end

  def update_p2_policy(%__MODULE__{} = env, policy) do
    %{env | p2_policy: compile_policy(policy)}
  end

  @doc """
  Update opponent from OpponentPool sample.

  Handles different opponent types:
  - %{type: :cpu, level: N} -> Sets CPU opponent
  - %{type: :current, params: params} -> Sets current policy
  - %{type: :historical, params: params} -> Sets historical policy
  """
  @spec update_opponent(t(), map()) :: t()
  def update_opponent(%__MODULE__{} = env, %{type: :cpu, level: level}) do
    %{env | p2_policy: :cpu, p2_cpu_level: level}
  end

  def update_opponent(%__MODULE__{} = env, %{params: params}) when not is_nil(params) do
    # Historical or current policy with params
    {model, _, _} = env.p1_policy  # Reuse P1's model architecture
    update_p2_policy(env, {model, params})
  end

  def update_opponent(%__MODULE__{} = env, _opponent) do
    # Fallback to CPU level 7
    %{env | p2_policy: :cpu, p2_cpu_level: 7}
  end

  @doc """
  Shutdown environment and release resources.
  """
  @spec shutdown(t()) :: :ok
  def shutdown(%__MODULE__{game_type: :mock}), do: :ok

  def shutdown(%__MODULE__{game_type: :dolphin, game: port}) do
    if is_pid(port) and Process.alive?(port) do
      MeleePort.stop(port)
    end
    :ok
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp compile_policy({model, params}) do
    {_init_fn, predict_fn} = Axon.build(model, mode: :inference)
    {model, params, predict_fn}
  end

  defp init_mock_game do
    %{
      type: :mock,
      frame: 0,
      p1: mock_player(1),
      p2: mock_player(2),
      done: false
    }
  end

  defp mock_player(port) do
    %{
      port: port,
      x: if(port == 1, do: -20.0, else: 20.0),
      y: 0.0,
      percent: 0.0,
      stock: 4,
      facing: if(port == 1, do: 1, else: -1),
      action: 14,  # Wait
      action_frame: 0
    }
  end

  defp init_dolphin_game(config) do
    {:ok, port} = MeleePort.start_link()

    :ok = MeleePort.init_console(port, %{
      dolphin_path: config.dolphin_path,
      iso_path: config.iso_path,
      character: config.character || "fox",
      stage: config.stage || "final_destination"
    })

    port
  end

  defp reset_dolphin_game(port) do
    # TODO: Implement Dolphin reset when MeleePort is ready
    # For now, just return the port (game continues from current state)
    port
  end

  defp get_game_state(%{game_type: :mock, game: game}) do
    {:ok, mock_to_game_state(game)}
  end

  defp get_game_state(%{game_type: :dolphin, game: port}) do
    MeleePort.step(port)
  end

  defp mock_to_game_state(mock) do
    %GameState{
      frame: mock.frame,
      stage: 2,  # Final Destination
      menu_state: 2,  # In-game
      players: %{
        1 => mock_to_player(mock.p1),
        2 => mock_to_player(mock.p2)
      },
      projectiles: []
    }
  end

  defp mock_to_player(p) do
    %ExPhil.Bridge.Player{
      x: p.x,
      y: p.y,
      percent: p.percent,
      stock: p.stock,
      facing: p.facing,
      action: p.action,
      action_frame: p.action_frame,
      shield_strength: 60.0,
      character: 2,  # Fox
      invulnerable: false,
      hitstun_frames_left: 0,
      jumps_left: 2,
      on_ground: true,
      speed_air_x_self: 0.0,
      speed_y_self: 0.0,
      speed_x_attack: 0.0,
      speed_y_attack: 0.0,
      speed_ground_x_self: 0.0,
      nana: nil,
      controller_state: nil
    }
  end

  defp get_p1_action(env, game_state) do
    {_model, params, predict_fn} = env.p1_policy
    embedded = Embeddings.Game.embed(game_state, env.embed_config)
    input = Nx.new_axis(embedded, 0)

    # Forward pass
    output = predict_fn.(params, input)

    # Handle different output formats
    {policy_logits, value} = case output do
      %{policy: p, value: v} -> {p, v}
      {p, v} when is_tuple(p) -> {p, v}
      tuple when is_tuple(tuple) ->
        # Assume last element is value, rest is policy
        policy = Tuple.delete_at(tuple, tuple_size(tuple) - 1)
        {policy, elem(tuple, tuple_size(tuple) - 1)}
    end

    # Sample action
    action = sample_action(policy_logits, deterministic: false)

    # Compute log probability
    log_prob = compute_log_prob(policy_logits, action)

    {action, log_prob, Nx.squeeze(value)}
  end

  defp get_p2_action(%{p2_policy: :cpu}, _game_state) do
    # CPU opponent - return nil, Dolphin handles CPU
    nil
  end

  defp get_p2_action(env, game_state) do
    {_model, params, predict_fn} = env.p2_policy
    deterministic = env.config.deterministic_opponent

    # Embed from P2's perspective (swap player ports)
    swapped_state = swap_player_perspective(game_state, env.p1_port, env.p2_port)
    embedded = Embeddings.Game.embed(swapped_state, env.embed_config)
    input = Nx.new_axis(embedded, 0)

    output = predict_fn.(params, input)

    policy_logits = case output do
      %{policy: p} -> p
      {p, _v} when is_tuple(p) -> p
      tuple when is_tuple(tuple) -> Tuple.delete_at(tuple, tuple_size(tuple) - 1)
    end

    sample_action(policy_logits, deterministic: deterministic)
  end

  defp swap_player_perspective(game_state, p1_port, p2_port) do
    # Swap players so P2 sees itself as "player 1"
    players = game_state.players
    swapped = %{
      p1_port => Map.get(players, p2_port),
      p2_port => Map.get(players, p1_port)
    }
    %{game_state | players: swapped}
  end

  defp sample_action(policy_logits, opts) do
    deterministic = Keyword.get(opts, :deterministic, false)

    {buttons, main_x, main_y, c_x, c_y, shoulder} = policy_logits

    %{
      buttons: sample_buttons(buttons, deterministic),
      main_x: sample_categorical(main_x, deterministic),
      main_y: sample_categorical(main_y, deterministic),
      c_x: sample_categorical(c_x, deterministic),
      c_y: sample_categorical(c_y, deterministic),
      shoulder: sample_categorical(shoulder, deterministic)
    }
  end

  defp sample_buttons(logits, deterministic) do
    probs = Nx.sigmoid(logits) |> Nx.squeeze()

    if deterministic do
      # Threshold at 0.5
      Nx.greater(probs, 0.5)
    else
      # Sample from Bernoulli
      key = Nx.Random.key(System.system_time())
      {uniform, _} = Nx.Random.uniform(key, shape: Nx.shape(probs))
      Nx.less(uniform, probs)
    end
  end

  defp sample_categorical(logits, deterministic) do
    if deterministic do
      Nx.argmax(logits, axis: -1) |> Nx.squeeze()
    else
      probs = Axon.Activations.softmax(logits, axis: -1) |> Nx.squeeze()
      # Sample using Gumbel-max trick
      key = Nx.Random.key(System.system_time())
      {uniform, _} = Nx.Random.uniform(key, shape: Nx.shape(probs))
      gumbel = Nx.negate(Nx.log(Nx.negate(Nx.log(uniform))))
      Nx.argmax(Nx.add(Nx.log(probs), gumbel), axis: -1)
    end
  end

  defp compute_log_prob(policy_logits, action) do
    {buttons, main_x, main_y, c_x, c_y, shoulder} = policy_logits

    # Button log probs (Bernoulli)
    btn_probs = Nx.sigmoid(buttons) |> Nx.squeeze()
    btn_action = action.buttons
    btn_log_prob = Nx.sum(
      Nx.add(
        Nx.multiply(btn_action, Nx.log(Nx.add(btn_probs, 1.0e-8))),
        Nx.multiply(Nx.subtract(1, btn_action), Nx.log(Nx.add(Nx.subtract(1, btn_probs), 1.0e-8)))
      )
    )

    # Categorical log probs
    cat_log_prob = fn logits, action_idx ->
      log_probs = Axon.Activations.log_softmax(logits, axis: -1) |> Nx.squeeze()
      Nx.take(log_probs, action_idx)
    end

    total_log_prob = btn_log_prob
    |> Nx.add(cat_log_prob.(main_x, action.main_x))
    |> Nx.add(cat_log_prob.(main_y, action.main_y))
    |> Nx.add(cat_log_prob.(c_x, action.c_x))
    |> Nx.add(cat_log_prob.(c_y, action.c_y))
    |> Nx.add(cat_log_prob.(shoulder, action.shoulder))

    total_log_prob
  end

  defp apply_actions(%{game_type: :mock} = env, p1_action, _p2_action) do
    # Mock environment: simulate simple physics
    game = env.game

    new_game = %{game |
      frame: game.frame + 1,
      p1: update_mock_player(game.p1, p1_action),
      p2: update_mock_player(game.p2, nil)  # Simple AI for P2 in mock
    }

    {:ok, new_game}
  end

  defp apply_actions(%{game_type: :dolphin, game: port}, p1_action, p2_action) do
    # Send actions to Dolphin
    MeleePort.send_controller(port, 1, action_to_controller(p1_action))

    if p2_action do
      MeleePort.send_controller(port, 2, action_to_controller(p2_action))
    end

    {:ok, port}
  end

  defp update_mock_player(player, nil) do
    # Simple mock AI: move toward center
    dx = if player.x > 0, do: -0.5, else: 0.5
    %{player | x: player.x + dx}
  end

  defp update_mock_player(player, action) do
    # Apply action to mock player
    main_x_val = Nx.to_number(action.main_x)
    dx = (main_x_val - 8) / 8.0 * 2.0  # Map 0-16 to -2..2

    %{player | x: player.x + dx}
  end

  defp action_to_controller(action) do
    %ControllerState{
      main_stick: %{
        x: (Nx.to_number(action.main_x) / 16.0),
        y: (Nx.to_number(action.main_y) / 16.0)
      },
      c_stick: %{
        x: (Nx.to_number(action.c_x) / 16.0),
        y: (Nx.to_number(action.c_y) / 16.0)
      },
      l_shoulder: Nx.to_number(action.shoulder) / 4.0,
      r_shoulder: 0.0,
      button_a: button_pressed?(action.buttons, 0),
      button_b: button_pressed?(action.buttons, 1),
      button_x: button_pressed?(action.buttons, 2),
      button_y: button_pressed?(action.buttons, 3),
      button_z: button_pressed?(action.buttons, 4),
      button_l: button_pressed?(action.buttons, 5),
      button_r: button_pressed?(action.buttons, 6),
      button_d_up: button_pressed?(action.buttons, 7)
    }
  end

  defp button_pressed?(buttons, idx) do
    buttons |> Nx.squeeze() |> Nx.slice([idx], [1]) |> Nx.to_number() > 0.5
  end

  defp compute_reward(env, prev_state, curr_state) do
    Rewards.compute_weighted(prev_state, curr_state, env.reward_config, player_port: env.p1_port)
  end

  defp is_episode_done(env, game_state) do
    # Episode ends when:
    # 1. Game is over (someone lost all stocks)
    # 2. Max frames reached
    cond do
      env.frame_count >= env.config.max_episode_frames ->
        true

      game_state.menu_state != 2 ->
        # Not in-game
        true

      true ->
        # Check stocks
        p1 = Map.get(game_state.players, env.p1_port)
        p2 = Map.get(game_state.players, env.p2_port)
        (p1 && p1.stock <= 0) || (p2 && p2.stock <= 0)
    end
  end

  defp handle_episode_end(env) do
    Logger.debug("Episode #{env.episode_count + 1} ended at frame #{env.frame_count}")
    %{env | episode_count: env.episode_count + 1, frame_count: 0}
  end

  defp collect_episode_loop(env, experiences) do
    case step(env) do
      {:ok, new_env, exp} ->
        collect_episode_loop(new_env, [exp | experiences])

      {:done, new_env, exp} ->
        {:ok, new_env, Enum.reverse([exp | experiences])}
    end
  end

  defp collect_steps_loop(env, 0, experiences) do
    {:ok, env, Enum.reverse(experiences)}
  end

  defp collect_steps_loop(env, remaining, experiences) do
    case step(env) do
      {:ok, new_env, exp} ->
        collect_steps_loop(new_env, remaining - 1, [exp | experiences])

      {:done, new_env, exp} ->
        # Reset and continue
        {:ok, reset_env} = reset(new_env)
        collect_steps_loop(reset_env, remaining - 1, [exp | experiences])
    end
  end
end
