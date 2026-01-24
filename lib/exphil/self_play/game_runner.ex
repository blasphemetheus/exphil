defmodule ExPhil.SelfPlay.GameRunner do
  @moduledoc """
  GenServer managing a single self-play game.

  Each GameRunner encapsulates one game instance (mock or Dolphin),
  handling the game loop, action sampling, and experience collection.

  ## Architecture

      ┌─────────────────────────────────────────────────────────────┐
      │                      GameRunner                              │
      │                                                              │
      │  ┌──────────────┐          ┌──────────────┐                 │
      │  │  P1 Policy   │          │  P2 Policy   │                 │
      │  │  (learner)   │          │  (opponent)  │                 │
      │  └──────┬───────┘          └──────┬───────┘                 │
      │         │                          │                         │
      │         └────────────┬─────────────┘                         │
      │                      ▼                                       │
      │         ┌──────────────────────────┐                        │
      │         │     Game State Manager    │                        │
      │         │   (Dolphin/Mock adapter)  │                        │
      │         └──────────────────────────┘                        │
      │                      │                                       │
      │                      ▼                                       │
      │         ┌──────────────────────────┐                        │
      │         │   Experience Collector    │                        │
      │         │  (via callback or direct) │                        │
      │         └──────────────────────────┘                        │
      │                                                              │
      └─────────────────────────────────────────────────────────────┘

  ## Usage

      # Start a game runner
      {:ok, runner} = GameRunner.start_link(
        game_id: "game_1",
        p1_policy_id: :current,
        p2_policy_id: :historical_v5,
        game_type: :mock
      )

      # Start the game (returns game state)
      :ok = GameRunner.start_game(runner)

      # Step the game and get experience
      {:ok, experience} = GameRunner.step(runner)

      # Swap opponent mid-game
      :ok = GameRunner.swap_policy(runner, :p2, :historical_v10)

      # Get current status
      status = GameRunner.get_status(runner)
      # => %{status: :playing, frame: 1234, episode_reward: 5.2}

  """

  use GenServer

  alias ExPhil.Bridge.ControllerState
  alias ExPhil.{Embeddings, Rewards}
  alias ExPhil.MockEnv.Game, as: MockGame
  alias ExPhil.SelfPlay.PopulationManager
  alias ExPhil.Training.Utils

  require Logger

  @type status :: :waiting | :playing | :finished
  @type game_type :: :mock | :dolphin

  defstruct [
    :game_id,
    :dolphin_pid,        # MeleePort process pid (or :mock for mock games)
    :game_type,
    :p1_policy_id,
    :p2_policy_id,
    :p1_policy,          # Compiled policy {model, params, predict_fn}
    :p2_policy,          # Compiled policy or :cpu
    :p2_cpu_level,
    :embed_config,
    :reward_config,
    :prev_state,
    :prev_action,        # Previous controller action (for embedding)
    :current_state,
    :frame_count,
    :episode_count,
    :episode_reward,
    :status,
    :p1_port,
    :p2_port,
    :config,
    :population_manager,  # PID of PopulationManager
    :experience_collector # PID of ExperienceCollector (optional)
  ]

  @default_config %{
    max_episode_frames: 28800,     # 8 minutes at 60fps
    frame_skip: 1,
    deterministic_opponent: true,
    auto_reset: true               # Auto-reset episode on done
  }

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts a GameRunner process.

  ## Options
    - `:game_id` - Unique identifier for this game (required)
    - `:p1_policy_id` - Policy ID for P1 (required)
    - `:p2_policy_id` - Policy ID for P2 (required)
    - `:game_type` - `:mock` or `:dolphin` (default: :mock)
    - `:population_manager` - PID of PopulationManager (optional)
    - `:experience_collector` - PID of ExperienceCollector (optional)
    - `:p1_port` - Controller port for P1 (default: 1)
    - `:p2_port` - Controller port for P2 (default: 2)
    - `:dolphin_config` - Config for Dolphin mode
  """
  def start_link(opts) do
    game_id = Keyword.fetch!(opts, :game_id)
    name = via_tuple(game_id)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Starts the game. Must be called before stepping.
  """
  def start_game(runner) do
    GenServer.call(runner, :start_game)
  end

  @doc """
  Steps the game forward by one frame.

  Returns experience tuple from P1's perspective:
  `{:ok, experience}` where experience contains state, action, reward, done, etc.
  """
  def step(runner) do
    GenServer.call(runner, :step, 30_000)
  end

  @doc """
  Collects N steps of experience.
  """
  def collect_steps(runner, n) do
    GenServer.call(runner, {:collect_steps, n}, 60_000)
  end

  @doc """
  Swaps the policy for the specified port.
  """
  def swap_policy(runner, port, policy_id) when port in [:p1, :p2] do
    GenServer.call(runner, {:swap_policy, port, policy_id})
  end

  @doc """
  Gets the current game status.
  """
  def get_status(runner) do
    GenServer.call(runner, :get_status)
  end

  @doc """
  Resets the game to initial state.
  """
  def reset(runner) do
    GenServer.call(runner, :reset)
  end

  @doc """
  Stops the game runner.
  """
  def stop(runner) do
    GenServer.stop(runner, :normal)
  end

  @doc """
  Gets a game runner by ID.
  """
  def whereis(game_id) do
    case Registry.lookup(ExPhil.SelfPlay.GameRegistry, game_id) do
      [{pid, _}] -> pid
      [] -> nil
    end
  end

  defp via_tuple(game_id) do
    {:via, Registry, {ExPhil.SelfPlay.GameRegistry, game_id}}
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    game_id = Keyword.fetch!(opts, :game_id)
    game_type = Keyword.get(opts, :game_type, :mock)
    p1_policy_id = Keyword.fetch!(opts, :p1_policy_id)
    p2_policy_id = Keyword.fetch!(opts, :p2_policy_id)

    embed_config = Keyword.get_lazy(opts, :embed_config, fn ->
      Embeddings.config([])
    end)

    reward_config = Keyword.get(opts, :reward_config, Rewards.default_config())
    config = Map.merge(@default_config, Map.new(Keyword.get(opts, :config, [])))

    state = %__MODULE__{
      game_id: game_id,
      game_type: game_type,
      dolphin_pid: nil,
      p1_policy_id: p1_policy_id,
      p2_policy_id: p2_policy_id,
      p1_policy: nil,
      p2_policy: nil,
      p2_cpu_level: Keyword.get(opts, :p2_cpu_level, 7),
      embed_config: embed_config,
      reward_config: reward_config,
      prev_state: nil,
      current_state: nil,
      frame_count: 0,
      episode_count: 0,
      episode_reward: 0.0,
      status: :waiting,
      p1_port: Keyword.get(opts, :p1_port, 1),
      p2_port: Keyword.get(opts, :p2_port, 2),
      config: config,
      population_manager: Keyword.get(opts, :population_manager),
      experience_collector: Keyword.get(opts, :experience_collector)
    }

    Logger.debug("[GameRunner #{game_id}] Initialized (#{game_type})")

    {:ok, state}
  end

  @impl true
  def handle_call(:start_game, _from, state) do
    case do_start_game(state) do
      {:ok, new_state} ->
        Logger.debug("[GameRunner #{state.game_id}] Game started")
        {:reply, :ok, new_state}

      {:error, reason} = error ->
        Logger.error("[GameRunner #{state.game_id}] Failed to start: #{inspect(reason)}")
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call(:step, _from, %{status: :waiting} = state) do
    {:reply, {:error, :game_not_started}, state}
  end

  @impl true
  def handle_call(:step, _from, %{status: :finished} = state) do
    if state.config.auto_reset do
      {:ok, reset_state} = do_reset(state)
      do_step(reset_state)
    else
      {:reply, {:error, :game_finished}, state}
    end
  end

  @impl true
  def handle_call(:step, _from, state) do
    do_step(state)
  end

  @impl true
  def handle_call({:collect_steps, n}, _from, state) do
    {:ok, new_state, experiences} = do_collect_steps(state, n, [])
    {:reply, {:ok, experiences}, new_state}
  end

  @impl true
  def handle_call({:swap_policy, port, policy_id}, _from, state) do
    new_state = case port do
      :p1 ->
        policy = load_policy(state, policy_id)
        %{state | p1_policy_id: policy_id, p1_policy: policy}

      :p2 ->
        policy = load_policy(state, policy_id)
        %{state | p2_policy_id: policy_id, p2_policy: policy}
    end

    Logger.debug("[GameRunner #{state.game_id}] Swapped #{port} to #{inspect(policy_id)}")
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    status = %{
      game_id: state.game_id,
      status: state.status,
      frame_count: state.frame_count,
      episode_count: state.episode_count,
      episode_reward: state.episode_reward,
      p1_policy_id: state.p1_policy_id,
      p2_policy_id: state.p2_policy_id
    }
    {:reply, status, state}
  end

  @impl true
  def handle_call(:reset, _from, state) do
    {:ok, new_state} = do_reset(state)
    {:reply, :ok, new_state}
  end

  @impl true
  def terminate(_reason, state) do
    # Cleanup Dolphin if running
    if state.dolphin_pid && is_pid(state.dolphin_pid) && Process.alive?(state.dolphin_pid) do
      Logger.debug("[GameRunner #{state.game_id}] Shutting down Dolphin")
      ExPhil.Bridge.MeleePort.stop(state.dolphin_pid)
    end

    :ok
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp do_start_game(state) do
    # Load policies
    p1_policy = load_policy(state, state.p1_policy_id)
    p2_policy = load_policy(state, state.p2_policy_id)

    # Initialize game
    case init_game(state) do
      {:ok, game_pid, initial_state} ->
        new_state = %{state |
          dolphin_pid: game_pid,
          p1_policy: p1_policy,
          p2_policy: p2_policy,
          current_state: initial_state,
          status: :playing,
          frame_count: 0,
          episode_reward: 0.0
        }
        {:ok, new_state}

      {:error, _} = error ->
        error
    end
  end

  defp do_step(state) do
    # Get actions from both policies
    {p1_action, p1_log_prob, p1_value} = get_p1_action(state)
    p2_action = get_p2_action(state)

    # Apply actions - returns {game_state, game_or_pid}
    {:ok, new_game_state, new_game_or_pid} = apply_actions(state, p1_action, p2_action)

    # Compute reward
    reward = Rewards.compute_weighted(
      state.current_state,
      new_game_state,
      state.reward_config,
      player_port: state.p1_port
    )

    # Check if done
    done = is_episode_done(state, new_game_state)

    # Build experience (embed with previous action for consistency)
    embedded_state = Embeddings.Game.embed(
      state.current_state,
      state.prev_action,
      state.p1_port,
      config: state.embed_config
    )
    experience = %{
      state: embedded_state,
      action: p1_action,
      log_prob: p1_log_prob,
      value: p1_value,
      reward: reward,
      done: done
    }

    # Submit to collector if configured
    maybe_submit_experience(state, experience)

    # Update state - track prev_action and persist game/pid
    new_state = %{state |
      prev_state: state.current_state,
      prev_action: action_to_controller_state(p1_action),
      current_state: new_game_state,
      dolphin_pid: new_game_or_pid,  # Persist mock game or Dolphin pid
      frame_count: state.frame_count + 1,
      episode_reward: state.episode_reward + reward
    }

    new_state = if done do
      Logger.debug("[GameRunner #{state.game_id}] Episode #{state.episode_count + 1} done, reward: #{Float.round(new_state.episode_reward, 2)}")

      # Report result to matchmaker for Elo tracking
      result = determine_winner(new_game_state, state.p1_port, state.p2_port)
      Logger.debug("[GameRunner #{state.game_id}] Winner: #{inspect(result)}, p1=#{inspect(state.p1_policy_id)}, p2=#{inspect(state.p2_policy_id)}")
      if result do
        ExPhil.SelfPlay.Supervisor.report_game_result(
          state.p1_policy_id,
          state.p2_policy_id,
          result
        )
      end

      %{new_state | status: :finished, episode_count: state.episode_count + 1}
    else
      new_state
    end

    {:reply, {:ok, experience}, new_state}
  end

  defp do_collect_steps(state, 0, acc) do
    {:ok, state, Enum.reverse(acc)}
  end

  defp do_collect_steps(%{status: :waiting} = state, remaining, acc) do
    # Auto-start game if not yet started
    case do_start_game(state) do
      {:ok, started_state} ->
        do_collect_steps(started_state, remaining, acc)

      {:error, reason} ->
        Logger.error("[GameRunner #{state.game_id}] Failed to auto-start: #{inspect(reason)}")
        {:ok, state, Enum.reverse(acc)}
    end
  end

  defp do_collect_steps(%{status: :finished} = state, remaining, acc) do
    if state.config.auto_reset do
      {:ok, reset_state} = do_reset(state)
      do_collect_steps(reset_state, remaining, acc)
    else
      {:ok, state, Enum.reverse(acc)}
    end
  end

  defp do_collect_steps(state, remaining, acc) do
    {:reply, {:ok, exp}, new_state} = do_step(state)
    do_collect_steps(new_state, remaining - 1, [exp | acc])
  end

  defp do_reset(state) do
    new_game_or_pid = case state.game_type do
      :mock -> init_mock_game()
      :dolphin -> reset_dolphin_game(state.dolphin_pid)
    end

    {:ok, game_state} = get_game_state(state.game_type, new_game_or_pid)

    new_state = %{state |
      current_state: game_state,
      dolphin_pid: new_game_or_pid,  # Persist new mock game or dolphin pid
      prev_state: nil,
      prev_action: nil,  # Reset prev_action on episode reset
      frame_count: 0,
      episode_reward: 0.0,
      status: :playing
    }

    {:ok, new_state}
  end

  defp init_game(%{game_type: :mock}) do
    game = init_mock_game()
    {:ok, initial_state} = get_game_state(:mock, game)
    {:ok, :mock, initial_state}
  end

  defp init_game(%{game_type: :dolphin} = state) do
    dolphin_config = state.config[:dolphin_config] || %{}

    case ExPhil.Bridge.MeleePort.start_link() do
      {:ok, pid} ->
        :ok = ExPhil.Bridge.MeleePort.init_console(pid, %{
          dolphin_path: dolphin_config[:dolphin_path],
          iso_path: dolphin_config[:iso_path],
          character: dolphin_config[:character] || "fox",
          stage: dolphin_config[:stage] || "final_destination"
        })

        case ExPhil.Bridge.MeleePort.step(pid) do
          {:ok, initial_state} -> {:ok, pid, initial_state}
          error -> error
        end

      error ->
        error
    end
  end

  defp load_policy(_state, :cpu), do: :cpu

  # Handle {:cpu, level} tuples from PopulationManager
  defp load_policy(_state, {:cpu, _level}), do: :cpu

  # Handle {:historical, version} tuples - need to look up the actual model
  defp load_policy(state, {:historical, _version} = policy_id) do
    if state.population_manager do
      case PopulationManager.get_policy(state.population_manager, policy_id) do
        {:ok, {model, params}} -> compile_policy({model, params})
        {:error, _} -> :cpu  # Fallback to CPU
      end
    else
      :cpu
    end
  end

  defp load_policy(state, policy_id) when is_atom(policy_id) or is_binary(policy_id) do
    if state.population_manager do
      case PopulationManager.get_policy(state.population_manager, policy_id) do
        {:ok, {model, params}} -> compile_policy({model, params})
        {:error, _} -> :cpu  # Fallback to CPU
      end
    else
      :cpu
    end
  end

  defp load_policy(_state, {model, params}) do
    compile_policy({model, params})
  end

  defp compile_policy({model, params}) do
    {_init_fn, predict_fn} = Axon.build(model, mode: :inference)
    {model, params, predict_fn}
  end

  defp get_p1_action(%{p1_policy: :cpu}) do
    # Return random action for CPU
    {random_action(), Nx.tensor(0.0), Nx.tensor(0.0)}
  end

  defp get_p1_action(state) do
    {_model, params, predict_fn} = state.p1_policy
    # Game.embed signature: (game_state, prev_action, own_port, opts)
    embedded = Embeddings.Game.embed(
      state.current_state,
      state.prev_action,
      state.p1_port,
      config: state.embed_config
    )
    input = Nx.new_axis(embedded, 0)

    output = predict_fn.(Utils.ensure_model_state(params), input)

    {policy_logits, value} = parse_model_output(output)
    action = sample_action(policy_logits, deterministic: false)
    log_prob = compute_log_prob(policy_logits, action)

    {action, log_prob, Nx.squeeze(value)}
  end

  defp get_p2_action(%{p2_policy: :cpu}), do: nil

  defp get_p2_action(state) do
    {_model, params, predict_fn} = state.p2_policy
    deterministic = state.config.deterministic_opponent

    # Embed from P2's perspective (swapped view)
    # Note: P2 doesn't need prev_action tracking, use nil
    embedded = Embeddings.Game.embed(
      state.current_state,
      nil,  # P2's prev_action not tracked
      state.p2_port,
      config: state.embed_config
    )
    input = Nx.new_axis(embedded, 0)

    output = predict_fn.(Utils.ensure_model_state(params), input)
    {policy_logits, _value} = parse_model_output(output)

    sample_action(policy_logits, deterministic: deterministic)
  end

  defp parse_model_output(output) do
    case output do
      %{policy: p, value: v} -> {p, v}
      {p, v} when is_tuple(p) -> {p, v}
      tuple when is_tuple(tuple) ->
        policy = Tuple.delete_at(tuple, tuple_size(tuple) - 1)
        {policy, elem(tuple, tuple_size(tuple) - 1)}
    end
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
      Nx.greater(probs, 0.5)
    else
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

    cat_log_prob = fn logits, action_idx ->
      log_probs = Axon.Activations.log_softmax(logits, axis: -1) |> Nx.squeeze()
      Nx.take(log_probs, action_idx)
    end

    btn_log_prob
    |> Nx.add(cat_log_prob.(main_x, action.main_x))
    |> Nx.add(cat_log_prob.(main_y, action.main_y))
    |> Nx.add(cat_log_prob.(c_x, action.c_x))
    |> Nx.add(cat_log_prob.(c_y, action.c_y))
    |> Nx.add(cat_log_prob.(shoulder, action.shoulder))
  end

  defp random_action do
    %{
      buttons: Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
      main_x: Nx.tensor(8),
      main_y: Nx.tensor(8),
      c_x: Nx.tensor(8),
      c_y: Nx.tensor(8),
      shoulder: Nx.tensor(0)
    }
  end

  defp apply_actions(%{game_type: :mock} = state, p1_action, _p2_action) do
    game = state.dolphin_pid  # For mock, dolphin_pid holds MockGame struct

    # Use frame from current_state if available, otherwise from game
    current_frame = if state.current_state, do: state.current_state.frame, else: 0

    new_game = update_mock_game(game, p1_action, current_frame)
    {:ok, game_state} = get_game_state(:mock, new_game)

    # Return both game_state (for embedding) and new_game (to persist physics state)
    {:ok, game_state, new_game}
  end

  defp apply_actions(%{game_type: :dolphin, dolphin_pid: pid}, p1_action, p2_action) do
    ExPhil.Bridge.MeleePort.send_controller(pid, 1, action_to_controller(p1_action))

    if p2_action do
      ExPhil.Bridge.MeleePort.send_controller(pid, 2, action_to_controller(p2_action))
    end

    case ExPhil.Bridge.MeleePort.step(pid) do
      {:ok, game_state} -> {:ok, game_state, pid}
      error -> error
    end
  end

  defp action_to_controller(action) do
    %ControllerState{
      main_stick: %{
        x: Nx.to_number(action.main_x) / 16.0,
        y: Nx.to_number(action.main_y) / 16.0
      },
      c_stick: %{
        x: Nx.to_number(action.c_x) / 16.0,
        y: Nx.to_number(action.c_y) / 16.0
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

  defp is_episode_done(state, game_state) do
    cond do
      state.frame_count >= state.config.max_episode_frames -> true
      game_state.menu_state != 2 -> true
      true ->
        p1 = Map.get(game_state.players, state.p1_port)
        p2 = Map.get(game_state.players, state.p2_port)
        (p1 && p1.stock <= 0) || (p2 && p2.stock <= 0)
    end
  end

  # Determine winner based on stock counts
  # Returns :win (p1 wins), :loss (p2 wins), or nil (undetermined)
  defp determine_winner(game_state, p1_port, p2_port) do
    p1 = Map.get(game_state.players, p1_port)
    p2 = Map.get(game_state.players, p2_port)

    cond do
      p1 == nil or p2 == nil -> nil
      p2.stock <= 0 and p1.stock > 0 -> :win
      p1.stock <= 0 and p2.stock > 0 -> :loss
      # Tie-breaker: lower percent wins
      p1.stock <= 0 and p2.stock <= 0 ->
        if p1.percent <= p2.percent, do: :win, else: :loss
      # Game ended by timeout or other reason
      true ->
        cond do
          p1.stock > p2.stock -> :win
          p2.stock > p1.stock -> :loss
          p1.percent < p2.percent -> :win
          p2.percent < p1.percent -> :loss
          true -> :draw
        end
    end
  end

  defp maybe_submit_experience(%{experience_collector: nil}, _experience), do: :ok

  defp maybe_submit_experience(%{experience_collector: collector}, experience) do
    ExPhil.SelfPlay.ExperienceCollector.submit(collector, experience)
  end

  # ============================================================================
  # Mock Game Functions (using physics-based MockEnv.Game)
  # ============================================================================

  defp init_mock_game do
    MockGame.new()
  end

  defp get_game_state(:mock, game) do
    {:ok, MockGame.to_game_state(game)}
  end

  defp get_game_state(:dolphin, pid) do
    ExPhil.Bridge.MeleePort.step(pid)
  end

  defp update_mock_game(game_or_mock, p1_action, _current_frame) do
    # Handle both mock game struct and :mock atom (initial state)
    game = case game_or_mock do
      :mock -> init_mock_game()
      %MockGame{} = g -> g
      # Legacy map format fallback
      m when is_map(m) -> init_mock_game()
    end

    # Convert policy action to mock action format
    mock_p1_action = policy_action_to_mock(p1_action)

    # P2 uses simple AI: move toward P1, occasionally jump
    mock_p2_action = simple_opponent_action(game)

    # Step the physics simulation
    MockGame.step(game, mock_p1_action, mock_p2_action)
  end

  # Convert policy action (discretized Nx tensors) to mock action format (floats/booleans)
  defp policy_action_to_mock(nil), do: nil

  defp policy_action_to_mock(action) do
    # Buttons are indices 0-7: A, B, X, Y, Z, L, R, Start
    buttons = action.buttons |> Nx.to_flat_list()

    # Stick values are 0-16, convert to -1.0 to 1.0
    stick_x = (Nx.to_number(action.main_x) - 8) / 8.0
    stick_y = (Nx.to_number(action.main_y) - 8) / 8.0

    %{
      stick_x: stick_x,
      stick_y: stick_y,
      button_a: Enum.at(buttons, 0) == 1,
      button_b: Enum.at(buttons, 1) == 1,
      button_x: Enum.at(buttons, 2) == 1,
      button_y: Enum.at(buttons, 3) == 1,
      button_l: Enum.at(buttons, 5) == 1,
      button_r: Enum.at(buttons, 6) == 1
    }
  end

  # Simple opponent AI for mock games
  defp simple_opponent_action(game) do
    p1 = game.p1
    p2 = game.p2

    # Move toward P1
    dx = p1.x - p2.x
    stick_x = cond do
      dx > 5 -> 0.8
      dx < -5 -> -0.8
      true -> 0.0
    end

    # Occasionally jump (10% chance when on ground)
    jump = p2.on_ground and :rand.uniform() < 0.1

    %{
      stick_x: stick_x,
      stick_y: 0.0,
      button_a: false,
      button_b: false,
      button_x: jump,
      button_y: false,
      button_l: false,
      button_r: false
    }
  end

  defp reset_dolphin_game(pid) do
    # Reset Dolphin game by stepping through menus until back in game.
    # The Python bridge's auto_menu=true handles menu navigation automatically.
    wait_for_game_start(pid, _max_frames = 1800)  # 30 seconds at 60fps
  end

  defp wait_for_game_start(pid, remaining) when remaining <= 0 do
    Logger.warning("[GameRunner] Timeout waiting for game to restart")
    pid
  end

  defp wait_for_game_start(pid, remaining) do
    case ExPhil.Bridge.MeleePort.step(pid, auto_menu: true) do
      {:ok, _game_state} ->
        # Back in game!
        Logger.debug("[GameRunner] Game restarted successfully")
        pid

      {:postgame, _state} ->
        # Still in postgame, keep stepping (menu helper will advance)
        wait_for_game_start(pid, remaining - 1)

      {:menu, _state} ->
        # In menus (character select, stage select, etc)
        wait_for_game_start(pid, remaining - 1)

      {:game_ended, reason} ->
        Logger.warning("[GameRunner] Game ended during reset: #{reason}")
        pid

      {:error, reason} ->
        Logger.error("[GameRunner] Error during reset: #{inspect(reason)}")
        pid
    end
  end

  # Convert policy action map to ControllerState for embedding
  # Policy action has discretized indices, ControllerState needs floats/booleans
  defp action_to_controller_state(action) do
    buttons = action.buttons |> Nx.to_flat_list()

    # Convert discretized stick indices to 0.0-1.0 range
    # Assuming 17-position discretization (indices 0-16)
    stick_to_float = fn idx ->
      Nx.to_number(idx) / 16.0
    end

    # Shoulder has fewer positions (typically 4)
    shoulder_to_float = fn idx ->
      Nx.to_number(idx) / 3.0
    end

    %ControllerState{
      main_stick: %{
        x: stick_to_float.(action.main_x),
        y: stick_to_float.(action.main_y)
      },
      c_stick: %{
        x: stick_to_float.(action.c_x),
        y: stick_to_float.(action.c_y)
      },
      l_shoulder: shoulder_to_float.(action.shoulder),
      r_shoulder: 0.0,
      button_a: Enum.at(buttons, 0) == 1,
      button_b: Enum.at(buttons, 1) == 1,
      button_x: Enum.at(buttons, 2) == 1,
      button_y: Enum.at(buttons, 3) == 1,
      button_z: Enum.at(buttons, 4) == 1,
      button_l: Enum.at(buttons, 5) == 1,
      button_r: Enum.at(buttons, 6) == 1,
      button_d_up: false  # Start button mapped, D-up not used
    }
  end
end
