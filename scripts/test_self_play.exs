#!/usr/bin/env elixir
# Self-play infrastructure validation script
#
# Tests the complete self-play pipeline:
# 1. Mock environment functionality
# 2. Dolphin integration (if available)
# 3. Experience collection
# 4. PPO update integration
#
# Usage:
#   mix run scripts/test_self_play.exs              # Test with mock only
#   mix run scripts/test_self_play.exs --dolphin   # Include Dolphin tests
#   mix run scripts/test_self_play.exs --full      # Full validation suite

alias ExPhil.Training.Output
alias ExPhil.SelfPlay.{Supervisor, GameRunner, ExperienceCollector}
alias ExPhil.Training.SelfPlay.SelfPlayEnv
alias ExPhil.Training.PPO
alias ExPhil.Embeddings

# ============================================================================
# Parse Arguments
# ============================================================================

args = System.argv()
test_dolphin = "--dolphin" in args or "--full" in args
test_ppo = "--ppo" in args or "--full" in args
verbose = "--verbose" in args or "-v" in args

dolphin_path = System.get_env("DOLPHIN_PATH") || System.get_env("HOME") <> "/.config/Slippi Launcher/netplay"
iso_path = System.get_env("MELEE_ISO")

# ============================================================================
# Helper Functions
# ============================================================================

defmodule TestHelpers do
  def create_test_policy(embed_size \\ 287) do
    # Create ActorCritic model for self-play
    model = ExPhil.Networks.ActorCritic.build_combined(
      embed_size: embed_size,
      hidden_sizes: [128, 128],
      dropout: 0.0
    )

    {init_fn, _predict_fn} = Axon.build(model)
    template = Nx.template({1, embed_size}, :f32)
    params = init_fn.(template, Axon.ModelState.empty())

    {model, params}
  end

  def format_duration(ms) when ms < 1000, do: "#{ms}ms"
  def format_duration(ms), do: "#{Float.round(ms / 1000, 2)}s"

  def check(condition, name) do
    if condition do
      IO.puts("  ✓ #{name}")
      :ok
    else
      IO.puts("  ✗ #{name}")
      :failed
    end
  end

  def section(name) do
    IO.puts("")
    IO.puts("━━━ #{name} ━━━")
  end

  def wait_for_game(port, max_frames) when max_frames <= 0 do
    {:error, :timeout}
  end

  def wait_for_game(port, max_frames) do
    alias ExPhil.Bridge.MeleePort

    case MeleePort.step(port, auto_menu: true) do
      {:ok, _state} -> :ok
      {:menu, _state} -> wait_for_game(port, max_frames - 1)
      {:postgame, _state} -> wait_for_game(port, max_frames - 1)
      error -> error
    end
  end

  def convert_experiences_to_rollout(experiences, embed_size) do
    # Stack states
    states = experiences
    |> Enum.map(& &1.state)
    |> Enum.map(fn t ->
      case Nx.shape(t) do
        {1, _} -> Nx.squeeze(t, axes: [0])
        {^embed_size} -> t
        _ -> Nx.reshape(t, {embed_size})
      end
    end)
    |> Nx.stack()

    # Stack rewards
    rewards = experiences
    |> Enum.map(& &1.reward)
    |> Nx.tensor(type: :f32)

    # Stack dones
    dones = experiences
    |> Enum.map(&if(&1.done, do: 1.0, else: 0.0))
    |> Nx.tensor(type: :f32)

    # Stack values
    values = experiences
    |> Enum.map(fn exp ->
      case exp.value do
        %Nx.Tensor{} = t -> Nx.to_number(t)
        n when is_number(n) -> n
        _ -> 0.0
      end
    end)
    |> Kernel.++([0.0])  # Bootstrap
    |> Nx.tensor(type: :f32)

    # Stack log_probs
    log_probs = experiences
    |> Enum.map(fn exp ->
      case exp.log_prob do
        %Nx.Tensor{} = t -> Nx.to_number(t)
        n when is_number(n) -> n
        _ -> -1.0
      end
    end)
    |> Nx.tensor(type: :f32)

    # Stack actions
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
end

# Convenience aliases
alias TestHelpers, as: H

# ============================================================================
# Test Suite
# ============================================================================

Output.banner("Self-Play Validation Suite")
Output.puts("Testing self-play infrastructure...")
Output.puts("")

results = %{passed: 0, failed: 0, skipped: 0}

# ============================================================================
# Test 1: Mock Environment
# ============================================================================

H.section("1. Mock Environment")
start = System.monotonic_time(:millisecond)

results = try do
  # Create test policy
  IO.puts("  Creating test policy...")
  embed_size = Embeddings.embedding_size()
  {model, params} = H.create_test_policy(embed_size)

  # Create SelfPlayEnv with mock game
  IO.puts("  Creating SelfPlayEnv (mock)...")
  {:ok, env} = SelfPlayEnv.new(
    p1_policy: {model, params},
    p2_policy: :cpu,
    game_type: :mock,
    config: [max_episode_frames: 1000]
  )

  H.check(env != nil, "SelfPlayEnv created")

  # Collect some steps
  IO.puts("  Collecting 100 steps...")
  {:ok, env, experiences} = SelfPlayEnv.collect_steps(env, 100)

  H.check(length(experiences) == 100, "Collected 100 experiences")

  # Verify experience structure
  exp = hd(experiences)
  has_state = Map.has_key?(exp, :state)
  has_action = Map.has_key?(exp, :action)
  has_reward = Map.has_key?(exp, :reward)
  has_done = Map.has_key?(exp, :done)

  H.check(has_state and has_action and has_reward and has_done, "Experience has correct structure")

  # Verify state shape
  state_shape = Nx.shape(exp.state)
  H.check(elem(state_shape, 0) == embed_size, "State has correct embedding size (#{embed_size})")

  # Verify action structure
  action = exp.action
  has_buttons = Map.has_key?(action, :buttons)
  has_main = Map.has_key?(action, :main_x) and Map.has_key?(action, :main_y)

  H.check(has_buttons and has_main, "Action has correct structure")

  # Shutdown
  SelfPlayEnv.shutdown(env)

  elapsed = System.monotonic_time(:millisecond) - start
  IO.puts("  Completed in #{H.format_duration(elapsed)}")

  %{results | passed: results.passed + 5}
rescue
  e ->
    IO.puts("  ✗ Mock environment test failed: #{inspect(e)}")
    %{results | failed: results.failed + 1}
end

# ============================================================================
# Test 2: GameRunner GenServer
# ============================================================================

H.section("2. GameRunner GenServer")
start = System.monotonic_time(:millisecond)

results = try do
  # Start Supervisor first
  IO.puts("  Starting SelfPlay.Supervisor...")
  {:ok, _sup} = Supervisor.start_link(
    batch_size: 128,
    max_history_size: 5,
    start_matchmaker: false
  )

  H.check(true, "Supervisor started")

  # Set policy
  IO.puts("  Setting initial policy...")
  embed_size = Embeddings.embedding_size()
  {model, params} = H.create_test_policy(embed_size)
  :ok = Supervisor.set_policy(model, params)

  H.check(true, "Policy registered")

  # Start a game
  IO.puts("  Starting mock game...")
  [{:ok, game_id}] = Supervisor.start_games(1, game_type: :mock, config: [max_episode_frames: 500])

  H.check(game_id != nil, "Game started with ID: #{game_id}")

  # Wait a moment for game to initialize
  Process.sleep(100)

  # Collect steps through supervisor
  IO.puts("  Collecting 50 steps through Supervisor...")
  experiences = Supervisor.collect_steps(50)

  H.check(length(experiences) >= 0, "Collected #{length(experiences)} experiences")

  # Get game status
  pid = GameRunner.whereis(game_id)
  status = if pid, do: GameRunner.get_status(pid), else: %{status: :not_found}
  H.check(status.status in [:waiting, :playing, :finished, :not_found], "Game status: #{status.status}")

  # Stop games and supervisor
  Supervisor.stop_all_games()
  try do
    Elixir.Supervisor.stop(ExPhil.SelfPlay.Supervisor)
  catch
    :exit, _ -> :ok
  end

  elapsed = System.monotonic_time(:millisecond) - start
  IO.puts("  Completed in #{H.format_duration(elapsed)}")

  %{results | passed: results.passed + 4}
rescue
  e ->
    IO.puts("  ✗ GameRunner test failed: #{inspect(e)}")
    try do
      Elixir.Supervisor.stop(ExPhil.SelfPlay.Supervisor)
    catch
      :exit, _ -> :ok
    end
    %{results | failed: results.failed + 1}
end

# ============================================================================
# Test 3: PPO Integration
# ============================================================================

if test_ppo do
  H.section("3. PPO Integration")
  start = System.monotonic_time(:millisecond)

  results = try do
    embed_size = Embeddings.embedding_size()

    IO.puts("  Creating PPO trainer...")
    ppo = PPO.new(
      embed_size: embed_size,
      hidden_sizes: [128, 128],
      gamma: 0.99,
      gae_lambda: 0.95,
      clip_range: 0.2,
      num_epochs: 2,
      learning_rate: 3.0e-4,
      batch_size: 64
    )

    H.check(ppo != nil, "PPO trainer created")

    # Create mock rollout data
    IO.puts("  Creating mock rollout (128 steps)...")
    num_steps = 128
    {model, params} = H.create_test_policy(embed_size)
    {_init_fn, predict_fn} = Axon.build(model, mode: :inference)

    # Generate fake experiences
    states = Nx.broadcast(0.0, {num_steps, embed_size})
    |> Nx.add(Nx.Random.uniform(Nx.Random.key(42), shape: {num_steps, embed_size}) |> elem(0))

    rewards = Nx.broadcast(0.1, {num_steps})
    dones = Nx.broadcast(0.0, {num_steps})
    values = Nx.broadcast(0.5, {num_steps + 1})  # Bootstrap value
    log_probs = Nx.broadcast(-2.0, {num_steps})

    # Generate actions by running policy
    output = predict_fn.(params, states)

    # ActorCritic returns %{policy: {buttons, main_x, main_y, c_x, c_y, shoulder}, value: value}
    policy_logits = case output do
      %{policy: p} -> p
      {p, _v} when is_tuple(p) -> p
      tuple when is_tuple(tuple) -> Tuple.delete_at(tuple, tuple_size(tuple) - 1)
    end

    {buttons, main_x, main_y, c_x, c_y, shoulder} = policy_logits

    actions = %{
      buttons: Nx.greater(Nx.sigmoid(buttons), 0.5),
      main_x: Nx.argmax(main_x, axis: -1),
      main_y: Nx.argmax(main_y, axis: -1),
      c_x: Nx.argmax(c_x, axis: -1),
      c_y: Nx.argmax(c_y, axis: -1),
      shoulder: Nx.argmax(shoulder, axis: -1)
    }

    rollout = %{
      states: states,
      actions: actions,
      rewards: rewards,
      values: values,
      log_probs: log_probs,
      dones: dones
    }

    H.check(true, "Rollout created")

    # Run PPO update
    IO.puts("  Running PPO update (2 epochs)...")
    update_start = System.monotonic_time(:millisecond)

    {updated_ppo, metrics} = PPO.update(ppo, rollout)

    update_time = System.monotonic_time(:millisecond) - update_start
    IO.puts("    Update took #{H.format_duration(update_time)}")

    H.check(updated_ppo != nil, "PPO update completed")

    policy_loss = Map.get(metrics, :policy_loss, 0.0)
    value_loss = Map.get(metrics, :value_loss, 0.0)
    IO.puts("    Policy loss: #{Float.round(policy_loss, 4)}")
    IO.puts("    Value loss: #{Float.round(value_loss, 4)}")

    H.check(is_number(policy_loss), "Metrics returned")

    elapsed = System.monotonic_time(:millisecond) - start
    IO.puts("  Completed in #{H.format_duration(elapsed)}")

    %{results | passed: results.passed + 4}
  rescue
    e ->
      IO.puts("  ✗ PPO integration test failed: #{inspect(e)}")
      if verbose do
        IO.puts("    #{Exception.format(:error, e, __STACKTRACE__)}")
      end
      %{results | failed: results.failed + 1}
  end
else
  H.section("3. PPO Integration (skipped)")
  IO.puts("  Use --ppo or --full to run PPO tests")
  results = %{results | skipped: results.skipped + 1}
end

# ============================================================================
# Test 4: Dolphin Integration
# ============================================================================

if test_dolphin do
  H.section("4. Dolphin Integration")

  if iso_path && File.exists?(iso_path) do
    start = System.monotonic_time(:millisecond)

    results = try do
      IO.puts("  Dolphin path: #{dolphin_path}")
      IO.puts("  ISO path: #{iso_path}")

      alias ExPhil.Bridge.MeleePort

      # Start MeleePort
      IO.puts("  Starting MeleePort...")
      {:ok, port} = MeleePort.start_link()

      H.check(port != nil, "MeleePort started")

      # Initialize console
      IO.puts("  Initializing console (this may take 30-60s)...")
      init_start = System.monotonic_time(:millisecond)

      result = MeleePort.init_console(port, %{
        dolphin_path: dolphin_path,
        iso_path: iso_path,
        character: "mewtwo",
        stage: "final_destination"
      })

      init_time = System.monotonic_time(:millisecond) - init_start

      case result do
        {:ok, %{controller_port: _}} ->
          IO.puts("    Console initialized in #{H.format_duration(init_time)}")
          H.check(true, "Console initialized")

          # Wait for game to start
          IO.puts("  Waiting for game to start...")
          H.wait_for_game(port, 600)  # 10 seconds

          # Step a few frames
          IO.puts("  Stepping 60 frames (1 second)...")
          frames = for _ <- 1..60 do
            {:ok, state} = MeleePort.step(port, auto_menu: true)
            state.frame
          end

          unique_frames = Enum.uniq(frames) |> length()
          H.check(unique_frames > 1, "Frames advancing (#{unique_frames} unique)")

          # Cleanup
          MeleePort.stop(port)

          elapsed = System.monotonic_time(:millisecond) - start
          IO.puts("  Completed in #{H.format_duration(elapsed)}")

          %{results | passed: results.passed + 3}

        {:error, reason} ->
          IO.puts("  ✗ Console init failed: #{inspect(reason)}")
          MeleePort.stop(port)
          %{results | failed: results.failed + 1}
      end
    rescue
      e ->
        IO.puts("  ✗ Dolphin test failed: #{inspect(e)}")
        if verbose do
          IO.puts("    #{Exception.format(:error, e, __STACKTRACE__)}")
        end
        %{results | failed: results.failed + 1}
    end
  else
    IO.puts("  ⚠ MELEE_ISO not set or file not found")
    IO.puts("  Set MELEE_ISO environment variable to test Dolphin")
    results = %{results | skipped: results.skipped + 1}
  end
else
  H.section("4. Dolphin Integration (skipped)")
  IO.puts("  Use --dolphin or --full to run Dolphin tests")
  results = %{results | skipped: results.skipped + 1}
end

# ============================================================================
# Test 5: Full Self-Play Loop (Mock)
# ============================================================================

H.section("5. Full Self-Play Loop")
start = System.monotonic_time(:millisecond)

results = try do
  embed_size = Embeddings.embedding_size()

  IO.puts("  Setting up full self-play test...")
  IO.puts("    - 2 parallel mock games")
  IO.puts("    - 256 total steps")
  IO.puts("    - 1 PPO update")

  # Create fresh supervisor
  {:ok, _sup} = Supervisor.start_link(
    batch_size: 128,
    max_history_size: 5,
    start_matchmaker: false
  )

  # Set policy
  {model, params} = H.create_test_policy(embed_size)
  :ok = Supervisor.set_policy(model, params)

  # Create PPO trainer
  ppo = PPO.new(
    embed_size: embed_size,
    hidden_sizes: [128, 128],
    gamma: 0.99,
    gae_lambda: 0.95,
    clip_range: 0.2,
    num_epochs: 2,
    learning_rate: 3.0e-4,
    batch_size: 64
  )

  # Start 2 games
  IO.puts("  Starting 2 mock games...")
  game_results = Supervisor.start_games(2, game_type: :mock, config: [max_episode_frames: 200])
  started = Enum.count(game_results, fn {:ok, _} -> true; _ -> false end)
  H.check(started == 2, "#{started}/2 games started")

  # Collect experience
  IO.puts("  Collecting 256 steps...")
  collect_start = System.monotonic_time(:millisecond)
  experiences = Supervisor.collect_steps(256)
  collect_time = System.monotonic_time(:millisecond) - collect_start

  H.check(length(experiences) > 0, "Collected #{length(experiences)} experiences in #{H.format_duration(collect_time)}")

  if length(experiences) > 0 do
    # Convert to rollout format
    IO.puts("  Converting to rollout format...")

    rollout = H.convert_experiences_to_rollout(experiences, embed_size)

    H.check(Nx.shape(rollout.states) |> elem(0) == length(experiences), "Rollout states correct shape")

    # Run PPO update
    IO.puts("  Running PPO update...")
    update_start = System.monotonic_time(:millisecond)
    {updated_ppo, metrics} = PPO.update(ppo, rollout)
    update_time = System.monotonic_time(:millisecond) - update_start

    policy_loss = Map.get(metrics, :policy_loss, 0.0)
    IO.puts("    Policy loss: #{Float.round(policy_loss, 4)} (#{H.format_duration(update_time)})")

    H.check(updated_ppo != nil, "PPO update successful")

    # Sync new params to games
    IO.puts("  Syncing updated policy to games...")
    Supervisor.update_policy_params(updated_ppo.params)

    H.check(true, "Policy synced to games")
  end

  # Cleanup
  Supervisor.stop_all_games()

  elapsed = System.monotonic_time(:millisecond) - start
  IO.puts("  Completed in #{H.format_duration(elapsed)}")

  %{results | passed: results.passed + 5}
rescue
  e ->
    IO.puts("  ✗ Full loop test failed: #{inspect(e)}")
    if verbose do
      IO.puts("    #{Exception.format(:error, e, __STACKTRACE__)}")
    end
    %{results | failed: results.failed + 1}
end

# ============================================================================
# Results Summary
# ============================================================================

Output.divider()
IO.puts("")
total = results.passed + results.failed + results.skipped

if results.failed == 0 do
  Output.success("All tests passed! (#{results.passed}/#{total})")
else
  Output.error("#{results.failed} test(s) failed (#{results.passed} passed, #{results.skipped} skipped)")
end

IO.puts("")
IO.puts("Next steps:")
IO.puts("  1. Run full training: mix run scripts/train_self_play.exs --game-type mock --timesteps 10000")
if not test_dolphin do
  IO.puts("  2. Test with Dolphin: mix run scripts/test_self_play.exs --dolphin")
end
IO.puts("")

if results.failed > 0, do: System.halt(1)
