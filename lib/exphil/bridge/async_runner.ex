defmodule ExPhil.Bridge.AsyncRunner do
  @moduledoc """
  Async game runner that decouples frame reading from inference.

  Architecture:
  ```
  ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
  │   FrameLoop     │────▶│   SharedState    │◀────│   Inference     │
  │   (fast, 60fps) │     │   (ETS table)    │     │   (slow, async) │
  └─────────────────┘     └──────────────────┘     └─────────────────┘
         │                        │
         │                        ▼
         │                 ┌─────────────────┐
         └────────────────▶│ ControllerSend  │
                           │ (uses last act) │
                           └─────────────────┘
  ```

  The FrameLoop reads frames as fast as possible (targeting 60fps).
  The Inference process runs async and updates the action when ready.
  FrameLoop always sends the most recent action, even if inference is slow.

  **Post-game behavior (configurable via `on_game_end` option):**
  - `:restart` (default) - Automatically start a new game after each match
  - `:stop` - Exit after a single game completes

  ## Usage

      # Start the async runner
      {:ok, runner} = AsyncRunner.start_link(
        agent: agent_pid,
        bridge: bridge_pid,
        player_port: 1
      )

      # Runner handles everything automatically, restarting games
      # Call stop when done
      AsyncRunner.stop(runner)
  """

  use GenServer
  require Logger

  alias ExPhil.Agents.Agent
  alias ExPhil.Bridge.MeleePort

  # ============================================================================
  # Types
  # ============================================================================

  @typedoc "Behavior when a game ends"
  @type on_game_end :: :restart | :stop

  @typedoc "Start link options"
  @type start_option ::
          {:agent, pid()}
          | {:bridge, pid()}
          | {:player_port, pos_integer()}
          | {:auto_menu, boolean()}
          | {:on_game_end, on_game_end()}

  @typedoc "Statistics returned by get_stats/1"
  @type stats :: %{
          frames_read: non_neg_integer(),
          inferences_run: non_neg_integer(),
          games_played: non_neg_integer(),
          elapsed_ms: non_neg_integer(),
          fps: float(),
          target_fps: pos_integer(),
          fps_ratio: float(),
          inference_rate: float(),
          latest_confidence: map() | nil,
          avg_confidence: float()
        }

  @typedoc "Stop result with final counts"
  @type stop_result :: %{
          frames: non_neg_integer(),
          inferences: non_neg_integer(),
          games: non_neg_integer()
        }

  defstruct [
    :agent,
    :bridge,
    :player_port,
    # ETS table for shared state
    :state_table,
    :frame_loop_pid,
    :inference_pid,
    :running,
    # :restart or :stop
    :on_game_end,
    :stats
  ]

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the async runner process.

  ## Options

    * `:agent` - PID of the ExPhil.Agents.Agent process (required)
    * `:bridge` - PID of the MeleePort process (required)
    * `:player_port` - Controller port for the agent (default: 1)
    * `:auto_menu` - Whether to auto-navigate menus (default: true)
    * `:on_game_end` - `:restart` to play another game, `:stop` to exit (default: `:restart`)

  ## Examples

      {:ok, runner} = AsyncRunner.start_link(
        agent: agent_pid,
        bridge: bridge_pid,
        player_port: 1
      )

  """
  @spec start_link([start_option()]) :: GenServer.on_start()
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts)
  end

  @doc """
  Stops the async runner and returns final statistics.

  Returns `{:ok, stats}` where stats contains the total frames read,
  inferences run, and games played.
  """
  @spec stop(GenServer.server()) :: {:ok, stop_result()}
  def stop(runner) do
    GenServer.call(runner, :stop)
  end

  @doc """
  Gets current runtime statistics.

  Returns a map with performance metrics including FPS, inference rate,
  and confidence statistics.
  """
  @spec get_stats(GenServer.server()) :: stats()
  def get_stats(runner) do
    GenServer.call(runner, :get_stats)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    agent = Keyword.fetch!(opts, :agent)
    bridge = Keyword.fetch!(opts, :bridge)
    player_port = Keyword.get(opts, :player_port, 1)
    auto_menu = Keyword.get(opts, :auto_menu, true)
    on_game_end = Keyword.get(opts, :on_game_end, :restart)
    # Elixir-driven opponent-port dummy (reactive drills): a module with
    # new/0 + step/2 returning {input, state}. Requires the bridge started
    # with dummy_mode: "external" so the port-2 controller exists.
    dummy_module = Keyword.get(opts, :dummy)

    # Create ETS table for shared state
    table = :ets.new(:async_runner_state, [:set, :public])

    if dummy_module do
      :ets.insert(table, {:elixir_dummy, dummy_module, dummy_module.new()})
    end

    # Initialize shared state
    :ets.insert(table, {:latest_game_state, nil})
    :ets.insert(table, {:latest_action, nil})
    :ets.insert(table, {:in_game, false})
    # Only true when stop() called or fatal error
    :ets.insert(table, {:should_stop, false})

    # Frame pacing (pace_hz opt): with blocking pipe input the game only
    # advances when we feed it, so the RUNNER is the throttle. The ExiAI
    # headless build has no internal throttle (audio-backend-tied on this
    # base, GOTCHA #56; emulation_speed config is a no-op) and unthrottled
    # policy-driven games quantize inputs to the act-cycle (~3.6 game
    # frames at 450fps) — SH presses mistimed, knockdowns collapse
    # (2026-07-17). pace_hz: 60 restores real-time input granularity for
    # probes; 0/absent = unpaced (scenario-style frame-locked replay
    # drives MeleePort directly and doesn't come through here).
    pace_hz = Keyword.get(opts, :pace_hz, 0)

    if is_integer(pace_hz) and pace_hz > 0 do
      :ets.insert(table, {:pace_ns, div(1_000_000_000, pace_hz)})
    end
    # :restart or :stop
    :ets.insert(table, {:on_game_end, on_game_end})
    :ets.insert(table, {:frame_count, 0})
    :ets.insert(table, {:inference_count, 0})
    :ets.insert(table, {:games_played, 0})
    :ets.insert(table, {:start_time, nil})
    :ets.insert(table, {:latest_confidence, nil})
    # For running average
    :ets.insert(table, {:confidence_sum, 0.0})

    state = %__MODULE__{
      agent: agent,
      bridge: bridge,
      player_port: player_port,
      state_table: table,
      running: true,
      on_game_end: on_game_end,
      stats: %{
        frames_read: 0,
        inferences_run: 0,
        actions_sent: 0,
        start_time: nil
      }
    }

    # Start the frame loop process
    frame_loop_pid =
      spawn_link(fn ->
        frame_loop(bridge, table, auto_menu, player_port, agent)
      end)

    # Start the inference process
    inference_pid =
      spawn_link(fn ->
        inference_loop(agent, table, player_port)
      end)

    state = %{state | frame_loop_pid: frame_loop_pid, inference_pid: inference_pid}

    Logger.info(
      "[AsyncRunner] Started with frame_loop=#{inspect(frame_loop_pid)}, inference=#{inspect(inference_pid)}"
    )

    {:ok, state}
  end

  @impl true
  def handle_call(:stop, _from, state) do
    Logger.info("[AsyncRunner] Stopping...")

    # Signal processes to stop
    :ets.insert(state.state_table, {:should_stop, true})

    # Give them a moment to clean up
    Process.sleep(100)

    # Kill if still alive
    if Process.alive?(state.frame_loop_pid), do: Process.exit(state.frame_loop_pid, :shutdown)
    if Process.alive?(state.inference_pid), do: Process.exit(state.inference_pid, :shutdown)

    # Get final stats
    [{:frame_count, frames}] = :ets.lookup(state.state_table, :frame_count)
    [{:inference_count, inferences}] = :ets.lookup(state.state_table, :inference_count)
    [{:games_played, games}] = :ets.lookup(state.state_table, :games_played)

    :ets.delete(state.state_table)

    {:reply, {:ok, %{frames: frames, inferences: inferences, games: games}},
     %{state | running: false}}
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    [{:frame_count, frames}] = :ets.lookup(state.state_table, :frame_count)
    [{:inference_count, inferences}] = :ets.lookup(state.state_table, :inference_count)
    [{:games_played, games}] = :ets.lookup(state.state_table, :games_played)
    [{:start_time, start_time}] = :ets.lookup(state.state_table, :start_time)
    [{:latest_confidence, latest_conf}] = :ets.lookup(state.state_table, :latest_confidence)
    [{:confidence_sum, conf_sum}] = :ets.lookup(state.state_table, :confidence_sum)

    elapsed = if start_time, do: System.monotonic_time(:millisecond) - start_time, else: 0
    fps = if(elapsed > 0, do: frames * 1000 / elapsed, else: 0)
    avg_confidence = if(inferences > 0, do: conf_sum / inferences, else: 0)

    stats = %{
      frames_read: frames,
      inferences_run: inferences,
      games_played: games,
      elapsed_ms: elapsed,
      fps: fps,
      target_fps: 60,
      # 1.0 = hitting target, <1.0 = falling behind
      fps_ratio: fps / 60,
      inference_rate: if(elapsed > 0, do: inferences * 1000 / elapsed, else: 0),
      # Confidence stats
      latest_confidence: latest_conf,
      avg_confidence: avg_confidence
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, pid, reason}, state) do
    cond do
      pid == state.frame_loop_pid ->
        Logger.warning("[AsyncRunner] Frame loop died: #{inspect(reason)}")

      pid == state.inference_pid ->
        Logger.warning("[AsyncRunner] Inference loop died: #{inspect(reason)}")

      true ->
        :ok
    end

    {:noreply, state}
  end

  # ============================================================================
  # Frame Loop (runs in separate process, fast)
  # ============================================================================

  # Hold the frame cadence set by :pace_ns (see init). Uses the frame
  # process's own dictionary for last-frame time — the loop is a single
  # process, no coordination needed.
  defp pace(table) do
    case :ets.lookup(table, :pace_ns) do
      [{:pace_ns, ns}] when ns > 0 ->
        now = System.monotonic_time(:nanosecond)
        last = Process.get(:last_frame_ns, now - ns)
        remaining_ms = div(ns - (now - last), 1_000_000)
        if remaining_ms > 0, do: Process.sleep(remaining_ms)
        Process.put(:last_frame_ns, System.monotonic_time(:nanosecond))

      _ ->
        :ok
    end
  end

  defp frame_loop(bridge, table, auto_menu, player_port, agent) do
    case should_stop?(table) do
      true ->
        # User requested stop
        send_neutral_controller(bridge)
        Logger.info("[AsyncRunner:FrameLoop] Stop requested, exiting")
        :ok

      false ->
        # Read next frame
        case MeleePort.step(bridge, auto_menu: auto_menu) do
          {:ok, game_state} ->
            handle_game_frame(bridge, table, game_state, player_port, agent)
            pace(table)
            frame_loop(bridge, table, auto_menu, player_port, agent)

          {:postgame, game_state} ->
            handle_postgame(bridge, table, game_state, agent)
            frame_loop(bridge, table, auto_menu, player_port, agent)

          {:menu, game_state} ->
            # Falling out of a game into ANY menu is the game end: this
            # Slippi setup never reports POSTGAME_SCORES (verified from
            # bridge logs — a full game→restart cycle showed IN_GAME →
            # CHARACTER_SELECT directly), so waiting for a :postgame event
            # meant on_game_end was never honored and :stop sessions ran
            # (and held the GPU) forever.
            if in_game?(table), do: handle_postgame(bridge, table, game_state, agent)

            # Otherwise just continue (auto_menu handles navigation)
            frame_loop(bridge, table, auto_menu, player_port, agent)

          {:game_ended, reason} ->
            # Fatal disconnect - always stop
            Logger.info("[AsyncRunner:FrameLoop] Fatal: #{reason}")
            send_neutral_controller(bridge)
            :ets.insert(table, {:should_stop, true})
            :ok

          {:error, reason} ->
            # Fatal error - always stop
            Logger.error("[AsyncRunner:FrameLoop] Error: #{inspect(reason)}")
            send_neutral_controller(bridge)
            :ets.insert(table, {:should_stop, true})
            :ok
        end
    end
  end

  defp handle_game_frame(bridge, table, game_state, player_port, agent) do
    # Update latest game state
    :ets.insert(table, {:latest_game_state, game_state})

    # Check if this is game start
    case :ets.lookup(table, :in_game) do
      [{:in_game, false}] ->
        [{:games_played, games}] = :ets.lookup(table, :games_played)

        Logger.info(
          "[AsyncRunner:FrameLoop] GAME #{games + 1} START at frame #{game_state.frame}"
        )

        :ets.insert(table, {:in_game, true})
        :ets.insert(table, {:start_time, System.monotonic_time(:millisecond)})

      _ ->
        :ok
    end

    # Increment frame count
    :ets.update_counter(table, :frame_count, 1)

    # Get latest action and send it
    case :ets.lookup(table, :latest_action) do
      [{:latest_action, nil}] ->
        # No action yet, send neutral
        :ok

      [{:latest_action, action}] ->
        # Send the action
        input = action_to_input(action, player_port)
        MeleePort.send_controller(bridge, input)
    end

    drive_elixir_dummy(bridge, table, game_state, player_port)

    # Check for game end via stocks
    check_stocks_for_game_end(bridge, table, game_state, player_port, agent)
  end

  defp handle_postgame(bridge, table, game_state, agent) do
    # Only log if we were in a game
    case :ets.lookup(table, :in_game) do
      [{:in_game, true}] ->
        :ets.update_counter(table, :games_played, 1)
        [{:games_played, games}] = :ets.lookup(table, :games_played)

        Logger.info(
          "[AsyncRunner:FrameLoop] POSTGAME (game #{games}) at frame #{game_state.frame}"
        )

        # Check on_game_end setting
        case :ets.lookup(table, :on_game_end) do
          [{:on_game_end, :stop}] ->
            Logger.info("[AsyncRunner:FrameLoop] on_game_end=:stop, finishing")
            :ets.insert(table, {:should_stop, true})

          [{:on_game_end, :restart}] ->
            Logger.info("[AsyncRunner:FrameLoop] on_game_end=:restart, continuing to next game")
        end

        # Reset for next game
        reset_between_games(table, agent)

      _ ->
        :ok
    end

    send_neutral_controller(bridge)
  end

  defp check_stocks_for_game_end(bridge, _table, game_state, player_port, _agent) do
    players = game_state.players || %{}
    agent_player = players[player_port]
    opponent_port = if player_port == 1, do: 2, else: 1
    opponent = players[opponent_port]

    agent_stocks = agent_player && agent_player.stock
    opponent_stocks = opponent && opponent.stock

    if agent_stocks == 0 or opponent_stocks == 0 do
      result = if agent_stocks == 0, do: "LOSS", else: "WIN"

      Logger.info(
        "[AsyncRunner:FrameLoop] Game over - #{result} (agent=#{agent_stocks}, opponent=#{opponent_stocks})"
      )

      # Send neutral immediately when game ends
      send_neutral_controller(bridge)

      # The postgame handler will deal with restart vs stop
      # We just mark that we detected the game end via stocks
      # (postgame event from libmelee will follow shortly)
    end
  end

  defp reset_between_games(table, agent) do
    # Clear action (fresh start for next game)
    :ets.insert(table, {:latest_action, nil})
    :ets.insert(table, {:in_game, false})
    :ets.insert(table, {:start_time, nil})

    # Reset agent's temporal buffer for fresh game context
    Agent.reset_buffer(agent)
  end

  defp should_stop?(table) do
    case :ets.lookup(table, :should_stop) do
      [{:should_stop, true}] -> true
      _ -> false
    end
  end

  # Elixir-driven opponent dummy: read the opponent's state, step the dummy
  # module, route its input to the opponent port. Synchronous with the frame
  # loop, same as the agent's own send.
  defp drive_elixir_dummy(bridge, table, game_state, player_port) do
    case :ets.lookup(table, :elixir_dummy) do
      [{:elixir_dummy, mod, dummy_state}] ->
        opponent_port = if player_port == 1, do: 2, else: 1
        opponent = game_state.players && game_state.players[opponent_port]
        bot = game_state.players && game_state.players[player_port]

        {input, dummy_state} = mod.step(opponent, bot, dummy_state)
        MeleePort.send_controller(bridge, Map.put(input, :port, opponent_port))
        :ets.insert(table, {:elixir_dummy, mod, dummy_state})

      _ ->
        :ok
    end
  end

  defp in_game?(table) do
    case :ets.lookup(table, :in_game) do
      [{:in_game, true}] -> true
      _ -> false
    end
  end

  # ============================================================================
  # Inference Loop (runs in separate process, slow is OK)
  # ============================================================================

  defp inference_loop(agent, table, player_port) do
    case should_stop?(table) do
      true ->
        Logger.info("[AsyncRunner:Inference] Stop requested, exiting")
        :ok

      false ->
        # Get latest game state
        case :ets.lookup(table, :latest_game_state) do
          [{:latest_game_state, nil}] ->
            # No game state yet, wait
            Process.sleep(10)
            inference_loop(agent, table, player_port)

          [{:latest_game_state, game_state}] ->
            # Check if we're in game
            case :ets.lookup(table, :in_game) do
              [{:in_game, true}] ->
                # Run inference with confidence. Must use the controller API:
                # the raw action map holds bucket INDICES (0..axis_buckets),
                # not analog values — sending those to libmelee clamps the
                # sticks to full deflection (the "drifts right forever" bug).
                case Agent.get_controller_with_confidence(agent, game_state,
                       player_port: player_port
                     ) do
                  {:ok, controller, confidence} ->
                    :ets.insert(table, {:latest_action, controller})
                    :ets.insert(table, {:latest_confidence, confidence})
                    :ets.update_counter(table, :inference_count, 1)
                    # Update running sum for average
                    overall_conf = Map.get(confidence, :overall, 0)
                    [{:confidence_sum, old_sum}] = :ets.lookup(table, :confidence_sum)
                    :ets.insert(table, {:confidence_sum, old_sum + overall_conf})

                  {:error, reason} ->
                    Logger.warning("[AsyncRunner:Inference] Error: #{inspect(reason)}")
                end

              _ ->
                # Not in game, wait
                Process.sleep(10)
            end

            inference_loop(agent, table, player_port)
        end
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  # Convert a fully-decoded ControllerState (sticks/shoulder undiscretized to
  # 0..1, buttons boolean — see Networks.Policy.to_controller_state/2) into
  # the bridge input map. Intentionally accepts ONLY %ControllerState{}: raw
  # action maps carry bucket indices and must never reach the bridge.
  defp action_to_input(%ExPhil.Bridge.ControllerState{} = cs, _player_port) do
    %{
      main_stick: %{x: cs.main_stick.x, y: cs.main_stick.y},
      c_stick: %{x: cs.c_stick.x, y: cs.c_stick.y},
      shoulder: cs.l_shoulder + cs.r_shoulder,
      buttons: %{
        a: cs.button_a,
        b: cs.button_b,
        x: cs.button_x,
        y: cs.button_y,
        z: cs.button_z,
        l: cs.button_l,
        r: cs.button_r,
        d_up: cs.button_d_up
      }
    }
  end

  # Send neutral controller state (all centered, no buttons)
  defp send_neutral_controller(bridge) do
    neutral = %{
      main_stick: %{x: 0.5, y: 0.5},
      c_stick: %{x: 0.5, y: 0.5},
      shoulder: 0.0,
      buttons: %{
        a: false,
        b: false,
        x: false,
        y: false,
        z: false,
        l: false,
        r: false,
        d_up: false
      }
    }

    MeleePort.send_controller(bridge, neutral)
  end
end
