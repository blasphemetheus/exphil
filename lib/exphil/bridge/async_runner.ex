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

  @type on_game_end :: :restart | :stop

  defstruct [
    :agent,
    :bridge,
    :player_port,
    :state_table,      # ETS table for shared state
    :frame_loop_pid,
    :inference_pid,
    :running,
    :on_game_end,      # :restart or :stop
    :stats
  ]

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts)
  end

  def stop(runner) do
    GenServer.call(runner, :stop)
  end

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

    # Create ETS table for shared state
    table = :ets.new(:async_runner_state, [:set, :public])

    # Initialize shared state
    :ets.insert(table, {:latest_game_state, nil})
    :ets.insert(table, {:latest_action, nil})
    :ets.insert(table, {:in_game, false})
    :ets.insert(table, {:should_stop, false})        # Only true when stop() called or fatal error
    :ets.insert(table, {:on_game_end, on_game_end})  # :restart or :stop
    :ets.insert(table, {:frame_count, 0})
    :ets.insert(table, {:inference_count, 0})
    :ets.insert(table, {:games_played, 0})
    :ets.insert(table, {:start_time, nil})
    :ets.insert(table, {:latest_confidence, nil})
    :ets.insert(table, {:confidence_sum, 0.0})  # For running average

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
    frame_loop_pid = spawn_link(fn ->
      frame_loop(bridge, table, auto_menu, player_port, agent)
    end)

    # Start the inference process
    inference_pid = spawn_link(fn ->
      inference_loop(agent, table, player_port)
    end)

    state = %{state |
      frame_loop_pid: frame_loop_pid,
      inference_pid: inference_pid
    }

    Logger.info("[AsyncRunner] Started with frame_loop=#{inspect(frame_loop_pid)}, inference=#{inspect(inference_pid)}")

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

    {:reply, {:ok, %{frames: frames, inferences: inferences, games: games}}, %{state | running: false}}
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
      fps_ratio: fps / 60,  # 1.0 = hitting target, <1.0 = falling behind
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
            frame_loop(bridge, table, auto_menu, player_port, agent)

          {:postgame, game_state} ->
            handle_postgame(bridge, table, game_state, agent)
            frame_loop(bridge, table, auto_menu, player_port, agent)

          {:menu, _game_state} ->
            # In menu, just continue (auto_menu handles navigation)
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
        Logger.info("[AsyncRunner:FrameLoop] GAME #{games + 1} START at frame #{game_state.frame}")
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

    # Check for game end via stocks
    check_stocks_for_game_end(bridge, table, game_state, player_port, agent)
  end

  defp handle_postgame(bridge, table, game_state, agent) do
    # Only log if we were in a game
    case :ets.lookup(table, :in_game) do
      [{:in_game, true}] ->
        :ets.update_counter(table, :games_played, 1)
        [{:games_played, games}] = :ets.lookup(table, :games_played)
        Logger.info("[AsyncRunner:FrameLoop] POSTGAME (game #{games}) at frame #{game_state.frame}")

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
      Logger.info("[AsyncRunner:FrameLoop] Game over - #{result} (agent=#{agent_stocks}, opponent=#{opponent_stocks})")

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
                # Run inference with confidence
                case Agent.get_action_with_confidence(agent, game_state, player_port: player_port) do
                  {:ok, action, confidence} ->
                    :ets.insert(table, {:latest_action, action})
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

  defp action_to_input(action, _player_port) do
    # Convert action map to controller input format
    # Need to convert Nx tensors to plain Elixir values
    %{
      main_stick: %{
        x: to_float(Map.get(action, :main_x, 0.5)),
        y: to_float(Map.get(action, :main_y, 0.5))
      },
      c_stick: %{
        x: to_float(Map.get(action, :c_x, 0.5)),
        y: to_float(Map.get(action, :c_y, 0.5))
      },
      shoulder: to_float(Map.get(action, :shoulder, 0.0)),
      buttons: %{
        a: to_bool(Map.get(action, :button_a, false)),
        b: to_bool(Map.get(action, :button_b, false)),
        x: to_bool(Map.get(action, :button_x, false)),
        y: to_bool(Map.get(action, :button_y, false)),
        z: to_bool(Map.get(action, :button_z, false)),
        l: to_bool(Map.get(action, :button_l, false)),
        r: to_bool(Map.get(action, :button_r, false)),
        d_up: to_bool(Map.get(action, :button_d_up, false))
      }
    }
  end

  # Convert Nx tensor or value to float
  defp to_float(%Nx.Tensor{} = t), do: t |> Nx.squeeze() |> Nx.to_number() |> convert_to_float()
  defp to_float(v) when is_number(v), do: v / 1.0
  defp to_float(_), do: 0.5

  defp convert_to_float(v) when is_integer(v), do: v / 1.0
  defp convert_to_float(v), do: v

  # Convert Nx tensor or value to bool
  defp to_bool(%Nx.Tensor{} = t), do: t |> Nx.squeeze() |> Nx.to_number() > 0.5
  defp to_bool(v) when is_boolean(v), do: v
  defp to_bool(v) when is_number(v), do: v > 0.5
  defp to_bool(_), do: false

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
