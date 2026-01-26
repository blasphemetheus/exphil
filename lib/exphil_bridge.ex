defmodule ExPhil.Bridge do
  @moduledoc """
  High-level interface for the Melee game bridge.

  This module provides a convenient API for:
  - Starting/stopping the bridge
  - Running game loops
  - Sending controller inputs

  ## Example

      # Start a game
      {:ok, bridge} = ExPhil.Bridge.start(
        dolphin_path: "/path/to/slippi",
        iso_path: "/path/to/melee.iso",
        character: :mewtwo
      )

      # Game loop
      ExPhil.Bridge.run_game(bridge, fn game_state ->
        # Your AI logic here
        input = decide_action(game_state)
        {:continue, input}
      end)

      # Stop
      ExPhil.Bridge.stop(bridge)

  """

  alias ExPhil.Bridge.{MeleePort, GameState, ControllerInput}
  require Logger

  @type bridge :: pid()
  @type game_callback :: (GameState.t() -> {:continue, map()} | {:stop, any()})

  @doc """
  Start the Melee bridge.

  ## Options
    - `:dolphin_path` - Path to Slippi/Dolphin folder (required)
    - `:iso_path` - Path to Melee 1.02 ISO (required)
    - `:character` - Character to select (atom like :mewtwo, :ganondorf)
    - `:stage` - Stage to select (atom like :final_destination, :battlefield)
    - `:online_delay` - Simulate online delay in frames (default: 0)

  ## Returns
    `{:ok, bridge_pid}` on success, `{:error, reason}` on failure.
  """
  @spec start(keyword()) :: {:ok, bridge()} | {:error, term()}
  def start(opts) do
    with {:ok, pid} <- MeleePort.start_link(opts),
         :ok <- init_with_opts(pid, opts) do
      {:ok, pid}
    end
  end

  defp init_with_opts(pid, opts) do
    config = %{
      dolphin_path: Keyword.fetch!(opts, :dolphin_path),
      iso_path: Keyword.fetch!(opts, :iso_path),
      character: Keyword.get(opts, :character, :fox),
      stage: Keyword.get(opts, :stage, :final_destination),
      online_delay: Keyword.get(opts, :online_delay, 0),
      controller_port: Keyword.get(opts, :controller_port, 1),
      opponent_port: Keyword.get(opts, :opponent_port, 2)
    }

    case MeleePort.init_console(pid, config) do
      {:ok, _} -> :ok
      {:error, _} = err -> err
    end
  end

  @doc """
  Stop the bridge and close Dolphin.
  """
  @spec stop(bridge()) :: :ok
  def stop(bridge) do
    MeleePort.stop(bridge)
  end

  @doc """
  Get the next game state.

  Returns `{:ok, game_state}` when in game,
  `{:menu, game_state}` when in menus,
  or `{:error, reason}` on failure.
  """
  @spec step(bridge(), keyword()) ::
          {:ok, GameState.t()} | {:menu, GameState.t()} | {:error, term()}
  def step(bridge, opts \\ []) do
    MeleePort.step(bridge, opts)
  end

  @doc """
  Send controller input.

  See `ExPhil.Bridge.ControllerInput` for helpers to create inputs.
  """
  @spec send_controller(bridge(), map()) :: :ok | {:error, term()}
  def send_controller(bridge, input) do
    MeleePort.send_controller(bridge, input)
  end

  @doc """
  Run a game loop with a callback function.

  The callback receives the current game state and should return:
  - `{:continue, input}` - Send the input and continue
  - `{:stop, reason}` - Stop the loop

  ## Example

      ExPhil.Bridge.run_game(bridge, fn game_state ->
        {p1, p2} = GameState.get_players(game_state)

        input = if p2.x > p1.x do
          ControllerInput.right()
        else
          ControllerInput.left()
        end

        {:continue, input}
      end)

  """
  @spec run_game(bridge(), game_callback(), keyword()) :: {:ok, term()} | {:error, term()}
  def run_game(bridge, callback, opts \\ []) do
    max_frames = Keyword.get(opts, :max_frames, :infinity)
    run_game_loop(bridge, callback, 0, max_frames)
  end

  defp run_game_loop(_bridge, _callback, frame, max_frames) when frame >= max_frames do
    {:ok, :max_frames_reached}
  end

  defp run_game_loop(bridge, callback, frame, max_frames) do
    case step(bridge) do
      {:ok, game_state} ->
        case callback.(game_state) do
          {:continue, input} ->
            case send_controller(bridge, input) do
              :ok -> run_game_loop(bridge, callback, frame + 1, max_frames)
              {:error, _} = err -> err
            end

          {:stop, reason} ->
            {:ok, reason}
        end

      {:menu, _game_state} ->
        # Menu navigation is handled automatically by the bridge
        run_game_loop(bridge, callback, frame, max_frames)

      {:error, _} = err ->
        err
    end
  end

  @doc """
  Run a simple AI that follows a decision function.

  The decision function receives (own_player, opponent_player, game_state)
  and returns a controller input map.

  ## Example

      ExPhil.Bridge.run_ai(bridge, fn own, opp, _gs ->
        # Simple spacing AI
        dist = opp.x - own.x

        cond do
          dist > 20 -> ControllerInput.right()
          dist < -20 -> ControllerInput.left()
          abs(dist) < 10 -> ControllerInput.a()  # Attack!
          true -> ControllerInput.neutral()
        end
      end)

  """
  @spec run_ai(bridge(), (map(), map(), GameState.t() -> map()), keyword()) ::
          {:ok, term()} | {:error, term()}
  def run_ai(bridge, decision_fn, opts \\ []) do
    own_port = Keyword.get(opts, :own_port, 1)
    opp_port = Keyword.get(opts, :opp_port, 2)

    run_game(
      bridge,
      fn game_state ->
        own = GameState.get_player(game_state, own_port)
        opp = GameState.get_player(game_state, opp_port)

        if own && opp do
          input = decision_fn.(own, opp, game_state)
          {:continue, input}
        else
          {:continue, ControllerInput.neutral()}
        end
      end,
      opts
    )
  end

  @doc """
  Check if the bridge is alive and responding.
  """
  @spec ping(bridge()) :: :pong | {:error, term()}
  def ping(bridge) do
    MeleePort.ping(bridge)
  end
end
