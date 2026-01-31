defmodule ExPhil.Data.Peppi do
  @moduledoc """
  Parse Slippi replay files using Peppi (Rust NIF).

  This module provides fast, native parsing of .slp files using the Peppi
  library via Rustler NIFs. It's the recommended way to parse replays for
  training.

  ## Usage

      # Parse a single replay
      {:ok, replay} = Peppi.parse("game.slp")
      replay.frames  # List of frame data
      replay.metadata  # Replay metadata

      # Parse with player filter
      {:ok, replay} = Peppi.parse("game.slp", player_port: 1)

      # Get just metadata (faster, for filtering)
      {:ok, meta} = Peppi.metadata("game.slp")

      # Convert to training format
      frames = Peppi.to_training_frames(replay)

  ## Struct Types

  The NIF returns data in these struct types:
  - `Peppi.ParsedReplay` - Complete parsed replay
  - `Peppi.GameFrame` - Single frame of game data
  - `Peppi.PlayerFrame` - Player state for one frame
  - `Peppi.Controller` - Controller inputs
  - `Peppi.ReplayMeta` - Replay metadata
  - `Peppi.PlayerMeta` - Per-player metadata

  """

  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  use Rustler,
    otp_app: :exphil,
    crate: "exphil_peppi",
    # Skip compilation in prod OR when cargo isn't available (e.g., Docker runtime)
    skip_compilation?: Mix.env() == :prod or System.find_executable("cargo") == nil

  # ============================================================================
  # NIF Struct Definitions
  # ============================================================================

  defmodule Controller do
    @moduledoc "Controller state from Peppi NIF"
    defstruct [
      :main_stick_x,
      :main_stick_y,
      :c_stick_x,
      :c_stick_y,
      :l_trigger,
      :r_trigger,
      :button_a,
      :button_b,
      :button_x,
      :button_y,
      :button_z,
      :button_l,
      :button_r,
      :button_start,
      :button_d_up,
      :button_d_down,
      :button_d_left,
      :button_d_right
    ]
  end

  defmodule PlayerFrame do
    @moduledoc "Player state for a single frame"
    defstruct [
      :character,
      :x,
      :y,
      :percent,
      :stock,
      :facing,
      :action,
      :action_frame,
      :invulnerable,
      :jumps_left,
      :on_ground,
      :shield_strength,
      :hitstun_frames_left,
      :speed_air_x_self,
      :speed_ground_x_self,
      :speed_y_self,
      :speed_x_attack,
      :speed_y_attack,
      :controller
    ]
  end

  defmodule GameFrame do
    @moduledoc "A single parsed game frame"
    defstruct [:frame_number, :players]
  end

  defmodule PlayerMeta do
    @moduledoc "Player metadata from game start"
    defstruct [:port, :character, :character_name, :tag]
  end

  defmodule ReplayMeta do
    @moduledoc "Replay metadata"
    defstruct [:path, :stage, :duration_frames, :players]
  end

  defmodule ParsedReplay do
    @moduledoc "Complete parsed replay"
    defstruct [:frames, :metadata]
  end

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Parse a single .slp replay file.

  ## Options
    - `:player_port` - Filter to specific player port (1-4)

  ## Examples

      {:ok, replay} = Peppi.parse("game.slp")
      {:ok, replay} = Peppi.parse("game.slp", player_port: 1)

  """
  @spec parse(Path.t(), keyword()) :: {:ok, ParsedReplay.t()} | {:error, term()}
  def parse(path, opts \\ []) do
    player_port = Keyword.get(opts, :player_port)

    result =
      try do
        if player_port do
          parse_replay_for_port(path, player_port)
        else
          parse_replay(path)
        end
      rescue
        # Catch NIF panics (corrupted/malformed replay files)
        e in ErlangError ->
          case e do
            %ErlangError{original: :nif_panicked} ->
              {:error, {:corrupted_replay, path, "NIF panic - likely malformed frame data"}}
            _ ->
              reraise e, __STACKTRACE__
          end
      end

    case result do
      {:ok, replay} -> {:ok, replay}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Get replay metadata without parsing all frames.

  Faster than full parsing, useful for filtering replays before processing.

  ## Examples

      {:ok, meta} = Peppi.metadata("game.slp")
      meta.duration_frames
      #=> 5400

  """
  @spec metadata(Path.t()) :: {:ok, ReplayMeta.t()} | {:error, term()}
  def metadata(path) do
    try do
      get_replay_metadata(path)
    rescue
      e in ErlangError ->
        case e do
          %ErlangError{original: :nif_panicked} ->
            {:error, {:corrupted_replay, path, "NIF panic in metadata"}}
          _ ->
            reraise e, __STACKTRACE__
        end
    end
  end

  @doc """
  Parse multiple replay files in parallel.

  ## Options
    - `:player_port` - Filter to specific player port
    - `:max_concurrency` - Max parallel tasks (default: System.schedulers_online())

  ## Examples

      paths = Path.wildcard("replays/*.slp")
      results = Peppi.parse_many(paths)
      successful = Enum.filter(results, &match?({:ok, _}, &1))

  """
  @spec parse_many([Path.t()], keyword()) :: [{:ok, ParsedReplay.t()} | {:error, term()}]
  def parse_many(paths, opts \\ []) do
    max_concurrency = Keyword.get(opts, :max_concurrency, System.schedulers_online())

    paths
    |> Task.async_stream(
      fn path -> parse(path, opts) end,
      max_concurrency: max_concurrency,
      timeout: :infinity
    )
    |> Enum.map(fn {:ok, result} -> result end)
  end

  @doc """
  Convert parsed replay to training frame format.

  Returns frames in the format expected by `ExPhil.Training.Data.from_frames/2`.

  ## Options
    - `:player_port` - Port of the player to train (default: 1)
    - `:opponent_port` - Port of the opponent (default: 2)
    - `:frame_delay` - Simulated online delay in frames (default: 0)

  ## Frame Delay

  When `frame_delay: N` is set, each training pair uses:
  - Game state from frame (t - N) (what the agent "sees")
  - Controller action from frame t (what action was actually taken)

  This simulates Slippi online conditions where there's 18+ frame delay
  between observing the game state and your input taking effect.

  ## Examples

      # Normal training (no delay)
      {:ok, replay} = Peppi.parse("game.slp", player_port: 1)
      frames = Peppi.to_training_frames(replay, player_port: 1)

      # Simulated 18-frame online delay
      frames = Peppi.to_training_frames(replay, player_port: 1, frame_delay: 18)

  """
  @spec to_training_frames(ParsedReplay.t(), keyword()) :: [map()]
  def to_training_frames(%ParsedReplay{} = replay, opts \\ []) do
    player_port = Keyword.get(opts, :player_port, 1)
    opponent_port = Keyword.get(opts, :opponent_port, 2)
    frame_delay = Keyword.get(opts, :frame_delay, 0)

    if frame_delay == 0 do
      # No delay - standard training
      extract_frames_no_delay(replay, player_port, opponent_port)
    else
      # With delay - pair old states with current actions
      extract_frames_with_delay(replay, player_port, opponent_port, frame_delay)
    end
  end

  @doc """
  Convert parsed replay to training frames with detailed statistics.

  Returns `{frames, stats}` where stats tracks why frames were dropped.
  Use this to diagnose issues like wrong port selection or corrupted data.

  ## Options
    Same as `to_training_frames/2`

  ## Examples

      {:ok, replay} = Peppi.parse("game.slp")
      {frames, stats} = Peppi.to_training_frames_with_stats(replay, player_port: 1)

      if ParseStats.has_issues?(stats) do
        ParseStats.print_summary(stats)
      end

  """
  @spec to_training_frames_with_stats(ParsedReplay.t(), keyword()) ::
          {[map()], ExPhil.Data.ParseStats.t()}
  def to_training_frames_with_stats(%ParsedReplay{} = replay, opts \\ []) do
    alias ExPhil.Data.ParseStats

    player_port = Keyword.get(opts, :player_port, 1)
    opponent_port = Keyword.get(opts, :opponent_port, 2)
    frame_delay = Keyword.get(opts, :frame_delay, 0)

    # Track frame extraction with stats
    {valid_frames, stats} =
      ParseStats.track_extraction(replay.frames, player_port, opponent_port,
        frame_delay: frame_delay
      )

    # Extract player tag from metadata for style-conditional training
    player_tag = get_player_tag(replay.metadata, player_port)

    # Convert valid frames to training format
    frames =
      valid_frames
      |> Enum.map(fn frame ->
        %{
          game_state: build_game_state(frame, player_port, opponent_port, replay.metadata),
          controller: build_controller_state(Map.get(frame.players, player_port)),
          player_tag: player_tag
        }
      end)
      |> apply_frame_delay(frame_delay)

    {frames, stats}
  end

  # Apply frame delay pairing (state from t-delay, action from t)
  defp apply_frame_delay(frames, 0), do: frames

  defp apply_frame_delay(frames, delay) when length(frames) <= delay, do: []

  defp apply_frame_delay(frames, delay) do
    frame_array = :array.from_list(frames)
    num_frames = length(frames)

    delay..(num_frames - 1)
    |> Enum.map(fn t ->
      delayed = :array.get(t - delay, frame_array)
      current = :array.get(t, frame_array)

      %{
        game_state: delayed.game_state,
        controller: current.controller,
        player_tag: current[:player_tag],
        frame_delay: delay,
        observed_frame: t - delay,
        action_frame: t
      }
    end)
  end

  defp extract_frames_no_delay(replay, player_port, opponent_port) do
    # Extract player tag from metadata for style-conditional training
    player_tag = get_player_tag(replay.metadata, player_port)

    Enum.map(replay.frames, fn frame ->
      player = Map.get(frame.players, player_port)
      _opponent = Map.get(frame.players, opponent_port)

      %{
        game_state: build_game_state(frame, player_port, opponent_port, replay.metadata),
        controller: build_controller_state(player),
        player_tag: player_tag
      }
    end)
    |> Enum.filter(fn f -> f.game_state != nil and f.controller != nil end)
  end

  defp extract_frames_with_delay(replay, player_port, opponent_port, delay) do
    frames = replay.frames
    num_frames = length(frames)

    if num_frames <= delay do
      # Not enough frames for this delay
      []
    else
      # Extract player tag from metadata for style-conditional training
      player_tag = get_player_tag(replay.metadata, player_port)

      # Convert to array for O(1) lookups
      frame_array = :array.from_list(frames)

      # For each frame t >= delay, pair state_{t-delay} with action_t
      delay..(num_frames - 1)
      |> Enum.map(fn t ->
        # Delayed game state (what agent "sees")
        delayed_frame = :array.get(t - delay, frame_array)

        delayed_state =
          build_game_state(delayed_frame, player_port, opponent_port, replay.metadata)

        # Current action (what was actually done)
        current_frame = :array.get(t, frame_array)
        current_player = Map.get(current_frame.players, player_port)
        current_action = build_controller_state(current_player)

        %{
          game_state: delayed_state,
          controller: current_action,
          player_tag: player_tag,
          # Include delay metadata for debugging/analysis
          frame_delay: delay,
          observed_frame: t - delay,
          action_frame: t
        }
      end)
      |> Enum.filter(fn f -> f.game_state != nil and f.controller != nil end)
    end
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  # Extract player tag from replay metadata for style-conditional training
  defp get_player_tag(%ReplayMeta{players: players}, player_port) when is_list(players) do
    case Enum.find(players, fn p -> p.port == player_port end) do
      %PlayerMeta{tag: tag} when is_binary(tag) and tag != "" -> tag
      _ -> nil
    end
  end

  defp get_player_tag(_, _), do: nil

  defp build_game_state(frame, player_port, opponent_port, metadata) do
    player = Map.get(frame.players, player_port)
    opponent = Map.get(frame.players, opponent_port)

    if player do
      players = %{player_port => build_player(player)}

      players =
        if opponent, do: Map.put(players, opponent_port, build_player(opponent)), else: players

      distance =
        if player && opponent do
          dx = player.x - opponent.x
          dy = player.y - opponent.y
          :math.sqrt(dx * dx + dy * dy)
        else
          0.0
        end

      %GameState{
        frame: frame.frame_number,
        stage: metadata.stage,
        # IN_GAME
        menu_state: 2,
        players: players,
        projectiles: [],
        distance: distance
      }
    else
      nil
    end
  end

  defp build_player(%PlayerFrame{} = pf) do
    %Player{
      character: pf.character,
      x: pf.x,
      y: pf.y,
      percent: pf.percent,
      stock: pf.stock,
      facing: pf.facing,
      action: pf.action,
      action_frame: round(pf.action_frame),
      invulnerable: pf.invulnerable,
      jumps_left: pf.jumps_left,
      on_ground: pf.on_ground,
      shield_strength: pf.shield_strength,
      hitstun_frames_left: round(pf.hitstun_frames_left),
      speed_air_x_self: pf.speed_air_x_self,
      speed_ground_x_self: pf.speed_ground_x_self,
      speed_y_self: pf.speed_y_self,
      speed_x_attack: pf.speed_x_attack,
      speed_y_attack: pf.speed_y_attack,
      nana: nil,
      controller_state: nil
    }
  end

  defp build_controller_state(nil), do: nil
  defp build_controller_state(%PlayerFrame{controller: nil}), do: nil

  defp build_controller_state(%PlayerFrame{controller: c}) do
    %ControllerState{
      main_stick: %{x: c.main_stick_x, y: c.main_stick_y},
      c_stick: %{x: c.c_stick_x, y: c.c_stick_y},
      l_shoulder: c.l_trigger,
      r_shoulder: c.r_trigger,
      button_a: c.button_a,
      button_b: c.button_b,
      button_x: c.button_x,
      button_y: c.button_y,
      button_z: c.button_z,
      button_l: c.button_l,
      button_r: c.button_r,
      button_d_up: c.button_d_up
    }
  end

  # ============================================================================
  # NIF Stubs - These are replaced by the Rust implementations
  # ============================================================================

  defp parse_replay(_path), do: :erlang.nif_error(:nif_not_loaded)
  defp parse_replay_for_port(_path, _port), do: :erlang.nif_error(:nif_not_loaded)
  defp get_replay_metadata(_path), do: :erlang.nif_error(:nif_not_loaded)
end
