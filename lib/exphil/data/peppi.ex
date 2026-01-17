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
    crate: "exphil_peppi"

  # ============================================================================
  # NIF Struct Definitions
  # ============================================================================

  defmodule Controller do
    @moduledoc "Controller state from Peppi NIF"
    defstruct [
      :main_stick_x, :main_stick_y,
      :c_stick_x, :c_stick_y,
      :l_trigger, :r_trigger,
      :button_a, :button_b, :button_x, :button_y,
      :button_z, :button_l, :button_r, :button_start,
      :button_d_up, :button_d_down, :button_d_left, :button_d_right
    ]
  end

  defmodule PlayerFrame do
    @moduledoc "Player state for a single frame"
    defstruct [
      :character, :x, :y, :percent, :stock, :facing,
      :action, :action_frame, :invulnerable, :jumps_left,
      :on_ground, :shield_strength, :hitstun_frames_left,
      :speed_air_x_self, :speed_ground_x_self, :speed_y_self,
      :speed_x_attack, :speed_y_attack, :controller
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

    result = if player_port do
      parse_replay_for_port(path, player_port)
    else
      parse_replay(path)
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
    get_replay_metadata(path)
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

  ## Examples

      {:ok, replay} = Peppi.parse("game.slp", player_port: 1)
      frames = Peppi.to_training_frames(replay, player_port: 1)
      dataset = ExPhil.Training.Data.from_frames(frames)

  """
  @spec to_training_frames(ParsedReplay.t(), keyword()) :: [map()]
  def to_training_frames(%ParsedReplay{} = replay, opts \\ []) do
    player_port = Keyword.get(opts, :player_port, 1)
    opponent_port = Keyword.get(opts, :opponent_port, 2)

    Enum.map(replay.frames, fn frame ->
      player = Map.get(frame.players, player_port)
      opponent = Map.get(frame.players, opponent_port)

      %{
        game_state: build_game_state(frame, player_port, opponent_port, replay.metadata),
        controller: build_controller_state(player)
      }
    end)
    |> Enum.filter(fn f -> f.game_state != nil and f.controller != nil end)
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp build_game_state(frame, player_port, opponent_port, metadata) do
    player = Map.get(frame.players, player_port)
    opponent = Map.get(frame.players, opponent_port)

    if player do
      players = %{player_port => build_player(player)}
      players = if opponent, do: Map.put(players, opponent_port, build_player(opponent)), else: players

      distance = if player && opponent do
        dx = player.x - opponent.x
        dy = player.y - opponent.y
        :math.sqrt(dx * dx + dy * dy)
      else
        0.0
      end

      %GameState{
        frame: frame.frame_number,
        stage: metadata.stage,
        menu_state: 2,  # IN_GAME
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
