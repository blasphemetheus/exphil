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

  ## Error Handling

  Returns structured `ExPhil.Error.ReplayError` on failure:

      case Peppi.parse("game.slp") do
        {:ok, replay} -> process(replay)
        {:error, %ReplayError{reason: :nif_panic}} ->
          Logger.warning("Corrupted replay, skipping")
      end

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
  alias ExPhil.Error.ReplayError

  use Rustler,
    otp_app: :exphil,
    crate: "exphil_peppi",
    # Skip compilation in prod OR when cargo isn't available (e.g., Docker
    # runtime) OR on explicit request (EXPHIL_SKIP_NIF_COMPILE=1 — use the
    # prebuilt priv/native .so when the current rustc can't rebuild old
    # crate pins, e.g. ethnum E0512 under rustc 1.97; found 2026-08-02)
    skip_compilation?:
      Mix.env() == :prod or System.find_executable("cargo") == nil or
        System.get_env("EXPHIL_SKIP_NIF_COMPILE") == "1"

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
    @moduledoc """
    A single parsed game frame. The stage-event fields (Slippi spec >=
    3.18, task #3) are nil on older replays: FoD side-platform heights
    persist between events (consumers hold the last seen value);
    stadium_event/type carry the PS transformation state machine.
    """
    defstruct [
      :frame_number,
      :players,
      :fod_platform_left,
      :fod_platform_right,
      :stadium_event,
      :stadium_type,
      :whispy_direction
    ]
  end

  defmodule PlayerMeta do
    @moduledoc "Player metadata from game start"
    # netplay_name/netplay_code: Slippi online identity (display name +
    # connect code). Populated for netplay replays, nil for local/console.
    # Style-conditional training keys on netplay_name — in-game `tag`
    # (cartridge name tag) is blank in netplay.
    defstruct [:port, :character, :character_name, :tag, :netplay_name, :netplay_code]
  end

  defmodule ReplayMeta do
    @moduledoc "Replay metadata"
    defstruct [:path, :stage, :duration_frames, :players]
    @type t :: %__MODULE__{}
  end

  # INVARIANTS.md item 4: the channels this parser actually POPULATES in the
  # game states it emits. `to_training_frames/3` hardcodes `projectiles: []`
  # and never fills items, so a training config that asks for projectile or
  # item dims gets a constant-zero block — while the live bridge fills real
  # values (off-distribution on every frame; undetectable by the layout
  # canary). `Embeddings.config_for_source/2` intersects the requested
  # config with this list; the checkpoint records it; the agent honors it.
  @provided_channels [:players, :stage, :stage_internals, :distance]

  @doc "Channels this parser populates (see INVARIANTS.md item 4)."
  @spec provides() :: [atom()]
  def provides, do: @provided_channels

  @doc "True when this parser populates `channel` in emitted game states."
  @spec provides?(atom()) :: boolean()
  def provides?(channel), do: channel in @provided_channels

  defmodule ParsedReplay do
    @moduledoc "Complete parsed replay"
    defstruct [:frames, :metadata]
    @type t :: %__MODULE__{}
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
  @spec parse(Path.t(), keyword()) :: {:ok, ParsedReplay.t()} | {:error, ReplayError.t() | term()}
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
              {:error, ReplayError.new(:nif_panic, path: path)}

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
  @spec metadata(Path.t()) :: {:ok, ReplayMeta.t()} | {:error, ReplayError.t() | term()}
  def metadata(path) do
    try do
      get_replay_metadata(path)
    rescue
      e in ErlangError ->
        case e do
          %ErlangError{original: :nif_panicked} ->
            {:error, ReplayError.new(:nif_panic, path: path)}

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
    - `:frame_delay` - ADDITIONAL reaction delay in frames (default: 0)

  ## Label convention (INVARIANTS.md item 1, GOTCHA #113)

  Slippi records each frame's controller input on the frame whose
  (post-update) state it PRODUCED, so the raw pair (state[t],
  controller[t]) is label-leaked. Every training frame emitted here
  therefore pairs state[t] with the input ISSUED from it — raw
  controller[t+1] (`ExPhil.Data.LabelConvention` `:causal`). The last
  frame of a replay has no successor and is dropped.

  `frame_delay: k` is reaction delay ON TOP of that: state[t] is paired
  with raw controller[t+1+k]. `frame_delay: 0` is the causal pairing;
  the leaked pairing cannot be built.

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
    remap_ports = Keyword.get(opts, :remap_ports, false)
    replay = forward_fill_stadium(replay)

    frames =
      replay
      |> extract_frames(player_port, opponent_port)
      |> causal_pairs()
      |> apply_frame_delay(frame_delay)

    # :remap_ports — normalize the players map to %{1 => subject, 2 =>
    # opponent} regardless of actual ports. The embedding/reward layers
    # hardcode own_port 1 / opponent 2, so WITHOUT this a non-port-1
    # subject embeds as an all-zero self with the subject in the OPPONENT
    # slot (the 09-01 port-2 corpus corruption: build_game_state(f, 2, 2)
    # even dropped the real opponent and zeroed distance). Peppi frames
    # carry no projectiles/items, so port-keyed ownership is not affected.
    cond do
      remap_ports and (player_port != 1 or opponent_port != 2) ->
        Enum.map(frames, fn f ->
          %{f | game_state: remap_players(f.game_state, player_port, opponent_port)}
        end)

      not remap_ports and player_port != 1 ->
        # Half-migrated call site self-report (GOTCHA #107): a non-port-1
        # subject without remap is exactly the shape of the E1b/E1c
        # corruptions — downstream port-1 hardcodes (embedding, AWBC) will
        # read the wrong slot. Warn once per process.
        unless Process.get(:peppi_remap_warned) do
          Process.put(:peppi_remap_warned, true)

          require Logger

          Logger.warning(
            "[Peppi] to_training_frames(player_port: #{player_port}) without remap_ports: true — " <>
              "port-1-convention consumers (embed own_port=1, AWBC) will read the WRONG player (GOTCHA #107)"
          )
        end

        frames

      true ->
        frames
    end
  end

  defp remap_players(%GameState{players: players} = gs, player_port, opponent_port) do
    remapped =
      [{1, Map.get(players, player_port)}, {2, Map.get(players, opponent_port)}]
      |> Enum.reject(fn {_k, v} -> is_nil(v) end)
      |> Map.new()

    # Stamp the convention (GOTCHA #107 enforcement): downstream embedding
    # asserts that its own_port argument matches this stamp, turning a
    # swapped perspective into a loud crash instead of a silent corruption.
    %{gs | players: remapped, own_port: 1}
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

    replay = forward_fill_stadium(replay)
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
      |> causal_pairs()
      |> apply_frame_delay(frame_delay)

    {frames, stats}
  end

  # The structural label pairing: frame i's :controller becomes the input
  # ISSUED from state i (raw controller of frame i+1). Consecutive frames
  # must be exactly one game-frame apart (no dropped frames / boundaries);
  # a frame without a contiguous successor is dropped, as is the last one.
  # Every consumer of Peppi training frames gets this — there is no
  # option to skip it (GOTCHA #113 made the leaked pairing a default once).
  @spec causal_pairs([map()]) :: [map()]
  def causal_pairs(frames) do
    frames
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.flat_map(fn [f, next] ->
      if next.game_state.frame == f.game_state.frame + 1 and next.controller != nil,
        do: [%{f | controller: next.controller}],
        else: []
    end)
  end

  # Additional reaction delay on top of the causal pairing: state from
  # t-delay, (already successor-aligned) controller from t.
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

  # Forward-fill :stadium_type across frames (2026-08-24): the stream
  # sets it only ON transformation-event frames (~7 frames per event),
  # but each event announces the next layout INCLUDING the type-5
  # revert to normal — so the carried value IS the active-layout
  # tracker (the same semantics the PS RAM transform digit holds).
  # :stadium_event stays raw/event-only. nil until the first event =
  # normal-so-far, which downstream defaults handle.
  defp forward_fill_stadium(%ParsedReplay{frames: frames} = replay) do
    {filled, _} =
      Enum.map_reduce(frames, nil, fn frame, current ->
        current = Map.get(frame, :stadium_type) || current

        frame =
          if current != nil and Map.get(frame, :stadium_type) == nil,
            do: Map.put(frame, :stadium_type, current),
            else: frame

        {frame, current}
      end)

    %{replay | frames: filled}
  end

  defp extract_frames(replay, player_port, opponent_port) do
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

  # ============================================================================
  # Private Helpers
  # ============================================================================

  # Player identity for style-conditional training. Prefer the netplay
  # display name (present + distinct in online replays); fall back to the
  # in-game cartridge name tag (blank in netplay, but set on console).
  defp get_player_tag(%ReplayMeta{players: players}, player_port) when is_list(players) do
    case Enum.find(players, fn p -> p.port == player_port end) do
      %PlayerMeta{netplay_name: name} when is_binary(name) and name != "" -> name
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
        distance: distance,
        fod_platform_left: Map.get(frame, :fod_platform_left),
        fod_platform_right: Map.get(frame, :fod_platform_right),
        stadium_event: Map.get(frame, :stadium_event),
        stadium_type: Map.get(frame, :stadium_type),
        whispy_direction: Map.get(frame, :whispy_direction)
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
