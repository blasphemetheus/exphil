defmodule ExPhil.Data.ReplayParser do
  @moduledoc """
  Parse Slippi replay files (.slp) into training data format.

  Uses py-slippi via Python subprocess for parsing, then converts
  the output to Elixir structs for use with the training pipeline.

  ## Usage

      # Parse a single replay
      {:ok, result} = ReplayParser.parse("game.slp")
      # result.frames contains list of %{game_state: ..., controller: ...}

      # Parse a directory of replays
      {:ok, stats} = ReplayParser.parse_directory("replays/", "parsed/",
        character: :mewtwo,
        min_length: 60
      )

      # Load a pre-parsed file
      {:ok, data} = ReplayParser.load_parsed("parsed/game.json.gz")

  ## Character Filtering

  Supported character atoms for filtering:
  - `:mewtwo`, `:game_and_watch`, `:link`, `:ganondorf`
  - `:fox`, `:falco`, `:marth`, `:sheik`, `:peach`, etc.

  """

  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  require Logger

  @python_script Path.join(:code.priv_dir(:exphil), "python/replay_parser.py")

  @doc """
  Parse a single .slp replay file.

  ## Options
    - `:player_port` - Which player's perspective to use (1-4)

  ## Returns
    `{:ok, result}` where result contains:
    - `:frames` - List of frame data
    - `:metadata` - Replay metadata
    - `:success` - Boolean

  ## Examples

      {:ok, result} = ReplayParser.parse("game.slp")
      length(result.frames)
      #=> 5400  (90 seconds at 60fps)

  """
  @spec parse(Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def parse(path, opts \\ []) do
    player_port = Keyword.get(opts, :player_port)

    args = ["parse", path, "-"]
    args = if player_port, do: args ++ ["--port", to_string(player_port)], else: args

    case run_parser(args) do
      {:ok, output} ->
        result = parse_json_output(output)
        {:ok, convert_result(result)}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Parse a directory of .slp files.

  ## Options
    - `:character` - Filter by character (atom like `:mewtwo`)
    - `:min_length` - Minimum game length in seconds
    - `:max_files` - Maximum number of files to process
    - `:player_port` - Which player's perspective (1-4)
    - `:workers` - Number of parallel workers (default: 4)

  ## Returns
    `{:ok, stats}` with parsing statistics:
    - `:total_files` - Total .slp files found
    - `:parsed_files` - Successfully parsed files
    - `:skipped_files` - Skipped due to filters
    - `:total_frames` - Total frames across all files

  ## Examples

      {:ok, stats} = ReplayParser.parse_directory(
        "replays/",
        "parsed/",
        character: :mewtwo,
        min_length: 60
      )

  """
  @spec parse_directory(Path.t(), Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def parse_directory(input_dir, output_dir, opts \\ []) do
    character = Keyword.get(opts, :character)
    min_length = Keyword.get(opts, :min_length, 0)
    max_files = Keyword.get(opts, :max_files)
    player_port = Keyword.get(opts, :player_port)
    workers = Keyword.get(opts, :workers, 4)

    args = ["parse_dir", input_dir, output_dir]
    args = if character, do: args ++ ["--character", character_name(character)], else: args
    args = if min_length > 0, do: args ++ ["--min-length", to_string(min_length)], else: args
    args = if max_files, do: args ++ ["--max-files", to_string(max_files)], else: args
    args = if player_port, do: args ++ ["--port", to_string(player_port)], else: args
    args = args ++ ["--workers", to_string(workers)]

    case run_parser(args) do
      {:ok, output} ->
        stats = parse_json_output(output)
        {:ok, stats}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Load a previously parsed replay file.

  Supports both `.json` and `.json.gz` (gzipped) formats.

  ## Examples

      {:ok, data} = ReplayParser.load_parsed("parsed/game.json.gz")
      data.frames
      #=> [%{game_state: ..., controller: ...}, ...]

  """
  @spec load_parsed(Path.t()) :: {:ok, map()} | {:error, term()}
  def load_parsed(path) do
    try do
      data = if String.ends_with?(path, ".gz") do
        path
        |> File.read!()
        |> :zlib.gunzip()
        |> Jason.decode!()
      else
        path
        |> File.read!()
        |> Jason.decode!()
      end

      {:ok, convert_result(data)}
    rescue
      e -> {:error, e}
    end
  end

  @doc """
  Load all parsed files from a directory.

  Returns a stream of loaded replay data for memory-efficient processing.

  ## Examples

      ReplayParser.stream_parsed("parsed/")
      |> Stream.flat_map(& &1.frames)
      |> Enum.take(1000)

  """
  @spec stream_parsed(Path.t()) :: Enumerable.t()
  def stream_parsed(dir) do
    dir
    |> Path.join("**/*.json*")
    |> Path.wildcard()
    |> Stream.map(fn path ->
      case load_parsed(path) do
        {:ok, data} -> data
        {:error, _} -> nil
      end
    end)
    |> Stream.reject(&is_nil/1)
  end

  @doc """
  Convert loaded replay data to format expected by Training.Data.

  This creates frames in the exact format that `Training.Data.from_frames/2` expects.

  ## Options
    - `:frame_delay` - Simulated online delay in frames (default: 0)

  ## Frame Delay

  When `frame_delay: N` is set, each training pair uses:
  - Game state from frame (t - N) (what the agent "sees")
  - Controller action from frame t (what action was actually taken)

  This simulates Slippi online conditions where there's 18+ frame delay
  between observing the game state and your input taking effect.

  ## Examples

      {:ok, data} = ReplayParser.load_parsed("game.json.gz")
      frames = ReplayParser.to_training_frames(data)

      # With 18-frame online delay simulation
      frames = ReplayParser.to_training_frames(data, frame_delay: 18)

  """
  @spec to_training_frames(map(), keyword()) :: [map()]
  def to_training_frames(%{frames: frames}, opts \\ []) do
    frame_delay = Keyword.get(opts, :frame_delay, 0)

    if frame_delay == 0 do
      # No delay - standard training
      Enum.map(frames, fn frame ->
        %{
          game_state: frame.game_state,
          controller: frame.controller
        }
      end)
    else
      # With delay - pair old states with current actions
      extract_with_delay(frames, frame_delay)
    end
  end

  defp extract_with_delay(frames, delay) do
    num_frames = length(frames)

    if num_frames <= delay do
      []
    else
      # Convert to array for O(1) lookups
      frame_array = :array.from_list(frames)

      delay..(num_frames - 1)
      |> Enum.map(fn t ->
        delayed_frame = :array.get(t - delay, frame_array)
        current_frame = :array.get(t, frame_array)

        %{
          game_state: delayed_frame.game_state,
          controller: current_frame.controller,
          frame_delay: delay,
          observed_frame: t - delay,
          action_frame: t
        }
      end)
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp run_parser(args) do
    python = System.find_executable("python3") || System.find_executable("python")

    if is_nil(python) do
      {:error, "Python not found in PATH"}
    else
      case System.cmd(python, [@python_script | args], stderr_to_stdout: true) do
        {output, 0} ->
          {:ok, output}

        {output, code} ->
          Logger.warning("Parser exited with code #{code}: #{output}")
          {:error, "Parser failed with code #{code}"}
      end
    end
  end

  defp parse_json_output(output) do
    # The output might have log messages before JSON, find the JSON part
    output
    |> String.split("\n")
    |> Enum.reverse()
    |> Enum.find_value(fn line ->
      case Jason.decode(line) do
        {:ok, data} -> data
        _ -> nil
      end
    end)
    |> case do
      nil -> %{"success" => false, "error" => "No valid JSON in output"}
      data -> data
    end
  end

  defp convert_result(data) when is_map(data) do
    frames = data
    |> Map.get("frames", [])
    |> Enum.map(&convert_frame/1)

    metadata = convert_metadata(Map.get(data, "metadata", %{}))

    %{
      frames: frames,
      metadata: metadata,
      success: Map.get(data, "success", true)
    }
  end

  defp convert_frame(frame) do
    %{
      game_state: convert_game_state(Map.get(frame, "game_state", %{})),
      controller: convert_controller(Map.get(frame, "controller")),
      metadata: Map.get(frame, "metadata", %{})
    }
  end

  defp convert_game_state(gs) when is_map(gs) do
    players = gs
    |> Map.get("players", %{})
    |> Enum.map(fn {port_str, player_data} ->
      port = String.to_integer(port_str)
      {port, convert_player(player_data)}
    end)
    |> Map.new()

    %GameState{
      frame: Map.get(gs, "frame", 0),
      stage: Map.get(gs, "stage", 0),
      menu_state: Map.get(gs, "menu_state", 2),
      players: players,
      projectiles: [],  # TODO: Parse projectiles
      distance: Map.get(gs, "distance", 0.0)
    }
  end

  defp convert_player(nil), do: nil
  defp convert_player(p) when is_map(p) do
    %Player{
      character: Map.get(p, "character", 0),
      x: Map.get(p, "x", 0.0),
      y: Map.get(p, "y", 0.0),
      percent: Map.get(p, "percent", 0.0),
      stock: Map.get(p, "stock", 4),
      facing: Map.get(p, "facing", 1),
      action: Map.get(p, "action", 0),
      action_frame: Map.get(p, "action_frame", 0),
      invulnerable: Map.get(p, "invulnerable", false),
      jumps_left: Map.get(p, "jumps_left", 2),
      on_ground: Map.get(p, "on_ground", true),
      shield_strength: Map.get(p, "shield_strength", 60.0),
      hitstun_frames_left: Map.get(p, "hitstun_frames_left", 0),
      speed_air_x_self: Map.get(p, "speed_air_x_self", 0.0),
      speed_ground_x_self: Map.get(p, "speed_ground_x_self", 0.0),
      speed_y_self: Map.get(p, "speed_y_self", 0.0),
      speed_x_attack: Map.get(p, "speed_x_attack", 0.0),
      speed_y_attack: Map.get(p, "speed_y_attack", 0.0),
      nana: nil,
      controller_state: convert_controller(Map.get(p, "controller_state"))
    }
  end

  defp convert_controller(nil), do: nil
  defp convert_controller(c) when is_map(c) do
    main_stick = Map.get(c, "main_stick", %{})
    c_stick = Map.get(c, "c_stick", %{})

    %ControllerState{
      main_stick: %{
        x: Map.get(main_stick, "x", 0.5),
        y: Map.get(main_stick, "y", 0.5)
      },
      c_stick: %{
        x: Map.get(c_stick, "x", 0.5),
        y: Map.get(c_stick, "y", 0.5)
      },
      l_shoulder: Map.get(c, "l_shoulder", 0.0),
      r_shoulder: Map.get(c, "r_shoulder", 0.0),
      button_a: Map.get(c, "button_a", false),
      button_b: Map.get(c, "button_b", false),
      button_x: Map.get(c, "button_x", false),
      button_y: Map.get(c, "button_y", false),
      button_z: Map.get(c, "button_z", false),
      button_l: Map.get(c, "button_l", false),
      button_r: Map.get(c, "button_r", false),
      button_d_up: Map.get(c, "button_d_up", false)
    }
  end

  defp convert_metadata(meta) do
    players = meta
    |> Map.get("players", %{})
    |> Enum.map(fn {port_str, info} ->
      port = if is_binary(port_str), do: String.to_integer(port_str), else: port_str
      {port, info}
    end)
    |> Map.new()

    %{
      path: Map.get(meta, "path"),
      stage: Map.get(meta, "stage", 0),
      duration_frames: Map.get(meta, "duration_frames", 0),
      player_port: Map.get(meta, "player_port"),
      opponent_port: Map.get(meta, "opponent_port"),
      players: players
    }
  end

  @character_names %{
    captain_falcon: "captain_falcon",
    donkey_kong: "donkey_kong",
    fox: "fox",
    game_and_watch: "game_and_watch",
    kirby: "kirby",
    bowser: "bowser",
    link: "link",
    luigi: "luigi",
    mario: "mario",
    marth: "marth",
    mewtwo: "mewtwo",
    ness: "ness",
    peach: "peach",
    pikachu: "pikachu",
    ice_climbers: "ice_climbers",
    jigglypuff: "jigglypuff",
    samus: "samus",
    yoshi: "yoshi",
    zelda: "zelda",
    sheik: "sheik",
    falco: "falco",
    young_link: "young_link",
    dr_mario: "dr_mario",
    roy: "roy",
    pichu: "pichu",
    ganondorf: "ganondorf"
  }

  defp character_name(atom) when is_atom(atom) do
    Map.get(@character_names, atom, Atom.to_string(atom))
  end

  defp character_name(string) when is_binary(string), do: string
end
