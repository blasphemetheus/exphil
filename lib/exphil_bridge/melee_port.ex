defmodule ExPhil.Bridge.MeleePort do
  @moduledoc """
  GenServer that manages communication with the Python melee_bridge.py script.

  Uses an Erlang Port for bidirectional communication with line-delimited JSON.

  ## Usage

      # Start the bridge
      {:ok, pid} = ExPhil.Bridge.MeleePort.start_link([])

      # Initialize with Dolphin/ISO paths
      :ok = ExPhil.Bridge.MeleePort.init_console(pid, %{
        dolphin_path: "/path/to/slippi",
        iso_path: "/path/to/melee.iso",
        character: "mewtwo",
        stage: "final_destination"
      })

      # Game loop
      {:ok, game_state} = ExPhil.Bridge.MeleePort.step(pid)

      # Send controller input
      :ok = ExPhil.Bridge.MeleePort.send_controller(pid, %{
        main_stick: %{x: 0.5, y: 1.0},
        buttons: %{a: true}
      })

      # Stop
      :ok = ExPhil.Bridge.MeleePort.stop(pid)

  """

  use GenServer
  require Logger

  @default_timeout 30_000

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the MeleePort GenServer.

  ## Options
    - `:python_path` - Path to Python executable (default: "python3")
    - `:script_path` - Path to melee_bridge.py (default: priv/python/melee_bridge.py)
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: opts[:name])
  end

  @doc """
  Initialize the Dolphin console and controller.

  ## Config
    - `:dolphin_path` - Path to Slippi/Dolphin folder (required)
    - `:iso_path` - Path to Melee 1.02 ISO (required)
    - `:controller_port` - Controller port (default: 1)
    - `:opponent_port` - Opponent port (default: 2)
    - `:character` - Character to select (atom or integer)
    - `:stage` - Stage to select (atom or integer)
    - `:online_delay` - Simulate online delay frames (default: 0)
  """
  def init_console(server, config, timeout \\ @default_timeout) do
    GenServer.call(server, {:init_console, config}, timeout)
  end

  @doc """
  Get the next game state.

  Returns `{:ok, game_state}` when in game, or `{:menu, game_state}` during menus.
  """
  def step(server, opts \\ [], timeout \\ @default_timeout) do
    GenServer.call(server, {:step, opts}, timeout)
  end

  @doc """
  Send controller input.

  ## Input format
      %{
        main_stick: %{x: 0.0..1.0, y: 0.0..1.0},
        c_stick: %{x: 0.0..1.0, y: 0.0..1.0},
        shoulder: 0.0..1.0,
        buttons: %{a: bool, b: bool, x: bool, y: bool, z: bool, l: bool, r: bool, d_up: bool}
      }
  """
  def send_controller(server, input, timeout \\ @default_timeout) do
    GenServer.call(server, {:send_controller, input}, timeout)
  end

  @doc """
  Ping the Python bridge to check if it's alive.
  """
  def ping(server, timeout \\ 5_000) do
    GenServer.call(server, :ping, timeout)
  end

  @doc """
  Stop the console and close the bridge.
  """
  def stop(server) do
    GenServer.call(server, :stop)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    python_path = Keyword.get(opts, :python_path, find_python())
    script_path = Keyword.get(opts, :script_path, default_script_path())

    Logger.info("[MeleePort] Starting Python bridge: #{script_path}")
    Logger.info("[MeleePort] Using Python: #{python_path}")

    port = Port.open(
      {:spawn_executable, python_path},
      [
        :binary,
        :exit_status,
        :use_stdio,
        :stderr_to_stdout,
        {:args, ["-u", script_path]},
        {:cd, File.cwd!()}
      ]
    )

    state = %{
      port: port,
      pending: nil,
      buffer: "",
      initialized: false
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:init_console, config}, from, state) do
    request = %{cmd: "init", config: normalize_config(config)}
    send_request(state.port, request)
    {:noreply, %{state | pending: {:init, from}}}
  end

  @impl true
  def handle_call({:step, opts}, from, state) do
    auto_menu = Keyword.get(opts, :auto_menu, true)
    request = %{cmd: "step", auto_menu: auto_menu}
    send_request(state.port, request)
    {:noreply, %{state | pending: {:step, from}}}
  end

  @impl true
  def handle_call({:send_controller, input}, from, state) do
    request = %{cmd: "send_controller", input: input}
    send_request(state.port, request)
    {:noreply, %{state | pending: {:send_controller, from}}}
  end

  @impl true
  def handle_call(:ping, from, state) do
    request = %{cmd: "ping"}
    send_request(state.port, request)
    {:noreply, %{state | pending: {:ping, from}}}
  end

  @impl true
  def handle_call(:stop, from, state) do
    request = %{cmd: "stop"}
    send_request(state.port, request)
    {:noreply, %{state | pending: {:stop, from}}}
  end

  @impl true
  def handle_info({port, {:data, data}}, %{port: port} = state) do
    # Accumulate data and try to parse complete JSON lines
    buffer = state.buffer <> data

    case parse_buffer(buffer) do
      {:ok, response, remaining} ->
        state = handle_response(response, state)
        {:noreply, %{state | buffer: remaining}}

      :incomplete ->
        {:noreply, %{state | buffer: buffer}}
    end
  end

  @impl true
  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.warning("[MeleePort] Python process exited with status: #{status}")

    if state.pending do
      {_type, from} = state.pending
      GenServer.reply(from, {:error, :port_closed})
    end

    {:stop, :normal, state}
  end

  @impl true
  def handle_info(msg, state) do
    Logger.debug("[MeleePort] Unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  @impl true
  def terminate(_reason, state) do
    if state.port do
      Port.close(state.port)
    end
    :ok
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp default_script_path do
    :code.priv_dir(:exphil)
    |> to_string()
    |> Path.join("python/melee_bridge.py")
  rescue
    # Fallback for development
    _ -> Path.join([File.cwd!(), "priv", "python", "melee_bridge.py"])
  end

  defp find_python do
    # Check for project venv first (has libmelee installed)
    venv_python = Path.join([File.cwd!(), ".venv", "bin", "python3"])

    cond do
      File.exists?(venv_python) ->
        venv_python

      # Check EXPHIL_PYTHON env var
      System.get_env("EXPHIL_PYTHON") ->
        System.get_env("EXPHIL_PYTHON")

      # Fall back to system python3
      true ->
        "python3"
    end
  end

  defp send_request(port, request) do
    json = Jason.encode!(request)
    Port.command(port, json <> "\n")
  end

  defp parse_buffer(buffer) do
    case String.split(buffer, "\n", parts: 2) do
      [line, remaining] when line != "" ->
        # Check if it's a log line (starts with [melee_bridge])
        if String.starts_with?(line, "[melee_bridge]") do
          Logger.debug("[Python] #{line}")
          parse_buffer(remaining)
        else
          case Jason.decode(line) do
            {:ok, response} -> {:ok, response, remaining}
            {:error, _} ->
              # Might be partial JSON or log output
              Logger.debug("[Python output] #{line}")
              parse_buffer(remaining)
          end
        end

      _ ->
        :incomplete
    end
  end

  defp handle_response(response, state) do
    case state.pending do
      nil ->
        Logger.warning("[MeleePort] Received response with no pending request: #{inspect(response)}")
        state

      {type, from} ->
        reply = format_reply(type, response)
        GenServer.reply(from, reply)
        %{state | pending: nil}
    end
  end

  defp format_reply(:init, %{"ok" => true} = response) do
    {:ok, %{controller_port: response["controller_port"]}}
  end

  defp format_reply(:step, %{"ok" => true} = response) do
    game_state = parse_game_state(response["game_state"])
    if response["is_menu"] do
      {:menu, game_state}
    else
      {:ok, game_state}
    end
  end

  defp format_reply(:send_controller, %{"ok" => true}), do: :ok

  defp format_reply(:ping, %{"ok" => true, "pong" => true}), do: :pong

  defp format_reply(:stop, %{"ok" => true}), do: :ok

  defp format_reply(_type, %{"error" => error}) do
    {:error, error}
  end

  defp format_reply(type, response) do
    Logger.warning("[MeleePort] Unexpected response for #{type}: #{inspect(response)}")
    {:error, :unexpected_response}
  end

  defp normalize_config(config) when is_map(config) do
    config
    |> Enum.map(fn
      {k, v} when is_atom(k) -> {Atom.to_string(k), normalize_value(v)}
      {k, v} -> {k, normalize_value(v)}
    end)
    |> Map.new()
  end

  defp normalize_value(v) when is_atom(v), do: Atom.to_string(v)
  defp normalize_value(v), do: v

  defp parse_game_state(nil), do: nil
  defp parse_game_state(gs) when is_map(gs) do
    %ExPhil.Bridge.GameState{
      frame: gs["frame"],
      stage: gs["stage"],
      menu_state: gs["menu_state"],
      players: parse_players(gs["players"]),
      projectiles: parse_projectiles(gs["projectiles"]),
      distance: gs["distance"]
    }
  end

  defp parse_players(nil), do: %{}
  defp parse_players(players) when is_map(players) do
    players
    |> Enum.map(fn {port, player} ->
      {String.to_integer(port), parse_player(player)}
    end)
    |> Map.new()
  end

  defp parse_player(nil), do: nil
  defp parse_player(p) when is_map(p) do
    %ExPhil.Bridge.Player{
      character: p["character"],
      x: p["x"],
      y: p["y"],
      percent: p["percent"],
      stock: p["stock"],
      facing: p["facing"],
      action: p["action"],
      action_frame: p["action_frame"],
      invulnerable: p["invulnerable"],
      jumps_left: p["jumps_left"],
      on_ground: p["on_ground"],
      shield_strength: p["shield_strength"],
      hitstun_frames_left: p["hitstun_frames_left"],
      speed_air_x_self: p["speed_air_x_self"],
      speed_ground_x_self: p["speed_ground_x_self"],
      speed_y_self: p["speed_y_self"],
      speed_x_attack: p["speed_x_attack"],
      speed_y_attack: p["speed_y_attack"],
      nana: parse_nana(p["nana"]),
      controller_state: parse_controller_state(p["controller_state"])
    }
  end

  defp parse_nana(nil), do: nil
  defp parse_nana(n) do
    %ExPhil.Bridge.Nana{
      x: n["x"],
      y: n["y"],
      percent: n["percent"],
      stock: n["stock"],
      action: n["action"],
      facing: n["facing"]
    }
  end

  defp parse_controller_state(nil), do: nil
  defp parse_controller_state(cs) do
    %ExPhil.Bridge.ControllerState{
      main_stick: parse_stick(cs["main_stick"]),
      c_stick: parse_stick(cs["c_stick"]),
      l_shoulder: cs["l_shoulder"],
      r_shoulder: cs["r_shoulder"],
      button_a: cs["button_a"],
      button_b: cs["button_b"],
      button_x: cs["button_x"],
      button_y: cs["button_y"],
      button_z: cs["button_z"],
      button_l: cs["button_l"],
      button_r: cs["button_r"],
      button_d_up: cs["button_d_up"]
    }
  end

  defp parse_stick(nil), do: %{x: 0.5, y: 0.5}
  defp parse_stick(s), do: %{x: s["x"], y: s["y"]}

  defp parse_projectiles(nil), do: []
  defp parse_projectiles(projs) when is_list(projs) do
    Enum.map(projs, fn p ->
      %ExPhil.Bridge.Projectile{
        owner: p["owner"],
        x: p["x"],
        y: p["y"],
        type: p["type"],
        subtype: p["subtype"],
        speed_x: p["speed_x"],
        speed_y: p["speed_y"]
      }
    end)
  end
end
