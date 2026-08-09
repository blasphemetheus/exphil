defmodule ExPhil.Bridge.MeleePort do
  @moduledoc """
  GenServer that manages a live Dolphin/Slippi session — natively, via
  `libmelee_ex` (no Python).

  This module preserves the public API of the original Python-bridge
  implementation (`priv/python/melee_bridge.py` over an Erlang Port);
  callers are unchanged. Internally it now drives `Melee.Dolphin`
  (process/config), `Melee.Console` (ENet spectator stream),
  `Melee.Controller` (pipe input), and `Melee.MenuHelper` (menu
  navigation) directly.

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

  alias ExPhil.Bridge.ActionQueue
  alias ExPhil.Error.BridgeError

  # 90s: netplay Direct holds emulation (no frames advance) while waiting
  # for the opponent peer to connect — a 30s step timeout crashed the
  # bridge before a human could enter the code (2026-07-21 Direct smoke).
  # Harmless for live play: this is a MAX wait, steps normally return ~16ms.
  @default_timeout 90_000

  # Transparent step retries for blocking callers when the bridge is in
  # polling mode (no_frame every ~100ms). 880 × ~100ms ≈ 88s — sized to the
  # same netplay-Direct budget as @default_timeout, and just under it so a
  # hung console surfaces as {:error, :console_hung} instead of a caller
  # GenServer.call timeout.
  @max_no_frame_retries 880

  # Menu enum wire values (Melee.Enums.Menu)
  @menu_in_game 2
  @menu_sudden_death 3
  @menu_postgame 4
  @menu_character_select 0

  # Frames to wait at CSS for the dummy's CPU setup before starting anyway.
  @dummy_setup_timeout_frames 600

  # ============================================================================
  # Types
  # ============================================================================

  @typedoc "GenServer reference (pid, name, or via tuple)"
  @type server :: GenServer.server()

  @typedoc "Timeout in milliseconds"
  @type timeout_ms :: non_neg_integer()

  @typedoc "Controller input for sending to the game"
  @type controller_input :: %{
          optional(:main_stick) => %{x: float(), y: float()},
          optional(:c_stick) => %{x: float(), y: float()},
          optional(:shoulder) => float(),
          optional(:buttons) => %{
            optional(:a) => boolean(),
            optional(:b) => boolean(),
            optional(:x) => boolean(),
            optional(:y) => boolean(),
            optional(:z) => boolean(),
            optional(:l) => boolean(),
            optional(:r) => boolean(),
            optional(:d_up) => boolean()
          }
        }

  @typedoc "Console initialization config"
  @type init_config :: %{
          required(:dolphin_path) => String.t(),
          required(:iso_path) => String.t(),
          optional(:controller_port) => pos_integer(),
          optional(:opponent_port) => pos_integer(),
          optional(:character) => atom() | pos_integer(),
          optional(:stage) => atom() | pos_integer(),
          optional(:online_delay) => non_neg_integer(),
          optional(:console_timeout) => number()
        }

  @typedoc "Start link options (Python-era options are accepted and ignored)"
  @type start_option ::
          {:python_path, String.t()}
          | {:script_path, String.t()}
          | {:name, GenServer.name()}

  @typedoc "Result of a step operation"
  @type step_result ::
          {:ok, ExPhil.Bridge.GameState.t()}
          | {:menu, ExPhil.Bridge.GameState.t()}
          | {:postgame, ExPhil.Bridge.GameState.t()}
          | {:game_ended, String.t()}
          | :no_frame
          | {:error, term()}

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the MeleePort GenServer.

  Python-era options (`:python_path`, `:script_path`) are accepted for
  compatibility and ignored — there is no Python process anymore.
  """
  @spec start_link([start_option()]) :: GenServer.on_start()
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: opts[:name])
  end

  @doc """
  Initialize the Dolphin console and controller.

  ## Config
    - `:dolphin_path` - Path to Slippi/Dolphin folder or executable (required)
    - `:iso_path` - Path to Melee 1.02 ISO (required)
    - `:controller_port` - Controller port (default: 1)
    - `:opponent_port` - Opponent port (default: 2)
    - `:character` - Character to select (atom, string, or internal id)
    - `:stage` - Stage to select (atom, string, or internal id)
    - `:online_delay` - Simulate online delay frames (default: 0)
    - `:console_timeout` - Polling-mode timeout in seconds (default: 0.1).
      `step/3` returns `:no_frame` when no frame arrives in time (paused
      game, load screen) instead of blocking. Pass `0` for legacy blocking
      dispatch.
    - `:headless`, `:gfx_backend`, `:emulation_speed`, `:blocking_input`,
      `:slippi_port`, `:replay_dir`, `:no_audio`, `:window_width`,
      `:window_height` - Dolphin knobs, semantics unchanged from the
      Python bridge
    - `:connect_code`, `:user_home` - Slippi Direct netplay
    - `:dummy_mode`, `:dummy_character`, `:dummy_cpu_level` - opponent-port
      dummy (none|stand|shield|jump|walk|cpu|external)
  """
  @spec init_console(server(), init_config() | keyword(), timeout_ms()) ::
          {:ok, %{controller_port: pos_integer()}} | {:error, term()}
  def init_console(server, config, timeout \\ @default_timeout) do
    GenServer.call(server, {:init_console, config}, timeout)
  end

  @doc """
  Get the next game state.

  Returns `{:ok, game_state}` when in game, or `{:menu, game_state}` during menus.

  ## Options
    - `:auto_menu` - Navigate menus automatically (default: true)
    - `:poll` - Surface `:no_frame` when the console produced no frame within
      its polling timeout (paused game, load screen). Default false: no_frame
      is absorbed by transparent re-polling, preserving blocking semantics
      for callers that don't handle `:no_frame`. LRAS-capable runners MUST
      pass `poll: true` — completing the quit requires sending controller
      input between polls while the game is paused.
  """
  @spec step(server(), keyword(), timeout_ms()) :: step_result()
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

  An optional `:port` key routes the input to that port's controller instead
  of the main one — used to drive the opponent port from Elixir (reactive
  dummies, self-play). Requires the bridge initialized with a `dummy_mode`
  so the second controller exists; use `dummy_mode: "external"` for
  Elixir-driven ports.

  An optional `:delay` key (frames) holds the action in the frame-keyed
  queue until the console reports `current_frame + delay`.
  """
  @spec send_controller(server(), controller_input(), timeout_ms()) ::
          :ok | {:game_ended, String.t()} | {:error, term()}
  def send_controller(server, input, timeout \\ @default_timeout) do
    GenServer.call(server, {:send_controller, input}, timeout)
  end

  @doc """
  Liveness check (formerly pinged the Python process).
  """
  @spec ping(server(), timeout_ms()) :: :pong | {:error, term()}
  def ping(server, timeout \\ 5_000) do
    GenServer.call(server, :ping, timeout)
  end

  @doc """
  Stop the console and close the bridge.
  """
  @spec stop(server()) :: :ok | {:error, term()}
  def stop(server) do
    GenServer.call(server, :stop)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  defmodule State do
    @moduledoc false
    defstruct dolphin: nil,
              console: nil,
              controller: nil,
              dummy_controller: nil,
              controller_port: 1,
              opponent_port: 2,
              config: %{},
              running: false,
              polling: false,
              menu_helper: nil,
              dummy_menu_helper: nil,
              action_queue: ExPhil.Bridge.ActionQueue.new(),
              current_frame: nil,
              dummy_mode: "none",
              dummy_frame: 0,
              dummy_wait_frames: 0,
              dummy_ready_logged: false,
              dummy_timeout_logged: false,
              postgame_reported: false,
              postgame_left_at: nil,
              last_in_game: false
  end

  @impl true
  def init(_opts) do
    {:ok, %State{}}
  end

  @impl true
  def handle_call({:init_console, config}, _from, state) do
    config = normalize_config(config)

    case do_init(config, state) do
      {:ok, state} ->
        {:reply, {:ok, %{controller_port: state.controller_port}}, state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:step, opts}, _from, state) do
    auto_menu = Keyword.get(opts, :auto_menu, true)
    poll = Keyword.get(opts, :poll, false)

    {reply, state} = do_step(state, auto_menu, poll, 0)
    {:reply, reply, state}
  end

  def handle_call({:send_controller, input}, _from, state) do
    {reply, state} = do_send_controller(state, input)
    {:reply, reply, state}
  end

  def handle_call(:ping, _from, state), do: {:reply, :pong, state}

  def handle_call(:stop, _from, state) do
    state = teardown(state)
    {:reply, :ok, state}
  end

  @impl true
  def handle_info(msg, state) do
    Logger.debug("[MeleePort] Unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  @impl true
  def terminate(_reason, state) do
    teardown(state)
    :ok
  end

  # ============================================================================
  # Init
  # ============================================================================

  defp do_init(config, _state) do
    with :ok <- require_key(config, :dolphin_path),
         :ok <- require_key(config, :iso_path),
         {:ok, config} <- validate_dummy_config(config) do
      controller_port = Map.get(config, :controller_port, 1)
      opponent_port = Map.get(config, :opponent_port, 2)
      connect_code = to_string(Map.get(config, :connect_code) || "")
      online = connect_code != ""
      headless = truthy?(Map.get(config, :headless))
      dummy_mode = normalize_dummy_mode(Map.get(config, :dummy_mode, "none"), online)

      exi_inputs = truthy?(Map.get(config, :exi_inputs))

      if exi_inputs do
        Logger.info(
          "[MeleePort] exi_inputs: enabling the Bot Input Overrides gecko " <>
            "code (requires the ExiAI Dolphin build). Inputs still travel " <>
            "over the pipe; the build injects them via EXI in-game."
        )
      end

      console_timeout = Map.get(config, :console_timeout) || 0.1
      polling = console_timeout > 0

      blocking_input =
        case Map.get(config, :blocking_input) do
          nil -> headless
          v -> truthy?(v)
        end

      slippi_port = Map.get(config, :slippi_port) || 51_441

      with {:ok, dolphin} <-
             launch_dolphin(config, %{
               online: online,
               headless: headless,
               blocking_input: blocking_input,
               slippi_port: slippi_port,
               exi_inputs: exi_inputs,
               controller_ports:
                 if(dummy_mode == "none",
                   do: [controller_port],
                   else: [controller_port, opponent_port]
                 )
             }),
           {:ok, console} <-
             start_console(slippi_port, polling, console_timeout, blocking_input),
           :ok <- connect_console_with_retries(console, 5),
           {:ok, controller} <- start_controller(dolphin, controller_port, console),
           {:ok, dummy_controller} <-
             maybe_start_dummy_controller(dolphin, dummy_mode, opponent_port, console) do
        if online, do: Logger.info("[MeleePort] Netplay mode: connecting to #{connect_code}")

        {:ok,
         %State{
           dolphin: dolphin,
           console: console,
           controller: controller,
           dummy_controller: dummy_controller,
           controller_port: controller_port,
           opponent_port: opponent_port,
           config: Map.put(config, :connect_code, connect_code),
           running: true,
           polling: polling,
           menu_helper: Melee.MenuHelper.new(),
           dummy_menu_helper: if(dummy_mode != "none", do: Melee.MenuHelper.new()),
           dummy_mode: dummy_mode
         }}
      end
    end
  end

  defp require_key(config, key) do
    if Map.get(config, key), do: :ok, else: {:error, "#{key} is required"}
  end

  # GOTCHAS #57 family: dummy_mode/cpu_level interaction guards, ported
  # verbatim from melee_bridge.py.
  defp validate_dummy_config(config) do
    mode = to_string(Map.get(config, :dummy_mode, "none"))
    level = Map.get(config, :dummy_cpu_level, 0) || 0

    driven_modes = ~w(external stand shield jump walk)

    cond do
      mode in driven_modes and level > 0 ->
        Logger.error(
          "[MeleePort] dummy_mode=#{mode} is controller-driven but " <>
            "dummy_cpu_level=#{level} would hand the port to the game AI " <>
            "(inputs ignored). Forcing cpu_level=0. Use dummy_mode=cpu for a CPU."
        )

        {:ok, Map.put(config, :dummy_cpu_level, 0)}

      mode == "cpu" ->
        dchar = to_string(Map.get(config, :dummy_character, "fox"))

        cond do
          String.downcase(dchar) == "sheik" ->
            {:error,
             "dummy_character='sheik' cannot be a CPU (libmelee semantics; " <>
               "Sheik is reached via Zelda). Use 'zelda'."}

          level <= 0 ->
            Logger.error(
              "[MeleePort] dummy_mode=cpu with dummy_cpu_level=#{level} is NOT " <>
                "a CPU — the port stays HUMAN and idle. Defaulting to level 1."
            )

            {:ok, Map.put(config, :dummy_cpu_level, 1)}

          level > 9 ->
            Logger.error("[MeleePort] dummy_cpu_level=#{level} out of range; clamping to 9.")
            {:ok, Map.put(config, :dummy_cpu_level, 9)}

          true ->
            {:ok, config}
        end

      true ->
        {:ok, config}
    end
  end

  defp normalize_dummy_mode(mode, online) do
    mode = to_string(mode || "none")

    if online and mode != "none" do
      Logger.warning("[MeleePort] Netplay mode: opponent is remote — dummy disabled")
      "none"
    else
      mode
    end
  end

  # -- libmelee_ex touchpoints -------------------------------------------------

  defp launch_dolphin(config, %{
         online: online,
         headless: headless,
         blocking_input: blocking_input,
         slippi_port: slippi_port,
         exi_inputs: exi_inputs,
         controller_ports: controller_ports
       }) do
    opts =
      [
        path: Map.fetch!(config, :dolphin_path),
        iso_path: Map.fetch!(config, :iso_path),
        slippi_port: slippi_port,
        headless: headless,
        blocking_input: blocking_input,
        online_delay: Map.get(config, :online_delay) || 0,
        emulation_speed: (headless && (Map.get(config, :emulation_speed) || 1.0) * 1.0) || 1.0,
        save_replays: Map.get(config, :replay_dir) != nil,
        controller_ports: controller_ports,
        gecko_extra_codes:
          if(exi_inputs, do: ["$Optional: Allow Bot Input Overrides"], else: [])
      ]
      |> put_if(:gfx_backend, Map.get(config, :gfx_backend))
      |> put_if(:replay_dir, Map.get(config, :replay_dir))
      # Memory cards are off by default (the "create game data?" dialog
      # hangs menus). Enable when the session needs save data — e.g. to
      # use an in-game nametag.
      |> put_if(:memory_card, Map.get(config, :memory_card))
      |> put_if(:copy_home_from, online && Map.get(config, :user_home))

    Melee.Dolphin.launch(opts)
  end

  defp put_if(opts, _key, nil), do: opts
  defp put_if(opts, _key, false), do: opts
  defp put_if(opts, key, value), do: Keyword.put(opts, key, value)

  defp start_console(slippi_port, polling, console_timeout, blocking_input) do
    Melee.Console.start_link(
      port: slippi_port,
      polling_mode: polling,
      polling_timeout: round(console_timeout * 1000),
      blocking_input: blocking_input
    )
  end

  # Retry with backoff: a lone Dolphin is listening quickly, but parallel
  # instances boot slower (ISO read + shader cache contention) — ported
  # budget from melee_bridge.py (~45s total).
  defp connect_console_with_retries(console, attempts) do
    Enum.reduce_while(1..attempts, {:error, :never_tried}, fn attempt, _acc ->
      case Melee.Console.connect(console, 10_000) do
        :ok ->
          {:halt, :ok}

        {:error, reason} ->
          if attempt < attempts do
            wait = 2_000 * attempt
            Logger.warning("[MeleePort] console connect attempt #{attempt} failed, retrying in #{wait}ms")
            Process.sleep(wait)
            {:cont, {:error, reason}}
          else
            {:halt, {:error, {:connect_failed, reason}}}
          end
      end
    end)
  end

  defp start_controller(dolphin, port, console) do
    with {:ok, pipe} <- Melee.Dolphin.setup_controller(dolphin, port),
         {:ok, controller} <- Melee.Controller.start_link(pipe_path: pipe),
         :ok <- Melee.Controller.connect(controller, 60_000),
         :ok <- Melee.Console.register_controller(console, controller) do
      {:ok, controller}
    end
  end

  defp maybe_start_dummy_controller(_dolphin, "none", _port, _console), do: {:ok, nil}

  defp maybe_start_dummy_controller(dolphin, _mode, port, console) do
    start_controller(dolphin, port, console)
  end

  # ============================================================================
  # Step
  # ============================================================================

  defp do_step(%{running: false} = state, _auto_menu, _poll, _retries),
    do: {{:error, "Console not initialized"}, state}

  defp do_step(state, auto_menu, poll, retries) do
    case Melee.Console.step(state.console) do
      nil when poll ->
        {:no_frame, state}

      nil ->
        if retries >= @max_no_frame_retries do
          {{:error, :console_hung}, state}
        else
          do_step(state, auto_menu, poll, retries + 1)
        end

      {:error, :enet_disconnected} ->
        Logger.info("[MeleePort] Dolphin disconnected")
        {{:game_ended, "dolphin_disconnected"}, %{state | running: false}}

      {:ok, gamestate} ->
        handle_frame(state, gamestate, auto_menu)
    end
  end

  defp handle_frame(state, gamestate, auto_menu) do
    is_in_game = gamestate.menu_state in [@menu_in_game, @menu_sudden_death]
    is_postgame = gamestate.menu_state == @menu_postgame
    is_menu = not is_in_game

    # Local delay queue: track the frame clock and apply everything
    # scheduled for it (writes land in the pipe now, flush at the top of
    # the next console step — same one-step semantics as immediate sends).
    state =
      if is_in_game do
        {due, queue} = ActionQueue.pop_due(state.action_queue, gamestate.frame)
        Enum.each(due, &apply_input_now(state, &1))
        %{state | action_queue: queue, current_frame: gamestate.frame}
      else
        %{state | current_frame: nil}
      end

    # Reset the dummy-setup watchdog once the game is running (budget is
    # per character-select visit).
    state =
      if is_in_game and state.dummy_wait_frames > 0 do
        %{state | dummy_wait_frames: 0, dummy_timeout_logged: false}
      else
        state
      end

    # Track transitions for the postgame-report protocol.
    state = if is_in_game and not state.last_in_game, do: %{state | postgame_reported: false}, else: state

    # Post-game grace timer for :postgame_delay: stamp when the game ends
    # by ANY route — the in-game -> menu transition catches quit-outs
    # (pause-menu quit / LRAS skip the score screen entirely; missed
    # 2026-08-08 v1, which only stamped postgame frames).
    left_game? = state.last_in_game and not is_in_game

    state =
      if left_game? or is_postgame,
        do: %{state | postgame_left_at: System.monotonic_time(:millisecond)},
        else: state

    state = %{state | last_in_game: is_in_game}

    if rem(gamestate.frame, 60) == 0 do
      log_frame(gamestate)
    end

    # Skip menu navigation on the FIRST postgame frame so the caller can
    # decide (restart vs stop); navigate on subsequent frames.
    skip_menu_nav = is_postgame and not state.postgame_reported
    state = if is_postgame, do: %{state | postgame_reported: true}, else: state


    state =
      if is_menu and auto_menu and not skip_menu_nav do
        navigate_menus(state, gamestate)
      else
        state
      end

    # Scripted dummy behaviors run every in-game frame.
    state =
      if is_in_game and state.dummy_controller != nil and
           state.dummy_mode in ~w(stand shield jump walk) do
        drive_dummy(state, gamestate)
      else
        state
      end

    reply_state = convert_game_state(gamestate, state)

    reply =
      cond do
        is_postgame -> {:postgame, reply_state}
        is_menu -> {:menu, reply_state}
        true -> {:ok, reply_state}
      end

    {reply, state}
  end

  defp log_frame(gamestate) do
    p1 = gamestate.players[1]
    p2 = gamestate.players[2]

    fmt = fn
      nil -> "?"
      p -> "#{round(p.percent)}%/#{p.stock}stk"
    end

    Logger.info(
      "[MeleePort] Frame #{gamestate.frame}: menu_state=#{gamestate.menu_state} | " <>
        "P1:#{fmt.(p1)} P2:#{fmt.(p2)}"
    )
  end

  # -- Menu navigation ---------------------------------------------------------

  defp navigate_menus(state, gamestate) do
    character = to_character_id(Map.get(state.config, :character, :fox))
    stage = to_stage_id(Map.get(state.config, :stage, :final_destination))

    # Dummy picks first (no autostart) so the main helper's autostart can't
    # fire before the opponent is on the roster. ONLY during character
    # select: stage select has one shared cursor and a second controller
    # fights port 1 there.
    state =
      if state.dummy_menu_helper != nil and
           gamestate.menu_state == @menu_character_select do
        dummy_char = to_character_id(Map.get(state.config, :dummy_character, "fox"))

        helper =
          Melee.MenuHelper.step(state.dummy_menu_helper, gamestate, state.dummy_controller,
            port: state.opponent_port,
            character: dummy_char,
            stage: stage,
            cpu_level: Map.get(state.config, :dummy_cpu_level, 0) || 0,
            autostart: false,
            swag: false
          )

        %{state | dummy_menu_helper: helper}
      else
        state
      end

    {autostart, state} = dummy_ready(state, gamestate)

    # :postgame_delay (seconds, default 0): hold autostart after a game
    # ends so a human can change character at the CSS without racing the
    # START press. The bot still navigates (tag, character); it just
    # won't start the match until the grace period passes.
    autostart =
      case {autostart, Map.get(state.config, :postgame_delay), state.postgame_left_at} do
        {false, _, _} -> false
        {true, nil, _} -> true
        {true, _, nil} -> true
        {true, delay_s, left_at} ->
          System.monotonic_time(:millisecond) - left_at >= delay_s * 1000
      end

    helper =
      Melee.MenuHelper.step(state.menu_helper, gamestate, state.controller,
        port: state.controller_port,
        character: character,
        stage: stage,
        # nil for local play — init_console normalizes the config value to
        # a string ("" when offline), but MenuHelper treats "" as "drive
        # the DIRECT-code keyboard", which hijacks and crashes the nametag
        # flow (same name-entry scene; hit twice 2026-08-07).
        connect_code:
          case Map.get(state.config, :connect_code) do
            "" -> nil
            code -> code
          end,
        # In-game tag (max 4 chars) typed at the CSS by the helper's
        # nametag flow; needs memory_card save data or the tag list is
        # empty. Local showcase sessions use "EXPH".
        nametag: Map.get(state.config, :nametag),
        autostart: autostart,
        swag: false
      )

    %{state | menu_helper: helper}
  end

  # Autostart gate: don't press START while the dummy's CPU-level slider
  # dance is mid-flight (measured 2026-07-26: unconditional autostart left
  # the dummy HUMAN in 5 of 6 recordings). Ported from melee_bridge.py.
  # Once configuration has been observed, stay ready: CSS status is only
  # readable AT character select — later scenes (stage select) reset the
  # players map to unplugged defaults, and unthrottled headless menus rack
  # up frames fast enough to trip the watchdog spuriously after success.
  defp dummy_ready(%{dummy_mode: "cpu", dummy_ready_logged: true} = state, _gamestate),
    do: {true, state}

  # Outside character select the status bytes are meaningless; don't count.
  defp dummy_ready(%{dummy_mode: "cpu"} = state, %{menu_state: menu} = _gamestate)
       when menu != @menu_character_select,
       do: {false, state}

  defp dummy_ready(%{dummy_mode: "cpu"} = state, gamestate) do
    want = Map.get(state.config, :dummy_cpu_level, 0) || 0

    if want <= 0 do
      {true, state}
    else
      case gamestate.players[state.opponent_port] do
        nil ->
          {false, state}

        player ->
          state = %{state | dummy_wait_frames: state.dummy_wait_frames + 1}

          # ControllerStatus.CONTROLLER_CPU
          is_cpu = player.controller_status == 1

          ready = is_cpu and player.cpu_level == want and not player.is_holding_cpu_slider

          cond do
            ready ->
              state =
                if state.dummy_ready_logged do
                  state
                else
                  Logger.info(
                    "[MeleePort] Dummy CPU configured: port=#{state.opponent_port} " <>
                      "level=#{want} (after #{state.dummy_wait_frames} CSS frames)"
                  )

                  %{state | dummy_ready_logged: true}
                end

              {true, state}

            state.dummy_wait_frames > @dummy_setup_timeout_frames ->
              state =
                if state.dummy_timeout_logged do
                  state
                else
                  Logger.error(
                    "[MeleePort] Dummy CPU setup TIMED OUT after " <>
                      "#{state.dummy_wait_frames} frames — starting anyway. " <>
                      "Requested level=#{want}, port #{state.opponent_port} reports " <>
                      "controller_status=#{inspect(player.controller_status)} " <>
                      "cpu_level=#{inspect(player.cpu_level)}."
                  )

                  %{state | dummy_timeout_logged: true}
                end

              {true, state}

            true ->
              {false, state}
          end
      end
    end
  end

  defp dummy_ready(state, _gamestate), do: {true, state}

  # -- Scripted dummies --------------------------------------------------------

  defp drive_dummy(state, gamestate) do
    c = state.dummy_controller
    t = state.dummy_frame + 1
    Melee.Controller.release_all(c)

    case state.dummy_mode do
      "shield" ->
        if rem(t, 180) < 120, do: Melee.Controller.press_button(c, :r)

      "jump" ->
        if rem(t, 90) == 0, do: Melee.Controller.press_button(c, :y)

      "walk" ->
        x =
          case gamestate.players[state.opponent_port] do
            nil -> 0.0
            p -> p.position.x
          end

        cond do
          x > 20.0 -> Melee.Controller.tilt_analog(c, :main, 0.35, 0.5)
          x < -20.0 -> Melee.Controller.tilt_analog(c, :main, 0.65, 0.5)
          rem(t, 240) < 120 -> Melee.Controller.tilt_analog(c, :main, 0.65, 0.5)
          true -> Melee.Controller.tilt_analog(c, :main, 0.35, 0.5)
        end

      _stand ->
        :ok
    end

    %{state | dummy_frame: t}
  end

  # ============================================================================
  # Controller input
  # ============================================================================

  defp do_send_controller(%{running: false} = state, _input),
    do: {{:error, "Controller not initialized"}, state}

  defp do_send_controller(state, input) do
    delay = get_in_any(input, :delay)

    if delay && state.current_frame != nil do
      apply_at = state.current_frame + trunc(delay)
      queue = ActionQueue.schedule(state.action_queue, apply_at, input)
      {:ok, %{state | action_queue: queue}}
    else
      case apply_input_now(state, input) do
        :ok -> {:ok, state}
        error -> {error, state}
      end
    end
  end

  defp apply_input_now(state, input) do
    port = get_in_any(input, :port)

    target =
      cond do
        port != nil and trunc(port) == state.opponent_port ->
          state.dummy_controller || {:error, "No controller on port #{port} (enable a dummy_mode at init)"}

        true ->
          state.controller
      end

    case target do
      {:error, _} = error ->
        error

      controller ->
        apply_controller_input(controller, input)
        :ok
    end
  end

  @button_map [
    a: :a,
    b: :b,
    x: :x,
    y: :y,
    z: :z,
    l: :l,
    r: :r,
    d_up: :d_up,
    # Start is SEND-ONLY (LRAS game-quit for replay finalization); it is
    # deliberately absent from the observed controller-state contract.
    start: :start
  ]

  defp apply_controller_input(controller, input) do
    Melee.Controller.release_all(controller)

    main = get_in_any(input, :main_stick) || %{}
    Melee.Controller.tilt_analog(
      controller,
      :main,
      get_in_any(main, :x) || 0.5,
      get_in_any(main, :y) || 0.5
    )

    c = get_in_any(input, :c_stick) || %{}
    Melee.Controller.tilt_analog(
      controller,
      :c,
      get_in_any(c, :x) || 0.5,
      get_in_any(c, :y) || 0.5
    )

    Melee.Controller.press_shoulder(controller, :l, get_in_any(input, :shoulder) || 0.0)

    buttons = get_in_any(input, :buttons) || %{}

    for {name, button} <- @button_map, truthy?(get_in_any(buttons, name)) do
      Melee.Controller.press_button(controller, button)
    end

    :ok
  end

  # ============================================================================
  # GameState conversion (Melee.* structs -> ExPhil.Bridge.* structs)
  # ============================================================================

  defp convert_game_state(gamestate, state) do
    players =
      Map.new(gamestate.players, fn {port, player} ->
        {port, convert_player(player)}
      end)

    %ExPhil.Bridge.GameState{
      frame: gamestate.frame,
      stage: gamestate.stage,
      menu_state: gamestate.menu_state,
      players: players,
      own_port: detect_own_port(gamestate, Map.get(state.config, :connect_code, "")),
      projectiles: Enum.map(gamestate.projectiles, &convert_projectile/1),
      distance: gamestate.distance
    }
    |> ExPhil.Eval.StateStreamTrace.maybe_emit()
  end

  defp convert_player(nil), do: nil

  defp convert_player(p) do
    %ExPhil.Bridge.Player{
      character: p.character,
      x: p.position.x,
      y: p.position.y,
      percent: p.percent,
      stock: p.stock,
      facing: if(p.facing, do: 1, else: -1),
      action: p.action,
      action_frame: p.action_frame,
      invulnerable: p.invulnerable,
      jumps_left: p.jumps_left,
      on_ground: p.on_ground,
      shield_strength: p.shield_strength,
      hitstun_frames_left: p.hitstun_frames_left,
      speed_air_x_self: p.speed_air_x_self,
      speed_ground_x_self: p.speed_ground_x_self,
      speed_y_self: p.speed_y_self,
      speed_x_attack: p.speed_x_attack,
      speed_y_attack: p.speed_y_attack,
      nana: convert_nana(p.nana),
      controller_state: convert_controller_state(p.controller_state),
      connect_code: p.connectCode || "",
      nametag: p.nametag || ""
    }
  end

  defp convert_nana(nil), do: nil

  defp convert_nana(n) do
    %ExPhil.Bridge.Nana{
      x: n.position.x,
      y: n.position.y,
      percent: n.percent,
      stock: n.stock,
      action: n.action,
      facing: if(n.facing, do: 1, else: -1)
    }
  end

  defp convert_controller_state(nil), do: nil

  defp convert_controller_state(cs) do
    {mx, my} = cs.main_stick
    {cx, cy} = cs.c_stick

    %ExPhil.Bridge.ControllerState{
      main_stick: %{x: mx, y: my},
      c_stick: %{x: cx, y: cy},
      l_shoulder: cs.l_shoulder,
      r_shoulder: cs.r_shoulder,
      button_a: cs.button.a,
      button_b: cs.button.b,
      button_x: cs.button.x,
      button_y: cs.button.y,
      button_z: cs.button.z,
      button_l: cs.button.l,
      button_r: cs.button.r,
      button_d_up: cs.button.d_up
    }
  end

  defp convert_projectile(p) do
    %ExPhil.Bridge.Projectile{
      owner: p.owner,
      x: p.position.x,
      y: p.position.y,
      type: p.type,
      subtype: p.subtype,
      speed_x: p.speed.x,
      speed_y: p.speed.y
    }
  end

  # Which in-game port is the bot, under Slippi Online? We know the
  # OPPONENT's connect code (it's what we searched for); the bot is the
  # other tagged player. nil offline / ambiguous.
  defp detect_own_port(_gamestate, ""), do: nil

  defp detect_own_port(gamestate, opponent_code) do
    norm = fn c -> (c || "") |> String.trim() |> String.upcase() end
    opp = norm.(opponent_code)

    codes =
      for {port, p} <- gamestate.players, p != nil, into: %{} do
        {port, norm.(p.connectCode)}
      end

    others = for {port, c} <- codes, c != "" and c != opp, do: port

    case others do
      [port] ->
        port

      _ ->
        matches = for {port, c} <- codes, c == opp, do: port

        with [m] <- matches,
             2 <- map_size(codes) do
          Enum.find(Map.keys(codes), &(&1 != m))
        else
          _ -> nil
        end
    end
  end

  # ============================================================================
  # Teardown / misc helpers
  # ============================================================================

  defp teardown(state) do
    if state.controller, do: safe(fn -> Melee.Controller.disconnect(state.controller) end)
    if state.dummy_controller, do: safe(fn -> Melee.Controller.disconnect(state.dummy_controller) end)
    if state.console, do: safe(fn -> Melee.Console.stop(state.console) end)
    if state.dolphin, do: safe(fn -> Melee.Dolphin.stop(state.dolphin) end)

    %{state | running: false, controller: nil, dummy_controller: nil, console: nil, dolphin: nil}
  end

  defp safe(fun) do
    fun.()
  catch
    kind, reason -> Logger.debug("[MeleePort] cleanup: #{inspect({kind, reason})}")
  end

  defp normalize_config(config) when is_list(config), do: normalize_config(Map.new(config))

  defp normalize_config(config) when is_map(config) do
    Map.new(config, fn
      {k, v} when is_binary(k) -> {String.to_existing_atom(k), v}
      {k, v} -> {k, v}
    end)
  end

  defp truthy?(nil), do: false
  defp truthy?(false), do: false
  defp truthy?(_), do: true

  # Input maps may arrive with atom or string keys depending on the caller.
  defp get_in_any(map, key) when is_map(map),
    do: Map.get(map, key) || Map.get(map, Atom.to_string(key))

  defp get_in_any(_map, _key), do: nil

  defp to_character_id(v) when is_integer(v), do: v
  defp to_character_id(v) when is_atom(v), do: Melee.Enums.Character.to_id(v)

  defp to_character_id(v) when is_binary(v),
    do: v |> String.downcase() |> String.to_existing_atom() |> Melee.Enums.Character.to_id()

  defp to_stage_id(v) when is_integer(v), do: v
  defp to_stage_id(v) when is_atom(v), do: Melee.Enums.Stage.to_id(v)

  defp to_stage_id(v) when is_binary(v),
    do: v |> String.downcase() |> String.to_existing_atom() |> Melee.Enums.Stage.to_id()

  # Error struct kept for API compatibility with error-matching callers.
  @doc false
  def bridge_error(reason), do: BridgeError.new(reason, bridge: :melee_port)
end
