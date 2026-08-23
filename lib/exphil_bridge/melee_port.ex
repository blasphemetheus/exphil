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
  @menu_stage_select 1

  # Frames to wait at CSS for the dummy's CPU setup before starting anyway.
  @dummy_setup_timeout_frames 600

  # External stage ids for :require_stage (the live game-state stage
  # field; NOT the internal ids menu navigation targets). Legal pool.
  @external_stage_ids %{
    fountain_of_dreams: 2, fod: 2, pokemon_stadium: 3, ps: 3,
    yoshis_story: 8, ys: 8, dreamland: 28, dl: 28,
    battlefield: 31, bf: 31, final_destination: 32, fd: 32
  }

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
    - `:menu_stuck_frames` - menu-watchdog threshold: frames of zero menu
      progress before the stuck warning fires (default 1800 = 30s)
    - `:menu_stuck_notify` - pid to send `{:melee_port, :menu_stuck,
      %{menu_state: m, frames: n}}` when the watchdog trips (once per
      stall episode), so harnesses can abort a wedged session instead of
      burning their deadline
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
              last_in_game: false,
              require_stage_id: nil,
              force_quit: false,
              stage_rejects: 0
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
                 |> add_human_adapter_port(config, headless, online)
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
           dummy_mode: dummy_mode,
           require_stage_id: to_external_stage_id(Map.get(config, :require_stage))
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

  # Human controller port (opt-in): EXPHIL_HUMAN_PORT=2 (or config
  # :human_port) declares that port as :gcn_adapter, so declare_ports/2
  # doesn't unplug it — without this, a fresh temp home makes the human
  # visit Dolphin's controller menu every single launch. Windowed local
  # sessions only: headless and netplay Dolphins must never claim the
  # GC adapter (it blocks the human's own Slippi session — 2026-08-09).
  defp add_human_adapter_port(ports, config, headless, online) do
    raw = Map.get(config, :human_port) || System.get_env("EXPHIL_HUMAN_PORT")

    with false <- headless or online or raw in [nil, ""],
         {port, _} when port in 1..4 <- Integer.parse(to_string(raw)) do
      if port in ports do
        Logger.warning("[MeleePort] human_port #{port} is already a bot port; ignoring")
        ports
      else
        ports ++ [{port, :gcn_adapter}]
      end
    else
      _ -> ports
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
      # RAM scene watching (MEMORY_WATCH_PROGRAM app #9, 2026-08-22b):
      # default ON for online sessions — the netplay-beta build streams
      # no live online-CSS state (GOTCHA #101), and the watcher's scene
      # word is what semi-closes the blind CSS fallback (BlindCss).
      # EXPHIL_MEMORY_WATCH=1 forces on for any session, =0 disables.
      |> put_if(
        :memory_watch,
        case System.get_env("EXPHIL_MEMORY_WATCH") do
          "0" -> false
          "1" -> Melee.MemoryMap.menu_with_canary() ++ Melee.MemoryMap.direct_code()
          _ -> online && Melee.MemoryMap.menu_with_canary() ++ Melee.MemoryMap.direct_code()
        end
      )

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

    # :require_stage (2026-08-09): Slippi Direct's game 1 is a RANDOM
    # legal stage (loser picks thereafter) — uncontrollable client-side.
    # Stage-controlled netplay evals therefore FILTER: if the game opens
    # on the wrong stage, LRAS out immediately and let the session
    # requeue (expect ~6 draws per specific stage). The decider of
    # 2026-08-09 was stage-confounded for exactly this reason.
    state =
      if is_in_game and not state.last_in_game and state.require_stage_id != nil and
           gamestate.stage != state.require_stage_id do
        n = state.stage_rejects + 1

        Logger.warning(
          "[MeleePort] stage #{gamestate.stage} != required #{state.require_stage_id} — " <>
            "LRAS requeue (reject ##{n})"
        )

        %{state | force_quit: true, stage_rejects: n}
      else
        state
      end

    # Clear the filter once the rejected game has been exited.
    state =
      if state.force_quit and not is_in_game and state.last_in_game,
        do: %{state | force_quit: false},
        else: state

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

    # #10 delay probe (2026-08-23): RAM frame counter (0x80479D60,
    # already in the menu watch set as :menu_frame) minus the stream
    # event's frame stamp, at arrival. Local sync baseline = +123
    # exactly (quartet, zero jitter); any drift here decomposes the
    # netplay session's effective delay into local vs network parts.
    # Every 300 in-game frames — 5s cadence, negligible cost.
    if is_in_game and rem(gamestate.frame, 300) == 0 do
      with watcher when watcher != nil <- state.dolphin && state.dolphin.memory_watcher,
           {:ok, ram_frame} <- safe_watch_get(watcher, :menu_frame) do
        Logger.info(
          "[MeleePort] #10 delay probe: ram_frame - stream_frame = " <>
            "#{ram_frame - gamestate.frame} (local-sync baseline 123) at frame #{gamestate.frame}"
        )
      else
        _ -> :ok
      end
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

    # Stage-filter quit: drive LRAS on the main controller (agent inputs
    # are dropped in do_send_controller while force_quit). START is
    # PULSED, not held — a pause instead of a quit needs a fresh edge
    # (the sd_until_game_end lesson).
    if is_in_game and state.force_quit do
      c = state.controller
      Melee.Controller.release_all(c)
      Melee.Controller.press_button(c, :l)
      Melee.Controller.press_button(c, :r)
      Melee.Controller.press_button(c, :a)
      if rem(gamestate.frame, 2) == 0, do: Melee.Controller.press_button(c, :start)
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
    # Frame counter for the loading animation + blind-fallback phases
    # (menu frames don't advance gamestate.frame reliably; GOTCHA #101
    # diagnosis history).
    css_n = Process.get(:css_debug_n, 0)
    Process.put(:css_debug_n, css_n + 1)

    # One RAM scene-word read per menu frame (nil/dead watcher -> :unknown).
    scene_word = ExPhil.Bridge.BlindCss.observe(state.dolphin && state.dolphin.memory_watcher)

    # Log scene-word CHANGES (menu frames only): the science trace the
    # 08-22 evening session lacked — code-entry minor, pending/previous
    # byte order, match-start word all land in the log for free.
    if scene_word != :unknown and Process.get(:last_scene_word) != scene_word do
      prev_word = Process.get(:last_scene_word)
      Process.put(:last_scene_word, scene_word)
      hex = scene_word |> Integer.to_string(16) |> String.pad_leading(8, "0")

      # The all-zero word is a known ~17ms load-noise transient (bot14
      # capture + tonight's session) — label it so the honest decode
      # ({:settled, :press_start}) doesn't read as a real scene visit.
      transient = if scene_word == 0, do: " (zero-word transient — load noise)", else: ""

      Logger.info(
        "[MeleePort] RAM scene word -> 0x#{hex} " <>
          "#{inspect(Melee.MemoryMap.scene_view(scene_word))}#{transient}"
      )

      # ONLINE GAME -> CSS word transition = a game just ended and the
      # pick was consumed. This is the RAM-only re-arm signal the
      # stream cannot provide: the whole online flow reports
      # menu_state 6 (CSS, code entry, search, SSS flash), so the
      # stream-based reset below never fires online — the 08-22 bot14
      # MENU STUCK (post-game CSS, fallback spent, unpicked forever).
      # The word held :slippi_online_game through the game (no menu
      # frames update it), so prev is unfakeable by menu flickers.
      if prev_word != nil and
           Melee.MemoryMap.scene_name(prev_word) == :slippi_online_game and
           Melee.MemoryMap.scene_name(scene_word) == :slippi_online_css do
        Process.put(:css_blind_done, false)
        Process.put(:css_blind_n, 0)
        Process.put(:css_blind_retries, 0)
        Logger.warning("[MeleePort] blind CSS: game ended (RAM scene word) — fallback re-armed")
      end
    end

    # Direct-code buffer readback (0x804A0740, 2026-08-23): feeds
    # MenuHelper's verify-before-confirm (:code_buffer option) and the
    # science log. :unknown without a watcher = legacy blind typing.
    code_buffer =
      if watcher = state.dolphin && state.dolphin.memory_watcher do
        code =
          Melee.MemoryMap.direct_code()
          |> Enum.map(fn {name, _} ->
            case safe_watch_get(watcher, name) do
              {:ok, v} -> v
              :unknown -> :unknown
            end
          end)
          |> Melee.MemoryMap.decode_direct_code()

        if code != "" and Process.get(:last_code_buf) != code do
          Process.put(:last_code_buf, code)
          Logger.info("[MeleePort] direct-code buffer reads \"#{code}\"")
        end

        code
      else
        :unknown
      end

    # Re-arm the blind CSS fallback when the session leaves the online
    # CSS scene (postgame flow passes through non-6 menu states before
    # the next CSS). Without this the done-flag persisted across games
    # and the post-game CSS sat unpicked forever — the 08-22 evening
    # MENU STUCK (ram_scene settled at online CSS, traffic healthy).
    if gamestate.menu_state != 6 and
         (Process.get(:css_blind_done, false) or Process.get(:css_blind_n, 0) > 0) do
      Process.put(:css_blind_done, false)
      Process.put(:css_blind_n, 0)
      Process.put(:css_blind_retries, 0)
      Logger.info("[MeleePort] blind CSS: left online-CSS scene — fallback re-armed")
    end

    # RAM menu-GameState merge (MEMORY_WATCH_PROGRAM core-plumbing,
    # 2026-08-23): overlay watcher CSS observations (cursor, hover,
    # selected/coin, status, ready banner) onto the stream gamestate
    # before any helper sees it — MenuHelper's feedback steering then
    # runs on RAM truth where the stream lies (dead coin_down offline,
    # frozen snapshot online, GOTCHA #101). Strictly additive: only
    # observed fields substitute, no watcher = stream unchanged.
    # OFFLINE CSS: on by default (addresses verified 08-22c).
    # ONLINE CSS: behind EXPHIL_RAM_MENU=1 until the cursor block +
    # selected array are validated at that scene (owed next Direct
    # session) — a stale heap address there would feed the helper
    # garbage cursors.
    gamestate = ram_menu_merge(state, gamestate)

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
        # The dummy is NOT being stepped here — but Dolphin holds a pipe
        # controller's LAST written state, so a tilt held at the moment
        # START fired keeps dragging the stage select screen's ONE shared
        # cursor while port 1 fights it back (observed 2026-08-09: the
        # hand bobbing vertically over FD for ~5s, selection only landing
        # once frames_on_stage accumulated through the oscillation).
        # Release everything the moment we're on a shared-cursor screen.
        if state.dummy_controller != nil and gamestate.menu_state == @menu_stage_select do
          Melee.Controller.release_all(state.dummy_controller)
        end

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

    # Menu watchdog (MenuHelper :on_stuck, libmelee_ex 2026-08-09): a
    # wedged menu used to look exactly like "still working" until some
    # external deadline killed the session — the steer_toward deadzone
    # freeze burned 3 pool-eval runs x ~7 minutes at stage select before
    # anything spoke up. The helper fires once per stall episode; we log
    # loudly and optionally message a watcher so harnesses can abort
    # instead of burning their deadline.
    #   :menu_stuck_frames  - frames of zero menu progress (default 1800
    #                         = 30s; netplay matchmaking waits stay under)
    #   :menu_stuck_notify  - pid to send {:melee_port, :menu_stuck, report}
    stuck_notify = Map.get(state.config, :menu_stuck_notify)
    stuck_watcher = state.dolphin && state.dolphin.memory_watcher

    on_stuck = fn report ->
      # RAM ground truth (MEMORY_WATCH_PROGRAM apps #2/#12): the scene
      # view names the actual screen (the stream collapses login/boot
      # dialogs to menu_state 255), and the traffic delta over ~250ms
      # separates "core wedged" (zero — dolphin sends a datagram every
      # step) from "core running, menuing stuck" (positive).
      report = Map.merge(report, ram_menu_diagnosis(stuck_watcher))

      # #2 watchdog suppression (2026-08-23): a legitimate hold — a
      # committed scene transition, or the post-pick online wait —
      # logs quietly instead of alarming, and re-arms the helper's
      # stuck detector (flag consumed in helper_drive below) so the
      # verdict is re-evaluated every stuck window: a hold that decays
      # into a core wedge alarms one window later. The bot14 wedge
      # class (online CSS, fallback NOT done) still alarms.
      case ExPhil.Bridge.StuckPolicy.verdict(report, Process.get(:css_blind_done, false)) do
        {:suppress, reason} ->
          Process.put(:menu_stuck_suppressed, true)

          Logger.info(
            "[MeleePort] menu hold (#{inspect(reason)}): no menu progress for " <>
              "#{report.frames} frames, RAM says legitimate " <>
              "(ram_scene=#{inspect(report.ram_scene)} " <>
              "ram_traffic_delta=#{inspect(report.ram_traffic_delta)}) — watchdog re-armed"
          )

        :alarm ->
          Logger.error(
            "[MeleePort] MENU STUCK: no menu progress for #{report.frames} frames " <>
              "(menu_state=#{report.menu_state} ram_scene=#{inspect(report.ram_scene)} " <>
              "ram_traffic_delta=#{inspect(report.ram_traffic_delta)}) — wedged session? " <>
              "(login screen, adapter contention, cursor not converging)"
          )

          if is_pid(stuck_notify), do: send(stuck_notify, {:melee_port, :menu_stuck, report})
      end
    end

    # :menu_ready_check (2026-08-21, warmup-race fix follow-up): a
    # zero-arity fun; while it returns false at the CHARACTER SELECT
    # screen, the bot does not navigate or confirm — it idles with a
    # visible slow stick wiggle (the "still loading" animation). Melee
    # won't start a match until every port confirms, so the unconfirmed
    # bot is an engine-enforced interlock against starting a game on a
    # half-JIT'd policy. The check MUST be non-blocking (a GenServer
    # call into the warming Agent would queue behind the compile — the
    # exact bug this replaces); callers use a :persistent_term flag.
    ready_check = Map.get(state.config, :menu_ready_check)
    online? = Map.get(state.config, :connect_code) not in [nil, ""]

    at_css? = gamestate.menu_state in [@menu_character_select, 6]

    # DEFAULT ON for online since 2026-08-22 (opt out with =0): the
    # netplay-beta build streams NO live online-CSS state — only a frame
    # counter ticks in the menu payload; cursor/character/coin are a
    # one-shot scene-entry snapshot (raw-dump diagnosis,
    # eval_runs/0822_netplay_crown/RESULTS.md). The snapshot holds the
    # account's PREVIOUSLY SELECTED character, so feedback menuing
    # "worked" only until the bot's own past fox picks poisoned it into
    # reading fox pre-pick. Online CSS is open-loop territory, period.
    blind_fallback? =
      online? and gamestate.menu_state == 6 and
        System.get_env("EXPHIL_CSS_BLIND_FALLBACK") != "0" and
        not Process.get(:css_blind_done, false) and
        (not is_function(ready_check, 0) or ready_check.())

    # The helper step, factored so the blind-fallback steering phase can
    # reuse it (defined before the cond; closes over this frame's vars).
    helper_drive = fn state ->
      helper =
        Melee.MenuHelper.step(state.menu_helper, gamestate, state.controller,
          port: state.controller_port,
          character: character,
          stage: stage,
          connect_code:
            case Map.get(state.config, :connect_code) do
              "" -> nil
              code -> code
            end,
          nametag: Map.get(state.config, :nametag),
          code_buffer: code_buffer,
          autostart: autostart,
          swag: false,
          stuck_after_frames: Map.get(state.config, :menu_stuck_frames, 1800),
          on_stuck: on_stuck
        )

      # Consume a suppressed-verdict flag set synchronously inside the
      # step above: rewind the stall episode so the detector re-fires
      # (and re-evaluates the hold) after another full stuck window.
      helper =
        if Process.get(:menu_stuck_suppressed, false) do
          Process.put(:menu_stuck_suppressed, false)
          %{helper | stuck_reported: false, stalled_frames: 0}
        else
          helper
        end

      %{state | menu_helper: helper}
    end

    # One-shot on warmup completion: a fully-steered counter rewinds to
    # the re-steer window (the loading animation orbited the cursor off
    # the portrait); a partial steer keeps its progress.
    if Process.get(:css_was_warming, false) and
         (not is_function(ready_check, 0) or ready_check.()) do
      Process.put(:css_was_warming, false)

      Process.put(
        :css_blind_n,
        ExPhil.Bridge.BlindCss.ready_resteer_reset(Process.get(:css_blind_n, 0))
      )
    end

    cond do
      # Warming + at a CSS (local 0 or slippi online 6): idle with a
      # visible LOADING ANIMATION, never confirm — the human sees
      # "loading" where the cursor is (Bradley's 08-22 UX request).
      # Candidates via EXPHIL_LOADING_ANIM (Bradley picks the keeper):
      #   wiggle   - left-right shimmy (default)
      #   infinity - small lemniscate (stick velocity x=cos t, y=cos 2t
      #              integrates to a figure-eight cursor path)
      #   circle   - small orbit
      #   nod      - vertical bob
      # Stick = cursor VELOCITY on the CSS, so each pattern is zero-mean
      # to keep the path closed; r sized above the deadzone.
      # Warming at the ONLINE CSS with steering still owed: run the
      # blind fallback's steer phase CONCURRENTLY with JIT warmup
      # (2026-08-22c overlap — same 480 frames of helper exposure,
      # ~8s earlier). Online only: the helper can't confirm there
      # pre-signal, so the warmup interlock holds; at the local CSS
      # the helper has real feedback and could confirm mid-JIT.
      is_function(ready_check, 0) and at_css? and not ready_check.() and online? and
          gamestate.menu_state == 6 and
          ExPhil.Bridge.BlindCss.warmup_step(Process.get(:css_blind_n, 0)) == :steer ->
        Process.put(:css_blind_n, Process.get(:css_blind_n, 0) + 1)
        Process.put(:css_was_warming, true)
        helper_drive.(state)

      is_function(ready_check, 0) and at_css? and not ready_check.() ->
        Process.put(:css_was_warming, true)
        # Piecewise square-wave segments, NOT sinusoids: the CSS cursor
        # response is nonlinear (deadzone + curve), so a trig pattern
        # leaves a DC residue and drifts (observed live: steady upward
        # crawl). Equal-duration opposite segments cancel exactly under
        # any monotone response.
        r = 0.18
        seg = rem(div(css_n, 18), 4)

        {x, y} =
          case System.get_env("EXPHIL_LOADING_ANIM", "wiggle") do
            # true two-lobed infinity: a diamond orbit traced left of
            # center, then one right of center (8 x 12-frame segments).
            # Built from pure-cardinal legs so per-axis response
            # asymmetry (which collapsed the diagonal bowtie into a "V")
            # cannot distort the shape, and sized down so the top legs
            # stay off the neighboring portrait frames.
            "infinity" ->
              case rem(div(css_n, 12), 8) do
                0 -> {0.5, 0.5 + r}
                1 -> {0.5 - r, 0.5}
                2 -> {0.5, 0.5 - r}
                3 -> {0.5 + r, 0.5}
                4 -> {0.5, 0.5 + r}
                5 -> {0.5 + r, 0.5}
                6 -> {0.5, 0.5 - r}
                7 -> {0.5 - r, 0.5}
              end

            # diamond orbit: N, E, S, W
            "circle" ->
              case seg do
                0 -> {0.5, 0.5 + r}
                1 -> {0.5 + r, 0.5}
                2 -> {0.5, 0.5 - r}
                3 -> {0.5 - r, 0.5}
              end

            "nod" ->
              {0.5, if(rem(div(css_n, 24), 2) == 0, do: 0.5 - r, else: 0.5 + r)}

            _ ->
              {if(rem(div(css_n, 24), 2) == 0, do: 0.5 - r, else: 0.5 + r), 0.5}
          end

        Melee.Controller.tilt_analog(state.controller, :main, x, y)
        state

      # BLIND CSS FALLBACK (EXPHIL_CSS_BLIND_FALLBACK=1, 2026-08-22):
      # this mainline-beta build streams NO online-CSS state (cursor/
      # character/coin bit-frozen across sessions), so the helper's
      # feedback steering can never confirm a pick. Observed live: the
      # actual cursor reliably parks on the target portrait — so after
      # warmup, press A open-loop, then pulse START, per the
      # ExPhil.Bridge.BlindCss decision table. Since 2026-08-22b the
      # loop is SEMI-CLOSED when a memory watcher runs: the RAM scene
      # word confirms the post-START departure (early handback) and
      # detects a pick that never landed (bounded A retry). With no
      # watcher every path is the validated open-loop timing.
      blind_fallback? and Process.get(:css_blind_n, 0) >= ExPhil.Bridge.BlindCss.a_press_at() ->
        n = Process.get(:css_blind_n, 0)
        Process.put(:css_blind_n, n + 1)

        # CENTER THE STICK first, every frame: the pipe latches the last
        # written state, so the helper's final steering tilt kept the
        # cursor moving under the presses (observed live: A-spam over a
        # drifting cursor, nothing selected).
        Melee.Controller.tilt_analog(state.controller, :main, 0.5, 0.5)

        progress = ExPhil.Bridge.BlindCss.classify(scene_word)
        retries = Process.get(:css_blind_retries, 0)

        # RAM selection state (css_p1_selected, 2026-08-22c): verified
        # at the OFFLINE CSS; whether the online CSS drives the same
        # array is the open question — the press-point log below is
        # the probe (science trace for the next Direct session).
        selection =
          ExPhil.Bridge.BlindCss.observe_selected(
            state.dolphin && state.dolphin.memory_watcher,
            1
          )

        if n == ExPhil.Bridge.BlindCss.a_press_at() do
          Logger.info(
            "[MeleePort] blind CSS press point: RAM selection reads #{inspect(selection)}" <>
              if(match?({:character, _}, selection),
                do: " — SKIPPING A press (already locked in)",
                else: ""
              )
          )
        end

        case ExPhil.Bridge.BlindCss.step(n, progress, retries, selection) do
          # ONE A press, held 3 frames, on the stationary cursor.
          # Exactly one: A over the selected portrait TOGGLES (observed
          # live — an even press count ended deselected).
          :press_a ->
            Melee.Controller.press_button(state.controller, :a)

          :release_a ->
            Melee.Controller.release_button(state.controller, :a)

          {:pulse_start, on?} ->
            Melee.Controller.release_button(state.controller, :a)

            if on?,
              do: Melee.Controller.press_button(state.controller, :start),
              else: Melee.Controller.release_button(state.controller, :start)

          # Scene evidence says the pick never landed: replay the A
          # press on the parked cursor (bounded; the odd press count is
          # preserved because a failed pick left the toggle unset).
          {:retry_a, r} ->
            Logger.warning(
              "[MeleePort] blind CSS: scene still at online CSS after START window " <>
                "(progress=#{progress} word=#{inspect(scene_word, base: :hex)}) — " <>
                "retrying A press (#{r})"
            )

            Process.put(:css_blind_retries, r)
            Process.put(:css_blind_n, ExPhil.Bridge.BlindCss.a_press_at())
            Melee.Controller.release_button(state.controller, :start)

          # HAND BACK to the helper permanently (via the done flag,
          # which un-matches blind_fallback? from the next frame):
          # menu_state 6 also covers the code/name-entry scene after
          # the CSS, and the helper has a real flow for it (Z-select
          # the autofilled code) — observed live 08-22: the fallback
          # kept pulsing START there ("goes to confirm and sits").
          :handback ->
            confirmed =
              if progress in [:departing, :elsewhere],
                do: " — departure CONFIRMED via RAM scene word",
                else: ""

            Logger.info(
              "[MeleePort] blind CSS: handing back to helper (progress=#{progress} " <>
                "word=#{inspect(scene_word, base: :hex)}, frame #{n})#{confirmed}"
            )

            Melee.Controller.release_button(state.controller, :start)
            Process.put(:css_blind_done, true)

          # n >= a_press_at here, so :steer is unreachable; be loud if
          # the table and the guard ever drift apart.
          :steer ->
            Logger.warning("[MeleePort] blind CSS: unexpected :steer at n=#{n}")
        end

        state

      blind_fallback? ->
        # count state-6 frames while the helper still drives (steering
        # phase of the blind fallback)
        Process.put(:css_blind_n, Process.get(:css_blind_n, 0) + 1)
        helper_drive.(state)

      true ->
        helper_drive.(state)
    end
  end

  # RAM ground truth for the stuck report — total over missing/dead
  # watchers (a diagnosis helper must never take the menu loop down).
  # The 250ms traffic window blocks the frame loop, which is fine
  # exactly here: on_stuck fires once per stall episode, 30s in.
  # Total watcher read: nil/dead watcher or unobserved name -> :unknown.
  defp safe_watch_get(watcher, name) do
    Melee.MemoryWatcher.get(watcher, name)
  catch
    :exit, _ -> :unknown
  end

  # Overlay watcher CSS observations onto a menu gamestate (pure merge
  # in Melee.MemoryMap.merge_css/3; the policy of WHEN and WHICH FIELDS
  # lives here). Validated 2026-08-23 live Direct session: the static
  # region (selection/hover/status/ready) reads correctly at the ONLINE
  # CSS (:none -> {:character, 2} across the pick), so the online merge
  # is DEFAULT ON for static fields. The CURSOR block is menu-HEAP and
  # only derived at the offline CSS — online it stays off until a
  # park-and-scan validates it there (EXPHIL_RAM_MENU=full opts in;
  # =0 disables the online merge entirely).
  defp ram_menu_merge(state, gamestate) do
    watcher = state.dolphin && state.dolphin.memory_watcher
    offline_css? = gamestate.menu_state == @menu_character_select
    online_css? = gamestate.menu_state == 6
    online_mode = System.get_env("EXPHIL_RAM_MENU", "static")

    fields =
      cond do
        watcher == nil -> nil
        offline_css? -> :all
        online_css? and online_mode == "full" -> :all
        online_css? and online_mode != "0" -> :static
        true -> nil
      end

    if fields do
      snapshot = Melee.MemoryWatcher.snapshot(watcher)

      unless Process.get(:ram_menu_merge_logged, false) do
        Process.put(:ram_menu_merge_logged, true)

        Logger.info(
          "[MeleePort] RAM menu merge active (fields=#{fields}, " <>
            "#{map_size(snapshot)} observed watches)"
        )
      end

      Melee.MemoryMap.merge_css(gamestate, snapshot, fields: fields)
    else
      gamestate
    end
  catch
    # A dead watcher must not take the menu loop down with it.
    :exit, _ -> gamestate
  end

  defp ram_menu_diagnosis(nil), do: %{ram_scene: :no_watcher, ram_traffic_delta: nil}

  defp ram_menu_diagnosis(watcher) do
    scene =
      case ExPhil.Bridge.BlindCss.observe(watcher) do
        :unknown -> :unknown
        word -> Melee.MemoryMap.scene_view(word)
      end

    delta =
      try do
        t0 = Melee.MemoryWatcher.traffic(watcher)
        Process.sleep(250)
        Melee.MemoryWatcher.traffic(watcher) - t0
      catch
        :exit, _ -> nil
      end

    %{ram_scene: scene, ram_traffic_delta: delta}
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

  # While the stage filter is LRAS-ing out of a wrong-stage game, the
  # agent's inputs are dropped — they would fight the quit chord.
  defp do_send_controller(%{force_quit: true} = state, _input), do: {:ok, state}

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

  # :require_stage accepts a name (atom/string, e.g. :final_destination /
  # "fd") or a bare EXTERNAL stage id.
  defp to_external_stage_id(nil), do: nil
  defp to_external_stage_id(v) when is_integer(v), do: v

  defp to_external_stage_id(v) do
    key = v |> to_string() |> String.downcase() |> String.to_atom()

    Map.get(@external_stage_ids, key) ||
      raise ArgumentError,
            "unknown require_stage #{inspect(v)} (known: #{inspect(Map.keys(@external_stage_ids))})"
  end

  defp to_stage_id(v) when is_integer(v), do: v
  defp to_stage_id(v) when is_atom(v), do: Melee.Enums.Stage.to_id(v)

  defp to_stage_id(v) when is_binary(v),
    do: v |> String.downcase() |> String.to_existing_atom() |> Melee.Enums.Stage.to_id()

  # Error struct kept for API compatibility with error-matching callers.
  @doc false
  def bridge_error(reason), do: BridgeError.new(reason, bridge: :melee_port)
end
