#!/usr/bin/env elixir
# Play against the trained agent in Dolphin via libmelee (ASYNC VERSION)
#
# This version uses separate processes for frame reading and inference,
# allowing the game loop to run at full speed even with slow models.
#
# Usage:
#   mix run scripts/play_dolphin_async.exs [options]
#
# Options:
#   --policy PATH       - Path to policy file (required)
#   --dolphin PATH      - Path to Slippi/Dolphin folder (required)
#   --iso PATH          - Path to Melee 1.02 ISO (required)
#   --port N            - Agent controller port (1-4, default: 1)
#   --opponent-port N   - Your controller port (1-4, default: 2)
#   --character NAME    - Agent character (default: mewtwo)
#   --stage NAME        - Stage (default: final_destination)
#   --frame-delay N     - Simulated online delay (default: 0)
#   --deterministic     - Use deterministic action selection

require Logger

alias ExPhil.CLI
alias ExPhil.Bridge.MeleePort
alias ExPhil.Bridge.AsyncRunner
alias ExPhil.Agents.Agent
alias ExPhil.Training.Output

# Parse command line arguments using CLI module
flag_groups = [:verbosity, :checkpoint, :replay, :dolphin, :common]

opts = CLI.parse_args(System.argv(),
  flags: flag_groups,
  defaults: [character: "mewtwo"]
)

# Setup verbosity early
CLI.setup_verbosity(opts)

# Convert character and stage to atoms if strings
opts = Keyword.update(opts, :character, :mewtwo, fn
  c when is_binary(c) -> String.to_atom(c)
  c -> c
end)

opts = Keyword.update(opts, :stage, :final_destination, fn
  s when is_binary(s) -> String.to_atom(s)
  s -> s
end)

# Convert on_game_end to atom
opts = Keyword.update(opts, :on_game_end, :restart, fn
  "restart" -> :restart
  "stop" -> :stop
  :restart -> :restart
  :stop -> :stop
  _ -> :restart
end)

# Handle help
CLI.maybe_show_help(opts, "play_dolphin_async.exs", flag_groups, fn ->
  IO.puts("""

  ASYNC VERSION - Separates frame reading from inference for smooth gameplay.

  EXAMPLES:
    mix run scripts/play_dolphin_async.exs \\
      --policy checkpoints/imitation_latest_policy.bin \\
      --dolphin ~/.local/share/Slippi\\ Launcher/netplay \\
      --iso ~/Games/SSBM.iso \\
      --character mewtwo \\
      --stage battlefield

  On game end modes:
    --on-game-end restart   Auto-start next game (default)
    --on-game-end stop      Exit after one game
  """)
end)

# Validate required args
CLI.require_options!(opts, [:policy, :dolphin, :iso])

Output.banner("ExPhil Dolphin Play (ASYNC)")

Output.config([
  {"Policy", opts[:policy]},
  {"Dolphin", opts[:dolphin]},
  {"ISO", opts[:iso]},
  {"af convention", if(opts[:live_af], do: "live -> parsed (GOTCHAS #81)", else: "parsed (no-op)")},
  {"Agent Port", opts[:port]},
  {"Your Port", opts[:opponent_port]},
  {"Character", opts[:character]},
  {"Stage", opts[:stage]},
  {"Frame Delay", opts[:frame_delay]},
  {"Deterministic", opts[:deterministic]},
  {"On Game End", opts[:on_game_end]},
  {"Architecture", "ASYNC (separate frame reader + inference processes)"}
])

# Step 1: Load the agent
Output.step(1, 5, "Loading agent")

{:ok, agent} =
  Agent.start_link(
    policy_path: opts[:policy],
    deterministic: opts[:deterministic],
    temperature: opts[:temperature] || 1.0,
    deterministic_buttons: opts[:deterministic_buttons] || false,
    press_threshold: opts[:press_threshold],
    release_threshold: opts[:release_threshold],
    jump_debounce: opts[:jump_debounce],
    frame_delay: opts[:frame_delay],
    ablate_prev_action: opts[:ablate_prev_action] || false,
    leace_eraser: opts[:leace_eraser],
    steer_vector: opts[:steer_vector],
    steer_alpha: opts[:steer_alpha] || 1.0,
    style_id: opts[:style_id],
    style_tag: opts[:style_tag],
    player_registry: opts[:player_registry],
    uncertainty_log: opts[:uncertainty_log],
    stateful_step: opts[:stateful_step] || false,
    af_convention: if(opts[:live_af], do: :live, else: :parsed)
  )

config = Agent.get_config(agent)
Output.success("Agent loaded")
Output.puts("    Temporal: #{config.temporal}")

if config.temporal do
  Output.puts("    Backbone: #{config.backbone}")
  Output.puts("    Window:   #{config.window_size} frames")

  if config[:stateful_step_active] do
    Output.puts("    Stateful: Edifice.Stateful step path (O(1)/frame)")
  end
end

# Step 2: Start the Melee bridge
Output.step(2, 5, "Starting Melee bridge")

{:ok, bridge} = MeleePort.start_link()
Output.success("Bridge process started")

# Step 3: Initialize Dolphin
Output.step(3, 5, "Initializing Dolphin")
Output.puts("  (This will launch Dolphin - make sure to plug in your controller!)")

# Elixir-driven dummies read game state and react (vs python's open-loop
# timing patterns); the bridge runs "external" so python doesn't fight them
elixir_dummies = %{"tech_random" => ExPhil.Agents.Dummies.TechRandom}
elixir_dummy = elixir_dummies[opts[:dummy]]

# --p2-policy (checkpoint ladder, task #19): a SECOND policy drives port 2
# through the same external-dummy hook, with sampling on (ladder wants
# game-to-game variance). Overrides --dummy.
elixir_dummy =
  if p2 = opts[:p2_policy] do
    {ExPhil.Agents.Dummies.PolicyOpponent,
     [
       policy_path: p2,
       press_threshold: opts[:press_threshold],
       release_threshold: opts[:release_threshold]
     ]}
  else
    elixir_dummy
  end

# Netplay (Slippi Direct): opponent is remote — no dummy of any kind
elixir_dummy = if opts[:connect_code], do: nil, else: elixir_dummy

if opts[:connect_code] do
  Output.puts("  Netplay: connecting to #{opts[:connect_code]} (dummies disabled)")
end

bridge_config = %{
  dolphin_path: opts[:dolphin],
  iso_path: opts[:iso],
  controller_port: opts[:port],
  opponent_port: opts[:opponent_port],
  character: opts[:character],
  stage: opts[:stage],
  online_delay: opts[:frame_delay],
  # Netplay account home (#9): EXPHIL_NETPLAY_HOME env (no CLI flag —
  # keeps this change out of compiled cli.ex). For the bot account:
  #   EXPHIL_NETPLAY_HOME=~/.config/SlippiOnline-bot
  user_home: System.get_env("EXPHIL_NETPLAY_HOME"),
  gfx_backend: System.get_env("EXPHIL_GFX"),
  # Port-2 dummy for drills (none|stand|shield|jump|walk|cpu|tech_random)
  dummy_mode:
    cond do
      opts[:connect_code] -> "none"
      elixir_dummy -> "external"
      true -> opts[:dummy] || "none"
    end,
  dummy_character: opts[:dummy_character] || "fox",
  dummy_cpu_level: opts[:dummy_cpu_level] || 0,
  no_audio: opts[:no_audio] || false,
  # Headless probes (task #5): Null gfx + no audio + blocking input. The
  # game paces to the policy loop instead of the video/audio throttle, so
  # every frame is seen even though emulation runs unthrottled (GOTCHA #56
  # is about the NON-blocking case).
  headless: opts[:headless] || false,
  replay_dir: opts[:replay_dir],
  slippi_port: opts[:slippi_port],
  # Slippi Direct netplay (see docs/planning/YETI_DEBUT.md); online_delay
  # rides the existing --frame-delay flag above
  connect_code: opts[:connect_code]
}

case MeleePort.init_console(bridge, bridge_config, 60_000) do
  {:ok, info} ->
    Output.success("Dolphin initialized and connected!")
    Output.puts("    Controller on port: #{info.controller_port}")

  :ok ->
    Output.success("Dolphin initialized and connected!")

  {:error, reason} ->
    Output.error("Failed to initialize Dolphin: #{inspect(reason)}")
    System.halt(1)
end

# Step 4: JIT Warmup
Output.step(4, 5, "JIT Warmup (this may take a minute for temporal models)")

case Agent.warmup(agent) do
  {:ok, warmup_ms} ->
    Output.success("JIT warmup complete (#{warmup_ms}ms)")

  {:error, reason} ->
    Output.warning("Warmup failed: #{inspect(reason)} (will warmup on first inference)")
end

# Step 5: Start async runner
Output.step(5, 5, "Starting async game runner")

{:ok, runner} =
  AsyncRunner.start_link(
    agent: agent,
    bridge: bridge,
    player_port: opts[:port],
    auto_menu: not opts[:no_auto_menu],
    on_game_end: opts[:on_game_end],
    dummy: elixir_dummy,
    # Headless (ExiAI) has no internal throttle: pace the frame loop to
    # 60Hz so input timing matches windowed play (--pace-hz to override;
    # 0 = unpaced). See AsyncRunner pace/1.
    pace_hz: opts[:pace_hz] || if(opts[:headless], do: 60, else: 0)
  )

Output.success("Async runner started")
Output.divider()
Output.section("ASYNC Game Loop Running!")
Output.puts("")
Output.puts("Frame reader and inference are running in separate processes.")
Output.puts("The game should respond smoothly even with slow LSTM models.")
Output.puts("")
Output.puts("Press Ctrl+C to stop.")
Output.puts("")

# Stats monitoring loop with enhanced FPS and confidence display
defmodule StatsMonitor do
  @target_fps 60

  def run(runner, interval_ms \\ 5000, on_game_end \\ :restart, max_seconds \\ nil) do
    Process.sleep(interval_ms)

    stats = ExPhil.Bridge.AsyncRunner.get_stats(runner)

    # With --on-game-end stop the frame loop exits after game 1, but nothing
    # stopped this monitor — the BEAM (and its multi-GB EXLA allocation)
    # lived on until killed by hand. Return so the script's cleanup runs.
    done? = on_game_end == :stop and stats.games_played >= 1

    # --seconds: return so the caller can SD the game to a clean end. Killing
    # the process instead leaves a TRUNCATED .slp that peppi rejects with
    # "failed to fill whole buffer" — Slippi only finalizes on game end, so a
    # timeout-kill throws the whole recording away.
    expired? = is_integer(max_seconds) and stats.elapsed_ms >= max_seconds * 1000

    if stats.elapsed_ms > 0 do
      elapsed_s = stats.elapsed_ms / 1000
      games_str = if stats.games_played > 0, do: " | Games: #{stats.games_played}", else: ""

      # FPS with target comparison and color
      fps = Float.round(stats.fps, 1)
      fps_color = fps_color_code(fps)
      fps_str = "#{fps_color}#{fps}/#{@target_fps} fps#{IO.ANSI.reset()}"

      # Confidence display
      conf_str = format_confidence(stats.latest_confidence, stats.avg_confidence)

      IO.puts(
        "[Stats] #{Float.round(elapsed_s, 1)}s | #{fps_str} | Inferences: #{stats.inferences_run}#{conf_str}#{games_str}"
      )
    end

    cond do
      done? ->
        IO.puts("[Stats] Game complete (on_game_end=stop) — shutting down")
        :ok

      expired? ->
        IO.puts("[Stats] --seconds reached — ending the game cleanly")
        :duration_reached

      true ->
        run(runner, interval_ms, on_game_end, max_seconds)
    end
  end

  # Color code based on FPS performance
  # Good (97%+ of target)
  defp fps_color_code(fps) when fps >= 58, do: IO.ANSI.green()
  # OK (83%+ of target)
  defp fps_color_code(fps) when fps >= 50, do: IO.ANSI.yellow()
  # Poor
  defp fps_color_code(_fps), do: IO.ANSI.red()

  defp format_confidence(nil, _avg), do: ""

  defp format_confidence(latest, avg) when is_map(latest) do
    overall = Map.get(latest, :overall, 0)
    avg_val = if is_number(avg), do: Float.round(avg, 2), else: 0

    # Color code confidence: green = high, yellow = medium, red = low
    conf_color = confidence_color(overall)
    " | #{conf_color}Conf: #{Float.round(overall, 2)} (avg: #{avg_val})#{IO.ANSI.reset()}"
  end

  defp format_confidence(_, _), do: ""

  defp confidence_color(conf) when conf >= 0.7, do: IO.ANSI.green()
  defp confidence_color(conf) when conf >= 0.4, do: IO.ANSI.yellow()
  defp confidence_color(_), do: IO.ANSI.red()
end

# Hold full left, no buttons: walk/fall off and don't recover, burning stocks
# until the game ENDS. Slippi only finalizes a .slp on game end — killing the
# process mid-game leaves a truncated file peppi rejects outright ("failed to
# fill whole buffer"), i.e. the entire session is unanalyzable. Same technique
# and input signature as record_multishine.exs, so the SD tail is filterable.
defmodule GracefulSD do
  @poll_ms 500
  @timeout_ms 120_000

  @doc """
  Wait for the frame loop's SD to reach a game end.

  The loop itself holds left (AsyncRunner.begin_sd/1); this only watches. ~2 min
  cap — a level-1 CPU will not save you from walking off, so exceeding it means
  something is wrong, not that it needs longer.
  """
  def await(runner, waited \\ 0) do
    cond do
      AsyncRunner.sd_complete?(runner) -> :ok
      waited >= @timeout_ms -> {:error, :sd_timeout}
      true ->
        Process.sleep(@poll_ms)
        await(runner, waited + @poll_ms)
    end
  end

  @hold_left %{
    main_stick: %{x: 0.0, y: 0.5},
    c_stick: %{x: 0.5, y: 0.5},
    shoulder: 0.0,
    buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
  }

  # L+R+A+Start: the instant match quit ("salty runback"). Ends the game in
  # ~1 frame with a proper Slippi game-end event — no walk-off, no deaths.
  # All four in ONE controller message (partial presses would pause instead;
  # LRAS also quits from the pause screen, so even that resolves). Start is
  # send-only in the bridge protocol (see melee_bridge.py button_map).
  @lras %{
    main_stick: %{x: 0.5, y: 0.5},
    c_stick: %{x: 0.5, y: 0.5},
    shoulder: 0.0,
    buttons: %{a: true, b: false, x: false, y: false, z: false, l: true, r: true, d_up: false, start: true}
  }

  # Frames of LRAS attempts before falling back to the hold-left walk-off.
  @lras_frames 120

  # MeleePort's default step timeout is 90 SECONDS. Stopping the AsyncRunner
  # can leave an in-flight request/response pair mid-transit, so the first
  # step() here blocks on a reply the dying reader already consumed. At 90s a
  # single desync eats most of the run's remaining budget and the process is
  # killed still SD-ing — which leaves a TRUNCATED .slp, losing the whole
  # recording. Observed 2026-07-26 on 2 of 6 runs: both hung with zero output
  # the instant SD began, while the successful runs finished SD in ~15s.
  #
  # So: a short per-step timeout, and treat a timeout as retryable rather than
  # fatal. A desync then costs ~2s, not 90.
  @step_timeout_ms 2_000
  @max_consecutive_timeouts 30

  @doc """
  Drain any in-flight bridge traffic after the runner stops.

  Cheap insurance: a few bounded steps let a half-delivered response clear
  before the SD loop starts counting on replies.
  """
  def drain(bridge, n \\ 5) do
    Enum.each(1..n, fn _ ->
      MeleePort.step(bridge, [auto_menu: false], @step_timeout_ms)
    end)
  catch
    :exit, _ -> :ok
  end

  # ~2 min safety cap; a level-1 CPU will not save you from walking off.
  def run(bridge, frames_left \\ 7200, timeouts \\ 0) do
    cond do
      frames_left <= 0 ->
        {:error, :sd_timeout}

      timeouts > @max_consecutive_timeouts ->
        {:error, {:bridge_unresponsive, timeouts}}

      true ->
        case step(bridge) do
          {:ok, game_state} ->
            frames_used = 7200 - frames_left
            input = if frames_used < @lras_frames, do: @lras, else: @hold_left

            # Observability (SD-flake diagnosis): does the input stream
            # actually land? If x isn't moving and stocks aren't dropping
            # during hold-left, the bridge stream is desynced (the
            # 2026-07-26 teardown failure mode), not the game refusing.
            if rem(frames_used, 60) == 0 do
              p = game_state.players[1]

              IO.puts(
                "[SD] f#{frames_used} phase=#{if(frames_used < @lras_frames, do: "LRAS", else: "hold-left")} " <>
                  "stocks=#{p && p.stock} x=#{p && Float.round(p.x * 1.0, 1)} action=#{p && p.action}"
              )
            end

            MeleePort.send_controller(bridge, input)
            run(bridge, frames_left - 1, 0)

          :timeout ->
            run(bridge, frames_left - 1, timeouts + 1)

          {:postgame, _} -> {:ok, :game_ended}
          {:menu, _} -> {:ok, :game_ended}
          {:game_ended, _} -> {:ok, :game_ended}
          {:error, reason} -> {:error, reason}
        end
    end
  end

  defp step(bridge) do
    MeleePort.step(bridge, [auto_menu: false], @step_timeout_ms)
  catch
    # GenServer.call raises on timeout; the bridge itself is usually fine, the
    # reply was just lost. Retry rather than abandon a nearly-complete run.
    :exit, {:timeout, _} -> :timeout
  end
end

# Run stats monitor
try do
  case StatsMonitor.run(runner, 5000, opts[:on_game_end], opts[:seconds]) do
    :duration_reached ->
      # SD runs INSIDE the frame loop. Stopping the runner first and driving
      # the bridge afterwards races the teardown — stop/1 exits the frame loop
      # process, which kills the Python bridge, so the caller ends up talking
      # to a corpse. That lost ~30% of recordings to truncated .slp files.
      Output.puts("SD-ing to end the game (Slippi finalizes the .slp on game end)...")
      AsyncRunner.begin_sd(runner)

      case GracefulSD.await(runner) do
        :ok -> Output.success("  Game end: replay finalized")
        {:error, reason} -> Output.error("  SD FAILED (#{inspect(reason)}) — .slp truncated")
      end

    _ ->
      :ok
  end
rescue
  e in RuntimeError ->
    Output.error("Error: #{Exception.message(e)}")
catch
  :exit, _ ->
    Output.puts("Exiting...")
end

# Cleanup
Output.puts("Cleaning up...")

case AsyncRunner.stop(runner) do
  {:ok, final_stats} ->
    Output.puts(
      "  Final stats: #{final_stats.frames} frames, #{final_stats.inferences} inferences, #{final_stats.games} games"
    )

    # Harness health: a stale send re-sent the previous action because no new
    # inference finished in between — for a 9-frame multishine cycle each one
    # is a timing slip the POLICY did not cause. High staleness = the machine,
    # not the model, degraded this run's score; see EXPOSURE_BIAS.md item 0a.
    stale = Map.get(final_stats, :stale_sends, 0)
    max_run = Map.get(final_stats, :max_stale_run, 0)
    frames = max(final_stats.frames, 1)
    pct = Float.round(stale * 100 / frames, 1)

    Output.puts(
      "  Staleness: #{stale}/#{final_stats.frames} sends stale (#{pct}%), longest stale run #{max_run} frames"
    )

  _ ->
    :ok
end

try do
  MeleePort.stop(bridge)
catch
  :exit, _ -> Output.puts("  (cleanup timed out, Dolphin may still be running)")
end

GenServer.stop(agent)

Output.divider()
Output.section("Session Complete!")
