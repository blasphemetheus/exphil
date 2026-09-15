#!/usr/bin/env elixir
# Scenario evaluation suite (task #18): virtual-savestate policy probes.
#
# For each manifest entry, boots a headless Dolphin, replays the RECORDED
# inputs of BOTH ports from game start up to the handoff frame (a pathology
# moment mined from a real probe game), verifies the live game still matches
# the replay (drift check), then hands port 1 to the policy and scores its
# response over a fixed window with ExPhil.Eval.ScenarioScore.
#
# Why input-prefix replay instead of Dolphin savestates: savestates are
# build-version-locked and can't be created headlessly; deterministic input
# replay (Mewtwo vs Fox on FD, no items/hazards) reaches the same state on
# any build, and the drift check catches the cases where it doesn't.
#
# P2 keeps replaying its recorded inputs after handoff: deterministic but
# NON-REACTIVE — it responds to the ghost of the original game, not to the
# policy. Scores measure the policy's first response to the situation, not
# a full interaction. See docs/guides/SCENARIOS.md.
#
# Usage:
#   mix run scripts/scenario_suite.exs --policy checkpoints/POLICY.bin [options]
#
# Options:
#   --policy PATH        Policy export (required)
#   --manifest PATH      Manifest JSON (default scenarios/manifest.json)
#   --types a,b          Only run these scenario types
#   --only N,M           Only run these entry indices (0-based, after type filter)
#   --runs N             Runs per entry (default 1; >1 only useful with --temperature)
#   --temperature T      Sampled runs at temperature T (default: deterministic)
#   --window N           Response window frames (default per type, 300)
#   --input-offset N     Recorded-input frame offset (default 1; see SCENARIOS.md)
#   --drift-tolerance F  Max |dx|/|dy| units at handoff (default 3.0)
#   --opponent-character NAME  Port-2 body (libmelee name). Default: the source
#                        replay's character (2026-09-14; was hard-coded fox).
#                        Override = a deliberately DIFFERENT body driven by the
#                        recorded inputs (drifts from the source; use with a
#                        frame-0 handoff to generate new input-driven games).
#   --dolphin PATH       Dolphin (default ~/.local/share/slippi/exi-ai/dolphin-emu-headless)
#   --direct-inputs      Atomic mixed protocol with byte inputs; requires the float build
#   --float-ports 1,2     Inject original processed inputs for these recorded ports
#                        (implies direct inputs; default build directory exi-ai-float).
#                        Audits every prefix frame before accepting a score.
#   --no-pipe-shim       Preserve analog triggers (default for direct/float mode)
#   --iso PATH           Melee ISO (default ~/isos/melee.iso)
#   --windowed           Disable headless (debugging; needs the netplay build)
#   --slippi-port N      Base slippi port (default 51480, +1 per run)
#   --out PATH           Scoreboard JSON (default logs/scenario_scores_<ts>.json)
#   --run-dir PATH       Base dir for per-run replay dirs (default logs/scenario_runs/<ts>)
#   --press-threshold F / --release-threshold F   Button hysteresis (probe recipe: 0.45/0.3)
#   --quiet / --verbose
#   --audit-teacher-labels Compare k-ahead labels with actual future teacher inputs
#   --no-orphan-sweep     Disable global cleanup; stop only this run's bridge
#   --prefix-history applied|committed|cold  Policy history convention (default applied)
#   --trace-policy-inputs Save issued/sent commands and observed action frames
#   --no-verify-input-timing Diagnostic only: disable recorded-input timing gate
#
# Probe-recipe example (r10, two types, deterministic):
#   EXLA_MEMORY_FRACTION=0.15 devenv shell -- bash -c \
#     'mix run scripts/scenario_suite.exs \
#        --policy checkpoints/mewtwo_combo_newera_r10_policy.bin \
#        --types tech_chase,opponent_behind'

require Logger

alias ExPhil.Agents.Agent
alias ExPhil.Bridge.MeleePort
alias ExPhil.Data.Peppi
alias ExPhil.Eval.{ScenarioScan, ScenarioScore}
alias ExPhil.Training.Output

defmodule ScenarioSuite do
  @moduledoc false

  @neutral %{
    main_stick: %{x: 0.5, y: 0.5},
    c_stick: %{x: 0.5, y: 0.5},
    shoulder: 0.0,
    buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false}
  }

  # --finalize suicide input: hold the main stick full-left (x=0.0) with
  # no jump/up-B, so the character walks off the FD edge and drifts into
  # the blast zone within a couple seconds — forcing GAME!. Slippi writes
  # the .slp's raw-data length ONLY at game-end (#73), so without this a
  # scenario stopped at handoff+window leaves an unparseable file. NOTE:
  # UNVERIFIED on a live run (written while r16 held the GPU, 2026-07-22)
  # — confirm Mewtwo actually SDs from center on FD with this input; if a
  # float/DJ saves it, add a down-hold or a walk-off from a known x.
  @suicide %{@neutral | main_stick: %{x: 0.0, y: 0.5}}

  # ~3 min of menu frames before declaring the boot stuck
  @menu_step_limit 12_000

  # --finalize: cap the self-destruct drive so a game that somehow won't
  # end can't hang the suite (unthrottled, so 1200 frames = seconds).
  @finalize_step_limit 1_200

  # ==========================================================================
  # Replay preparation (once per source .slp)
  # ==========================================================================

  def prepare_replay(slp, opts \\ []) do
    # pipe_shim: convert recorded analog trigger holds to digital presses.
    # Needed ONLY for the pipe input path on the ExiAI build (GOTCHAS #66);
    # with EXI inputs (bridge default when headless) analog triggers
    # round-trip and the shim must be OFF or light shields replay wrong.
    shim = Keyword.get(opts, :pipe_shim, false)
    float_ports = Keyword.get(opts, :float_ports, [])
    {:ok, replay} = Peppi.parse(Path.expand(slp))
    if float_ports != [] and replay.metadata.stage != 32,
      do: raise("float replay currently requires a Final Destination source (the suite pins this stage)")
    if float_ports != [] and Enum.any?(replay.frames, fn frame ->
      Enum.any?(float_ports, fn port -> frame.players[port] && frame.players[port].character in [10, 11] end)
    end), do: raise("exact follower input replay is not implemented")

    {inputs, ref} =
      Enum.reduce(replay.frames, {%{}, %{}}, fn f, {inputs, ref} ->
        p1 = f.players[1]
        p2 = f.players[2]

        if p1 && p2 do
          {
            Map.put(
              inputs,
              f.frame_number,
              {replay_input(p1.controller, shim, 1 in float_ports), replay_input(p2.controller, shim, 2 in float_ports)}
            ),
            Map.put(ref, f.frame_number, %{
              p1: ScenarioScan.player_summary(p1),
              p2: ScenarioScan.player_summary(p2)
            })
          }
        else
          {inputs, ref}
        end
      end)

    first = Enum.find(replay.frames, fn f -> f.players[1] && f.players[2] end)
    opponent = first && first.players[2].character && libmelee_character(trunc(first.players[2].character))
    %{inputs: inputs, ref: ref, opponent_character: opponent, source_character: first && libmelee_character(trunc(first.players[1].character))}
  end

  # Internal (in-game) character id -> libmelee Character name, for the
  # bridge's dummy_character. The suite used to hard-code "fox" for port 2:
  # every non-Fox source replay then drifted from frame -38 (the wrong body
  # in the entry animation) and no held-out opponent handoff could qualify
  # (2026-09-14 coverage round, 70/77 diverged).
  @libmelee_characters %{
    0 => "mario", 1 => "fox", 2 => "cptfalcon", 3 => "dk", 4 => "kirby", 5 => "bowser",
    6 => "link", 7 => "sheik", 8 => "ness", 9 => "peach", 10 => "popo", 11 => "nana",
    12 => "pikachu", 13 => "samus", 14 => "yoshi", 15 => "jigglypuff", 16 => "mewtwo",
    17 => "luigi", 18 => "marth", 19 => "zelda", 20 => "ylink", 21 => "doc", 22 => "falco",
    23 => "pichu", 24 => "gameandwatch", 25 => "ganondorf", 26 => "roy"
  }
  def libmelee_character(id), do: Map.get(@libmelee_characters, id) || raise("unknown internal character id #{id}")

  # Recorded Peppi controller -> bridge input map. Peppi already normalizes
  # sticks to the bridge's 0..1 range ((raw+1)/2 in the NIF), and stick
  # values round-trip exactly through libmelee's tilt_analog (verified:
  # TechRandom's 0.35/0.65 walk tilts come back as exactly those values).
  #
  # Triggers do NOT round-trip on the ExiAI headless build: analog "SET L"
  # pipe commands are silently ignored there (GOTCHAS #66) while digital
  # PRESS L works — so recorded analog trigger holds (TechRandom shields)
  # are converted to digital presses past Melee's analog-shield threshold
  # (raw 43/140 ~ 0.31). Cost: light shields replay as full shields (none
  # in probe games). The analog value is still sent for builds that honor it.
  defp replay_input(c, shim, false), do: rec_input(c, shim)
  defp replay_input(c, _shim, true) do
    unless is_map(c.processed), do: raise("rebuild the Peppi NIF: original processed inputs are missing")
    processed = c.processed |> Map.from_struct() |> Map.merge(%{l_trigger: c.l_trigger, r_trigger: c.r_trigger})
    Melee.SlippiPad.pack_processed(processed)
    Map.put(rec_input(c, false), :processed_input, processed)
  end

  @trigger_digital_threshold 0.31

  defp rec_input(c, pipe_shim) do
    %{
      main_stick: %{x: c.main_stick_x, y: c.main_stick_y},
      c_stick: %{x: c.c_stick_x, y: c.c_stick_y},
      shoulder: max(c.l_trigger, c.r_trigger),
      buttons: %{
        a: c.button_a,
        b: c.button_b,
        x: c.button_x,
        y: c.button_y,
        z: c.button_z,
        l: c.button_l or (pipe_shim and c.l_trigger > @trigger_digital_threshold),
        r: c.button_r or (pipe_shim and c.r_trigger > @trigger_digital_threshold),
        d_up: c.button_d_up
      }
    }
  end

  # Live ControllerState (Agent output) -> bridge input map (AsyncRunner shape)
  defp controller_to_input(%ExPhil.Bridge.ControllerState{} = cs) do
    %{
      main_stick: %{x: cs.main_stick.x, y: cs.main_stick.y},
      c_stick: %{x: cs.c_stick.x, y: cs.c_stick.y},
      shoulder: cs.l_shoulder + cs.r_shoulder,
      buttons: %{
        a: cs.button_a,
        b: cs.button_b,
        x: cs.button_x,
        y: cs.button_y,
        z: cs.button_z,
        l: cs.button_l,
        r: cs.button_r,
        d_up: cs.button_d_up
      }
    }
  end

  # ==========================================================================
  # One scenario run
  # ==========================================================================

  def run_one(entry, prep, agent, run_idx, seq, opts) do
    t0 = System.monotonic_time(:millisecond)
    type = entry.type
    window = opts[:window] || ScenarioScore.window(type)

    run_dir = Path.join(opts[:run_base], "run#{pad(seq)}_#{type}")
    File.mkdir_p!(run_dir)

    {:ok, bridge} = MeleePort.start_link()

    config = %{
      dolphin_path: opts[:dolphin],
      direct_inputs: opts[:direct_inputs] || false,
      processed_inputs: opts[:direct_inputs] || false,
      iso_path: opts[:iso],
      controller_port: 1,
      opponent_port: 2,
      character: String.to_atom(opts[:character] || if(opts[:float_ports] not in [nil, []], do: prep.source_character, else: "mewtwo")),
      stage: :final_destination,
      online_delay: 0,
      dummy_mode: "external",
      # port 2 = the source replay's character unless --opponent-character overrides
      dummy_character: opts[:opponent_character] || prep.opponent_character || "fox",
      dummy_cpu_level: 0,
      no_audio: true,
      headless: not opts[:windowed],
      # Unthrottled is SAFE here (and 6-7x faster): prefix inputs are
      # frame-indexed and blocking-paced, timing-exact at any speed —
      # unlike policy-driven probes, which need real-time (see bridge).
      emulation_speed: 0.0,
      console_timeout: opts[:console_timeout],
      replay_dir: run_dir,
      slippi_port: opts[:slippi_port] + seq
    }

    if agent, do: Agent.reset_buffer(agent)

    result =
      case MeleePort.init_console(bridge, config, 180_000) do
        {:ok, _} -> game_loop(bridge, agent, entry, prep, window, opts)
        :ok -> game_loop(bridge, agent, entry, prep, window, opts)
        {:error, reason} -> %{error: "init failed: #{inspect(reason)}"}
      end

    try do
      MeleePort.stop(bridge)
    catch
      :exit, _ -> :ok
    end

    if Process.alive?(bridge), do: GenServer.stop(bridge, :normal, 5_000)

    result =
      if opts[:verify_input_timing] and opts[:driver] == :policy and is_nil(result[:error]) do
        timing = ExPhil.Eval.ScenarioInputTiming.verify_directory(
          run_dir, result[:policy_inputs] || [], opts[:response_delay]
        )

        result = result |> Map.put(:input_timing, timing) |> Map.put(:timing_valid, timing.valid)
        if timing.valid,
          do: result,
          else: result |> Map.put(:unvalidated_score, result[:score])
            |> Map.put(:score, nil) |> Map.put(:pass, false)
      else
        result
      end

    result =
      if opts[:float_ports] not in [nil, []] and is_nil(result[:error]) do
        audit = ExPhil.Eval.ReplayPrefixAudit.verify_directory(
          Path.expand(entry.slp), run_dir, -39, entry.frame - 1, opts[:float_ports]
        )
        File.write!(Path.join(run_dir, "prefix_audit.json"), Jason.encode!(audit, pretty: true))
        result = Map.put(result, :prefix_audit, audit)
        if audit.valid, do: result,
          else: result |> Map.put(:unvalidated_score, result[:score]) |> Map.put(:score, nil)
                       |> Map.put(:pass, false) |> Map.put(:diverged, true)
      else
        result
      end

    wall_s = (System.monotonic_time(:millisecond) - t0) / 1000

    Map.merge(
      %{
        type: type,
        slp: entry.slp,
        frame: entry.frame,
        note: entry.note,
        run: run_idx,
        window: window,
        opponent_character: config.dummy_character,
        float_ports: opts[:float_ports],
        input_transport: if(config.direct_inputs, do: "direct", else: "pipe"),
        wall_s: Float.round(wall_s, 1),
        replay_dir: run_dir
      },
      result
    )
  rescue
    e ->
      %{
        type: entry.type,
        slp: entry.slp,
        frame: entry.frame,
        run: run_idx,
        error: Exception.message(e)
      }
  end

  # ---- phase state machine --------------------------------------------------

  defp game_loop(bridge, agent, entry, prep, window, opts) do
    st = %{
      phase: :menu,
      menu_steps: 0,
      scenario_type: entry.type,
      handoff: entry.frame,
      window: window,
      offset: opts[:input_offset],
      tolerance: opts[:drift_tolerance],
      inputs: prep.inputs,
      ref: prep.ref,
      agent: agent,
      drift: nil,
      diverged: false,
      handoff_snapshot: nil,
      obs: [],
      truncated: nil,
      # --finalize (#38/#73): after the response window, drive the game to
      # a natural end so Slippi finalizes the .slp and it can be reused as
      # a training rollout. Off by default (eval only needs the window).
      finalize: opts[:finalize] || false,
      # --driver plumbing (2026-09-12)
      driver: opts[:driver] || :policy,
      expert: opts[:expert],
      prev_controller: nil,
      audit_teacher_labels: opts[:audit_teacher_labels] || false,
      teacher_samples: [],
      prefix_history: opts[:prefix_history] || "applied",
      opponent_character: opts[:opponent_character] || prep.opponent_character || "fox",
      response_opponent: opts[:response_opponent] || "replay",
      trace_policy_inputs: opts[:trace_policy_inputs] ||
        (opts[:verify_input_timing] and opts[:driver] == :policy),
      policy_inputs: [],
      # --response-delay: decisions in flight (oldest first)
      pending: [],
      response_delay: opts[:response_delay] || 0,
      finalize_steps: 0,
      # Prefix drift diagnostics: sampled trace + first frame past thresholds
      trace_all: opts[:trace_all] || false,
      drift_trace: [],
      first_drift: %{}
    }

    loop(bridge, st)
  end

  defp loop(bridge, st) do
    case MeleePort.step(bridge, auto_menu: true) do
      {:menu, _gs} ->
        case st.phase do
          :menu ->
            if st.menu_steps > @menu_step_limit do
              %{error: "stuck in menus after #{st.menu_steps} steps"}
            else
              loop(bridge, %{st | menu_steps: st.menu_steps + 1})
            end

          :finalize ->
            finish(%{st | truncated: :finalized})

          :respond ->
            finish(%{st | truncated: :game_ended})

          :prefix ->
            %{error: "game ended during prefix (catastrophic divergence?)"}
        end

      {:postgame, _gs} ->
        case st.phase do
          :finalize -> finish(%{st | truncated: :finalized})
          :respond -> finish(%{st | truncated: :game_ended})
          :menu -> loop(bridge, %{st | menu_steps: st.menu_steps + 1})
          :prefix -> %{error: "game ended during prefix (catastrophic divergence?)"}
        end

      {:ok, gs} ->
        st = if st.phase == :menu, do: %{st | phase: :prefix}, else: st

        case st.phase do
          :prefix -> prefix_frame(bridge, gs, st)
          :respond -> respond_frame(bridge, gs, st)
          :finalize -> finalize_frame(bridge, gs, st)
        end

      {:game_ended, reason} ->
        cond do
          # --finalize succeeded: the SD reached GAME!, so the .slp is now
          # finalized/parseable — the goal, not an error.
          st.phase == :finalize -> finish(%{st | truncated: :finalized})
          st.phase == :respond -> finish(%{st | truncated: :game_ended})
          true -> %{error: "dolphin ended: #{reason}"}
        end

      {:error, reason} ->
        %{error: "step error: #{inspect(reason)}"}
    end
  end

  defp prefix_frame(bridge, gs, st) do
    f = gs.frame

    st = track_drift(st, gs, f)

    if f >= st.handoff do
      st = ExPhil.Eval.ScenarioHistory.prepare_handoff(st)
      # Handoff: verify the live game still matches the source replay.
      drift = drift_check(gs, st.ref[f], st.tolerance)
      snapshot = snapshot(gs)

      st = %{
        st
        | phase: :respond,
          drift: drift,
          diverged: drift.diverged,
          handoff_snapshot: snapshot
      }

      respond_frame(bridge, gs, st)
    else
      # Observe-only warm-up (2026-09-12): the driver sees the prefix as
      # HISTORY. The policy agent embeds every recorded frame and takes the
      # recorded p1 input as its own (prev-action slot + queue ring), so the
      # handoff starts with a saturated window and a truthful own-input
      # queue instead of the cold start no policy can chain from
      # (RESULTS.md §6 control). The teacher's `prev` is warmed the same
      # way, so its recovery taps alternate off the real last input.
      st =
        case st.inputs[f + st.offset] do
          {p1, p2} ->
            MeleePort.send_controller(bridge, p1)
            MeleePort.send_controller(bridge, Map.put(p2, :port, 2))
            observe_prefix(gs, st, ExPhil.Bridge.ControllerState.from_input(p1))

          nil ->
            MeleePort.send_controller(bridge, @neutral)
            MeleePort.send_controller(bridge, Map.put(@neutral, :port, 2))
            observe_prefix(gs, st, ExPhil.Bridge.ControllerState.from_input(@neutral))
        end

      loop(bridge, st)
    end
  end

  defp observe_prefix(_gs, %{driver: :policy, prefix_history: "cold"} = st, _controller),
    do: st

  defp observe_prefix(gs, %{driver: :policy, agent: agent} = st, controller) when agent != nil do
    controller =
      if st.prefix_history == "committed" do
        st.inputs
        |> ExPhil.Eval.ScenarioHistory.committed_input(gs.frame, st.offset, st.response_delay)
        |> ExPhil.Bridge.ControllerState.from_input()
      else
        controller
      end

    case Agent.observe(agent, gs, controller, player_port: 1) do
      :ok -> %{st | prev_controller: controller}
      {:error, reason} ->
        Logger.warning("[scenario] agent observe error at f=#{gs.frame}: #{inspect(reason)}")
        st
    end
  end

  defp observe_prefix(_gs, st, controller), do: %{st | prev_controller: controller}

  # The source replay's p1 input for live frame f (offset-adjusted), or
  # neutral past the recording.
  defp recorded_p1(st, f) do
    case st.inputs[f + st.offset] do
      {p1, _p2} -> p1
      nil -> @neutral
    end
  end

  defp respond_frame(bridge, gs, st) do
    f = gs.frame

    st =
      if f > st.handoff and gs.players[1] != nil and gs.players[2] != nil do
        obs = %{
          frame: f,
          p1: ScenarioScan.player_summary(gs.players[1]),
          p2: ScenarioScan.player_summary(gs.players[2])
        }

        %{st | obs: [obs | st.obs]}
      else
        st
      end

    if f >= st.handoff + st.window do
      # Window over. --finalize: play on to a natural game end so the
      # .slp finalizes (#38/#73); otherwise stop here (eval is done — the
      # score comes from st.obs collected during the window).
      if st.finalize do
        finalize_frame(bridge, gs, %{st | phase: :finalize})
      else
        finish(st)
      end
    else
      # Port 1: the driver takes over (policy | teacher | neutral).
      st =
        case st.driver do
          :policy ->
            case Agent.get_controller(st.agent, gs, player_port: 1) do
              {:ok, controller} ->
                # --response-delay N (2026-09-12): hold each decision N extra
                # frames so decision->application latency is 1 + N. The
                # drill trains labels at delay-id d + pipeline offset 2, so
                # this harness's native latency 1 is a rung NO policy was
                # trained at; N = 1/2/3 lands on ids 0/1/2. While a decision
                # is in flight the RECORDED input keeps playing (the
                # already-committed pipeline), exactly as at handoff.
                pending = st.pending ++ [controller]

                {to_send, pending} =
                  if length(pending) > st.response_delay do
                    [head | rest] = pending
                    {controller_to_input(head), rest}
                  else
                    {recorded_p1(st, f), pending}
                  end

                MeleePort.send_controller(bridge, to_send)
                trace =
                  if st.trace_policy_inputs do
                    [
                      %{
                        frame: f,
                        issued: controller_to_input(controller),
                        sent: to_send,
                        action: gs.players[1].action,
                        action_frame: gs.players[1].action_frame,
                        on_ground: gs.players[1].on_ground
                      } | st.policy_inputs
                    ]
                  else
                    st.policy_inputs
                  end

                %{st | prev_controller: controller, pending: pending, policy_inputs: trace}

              {:error, reason} ->
                Logger.warning("[scenario] agent error at f=#{f}: #{inspect(reason)}")
                MeleePort.send_controller(bridge, @neutral)
                st
            end

          :teacher ->
            # The expert labels the LIVE state (same call the relabel makes on
            # rollout frames), with the input it issued last frame as `prev`.
            # GOTCHA #81: the table is keyed in PARSED action_frame numbering
            # (Slippi), the bridge reports LIVE numbering (one higher on most
            # actions) — convert first, or the last-jumpsquat press lands a
            # frame late and every re-entry floats (first run, 2026-09-12).
            p1 =
              case gs.players[1] do
                nil ->
                  nil

                p ->
                  %{p | action_frame: ExPhil.Data.ActionFrameConvention.libmelee_to_parsed(p.character, p.action, p.action_frame)}
              end

            case p1 && ExPhil.Agents.MultishineExpert.label(st.expert, p1, st.prev_controller) do
              {:ok, controller} ->
                MeleePort.send_controller(bridge, controller_to_input(controller))
                samples =
                  if st.audit_teacher_labels do
                    [
                      ExPhil.Eval.RecoveryLabelAudit.sample(
                        st.expert, p1, st.prev_controller, controller, f
                      ) | st.teacher_samples
                    ]
                  else
                    st.teacher_samples
                  end

                %{st | prev_controller: controller, teacher_samples: samples}

              _ ->
                MeleePort.send_controller(bridge, @neutral)
                st
            end

          :neutral ->
            MeleePort.send_controller(bridge, @neutral)
            st
        end

      p2 =
        ExPhil.Eval.ScenarioOpponent.input(st.response_opponent, st.inputs[f + st.offset], @neutral)

      MeleePort.send_controller(bridge, Map.put(p2, :port, 2))

      loop(bridge, st)
    end
  end

  # --finalize phase: after the scored window, drive P1 off-stage (P2
  # neutral) to force GAME! so Slippi closes the .slp. The frames added
  # here are junk (a walk-off SD), but they sit AFTER the useful prefix +
  # response and the point is only to make the file parseable as a
  # training rollout; the expert-relabel + downstream min-frame/empty
  # filters tolerate the short SD tail. Capped by @finalize_step_limit.
  defp finalize_frame(bridge, gs, st) do
    _ = gs

    if st.finalize_steps >= @finalize_step_limit do
      # Never reached GAME! (recovered? off-stage-immune?) — stop anyway;
      # the .slp stays unfinalized, flagged so a caller can drop it.
      finish(%{st | truncated: :finalize_timeout})
    else
      MeleePort.send_controller(bridge, @suicide)
      MeleePort.send_controller(bridge, Map.put(@neutral, :port, 2))
      loop(bridge, %{st | finalize_steps: st.finalize_steps + 1})
    end
  end

  defp finish(st) do
    window = Enum.reverse(st.obs)

    scored =
      if st.handoff_snapshot && window != [] do
        ExPhil.Eval.ScenarioScore.score(st.scenario_type, st.handoff_snapshot, window)
      else
        %{score: 0.0, pass: false, details: %{note: :no_observations}}
      end

    %{
      drift: st.drift,
      diverged: st.diverged,
      truncated: st.truncated,
      frames_observed: length(window),
      score: scored.score,
      pass: scored.pass,
      details: scored.details,
      # What the policy actually did, compressed: [[action, run_length], ...]
      p1_actions_rle: rle(Enum.map(window, & &1.p1.action), 30),
      first_drift: st.first_drift,
      drift_trace: Enum.reverse(st.drift_trace)
    }
    |> then(fn result ->
      if st.trace_policy_inputs,
        do: Map.put(result, :policy_inputs, Enum.reverse(st.policy_inputs)),
        else: result
    end)
    |> then(fn result ->
      if st.audit_teacher_labels do
        Map.put(
          result,
          :teacher_label_audit,
          ExPhil.Eval.RecoveryLabelAudit.report(Enum.reverse(st.teacher_samples))
        )
      else
        result
      end
    end)
  end

  defp rle(list, max_runs) do
    list
    |> Enum.chunk_by(& &1)
    |> Enum.take(max_runs)
    |> Enum.map(fn [a | _] = run -> [a, length(run)] end)
  end

  # Prefix diagnostics: record the first frame each port exceeds small drift
  # thresholds and a sampled trace every 120 frames — this localizes WHERE a
  # divergence starts (frame-alignment bugs show up at frame ~0; event
  # nondeterminism shows up at a specific hit/tech).
  defp track_drift(st, gs, f) do
    case st.ref[f] do
      nil ->
        st

      ref ->
        live = snapshot(gs)
        d1 = live.p1.x - ref.p1.x
        d2 = live.p2.x - ref.p2.x

        st =
          Enum.reduce([{:p1, d1}, {:p2, d2}], st, fn {port, d}, acc ->
            key = "#{port}_gt_0.5"

            if abs(d) > 0.5 and not Map.has_key?(acc.first_drift, key) do
              %{acc | first_drift: Map.put(acc.first_drift, key, %{frame: f, dx: Float.round(d, 2)})}
            else
              acc
            end
          end)

        if st.trace_all or rem(f, 120) == 0 do
          entry = %{
            f: f,
            p1_dx: Float.round(d1, 2),
            p2_dx: Float.round(d2, 2),
            p1_a: [live.p1.action, ref.p1.action],
            p2_a: [live.p2.action, ref.p2.action],
            p1_pct: Float.round(live.p1.percent - ref.p1.percent, 1),
            p2_pct: Float.round(live.p2.percent - ref.p2.percent, 1)
          }

          %{st | drift_trace: [entry | st.drift_trace]}
        else
          st
        end
    end
  end

  defp snapshot(gs) do
    %{
      p1: ScenarioScan.player_summary(gs.players[1]),
      p2: ScenarioScan.player_summary(gs.players[2])
    }
  end

  defp drift_check(_gs, nil, _tol), do: %{diverged: true, reason: :no_reference_frame}

  defp drift_check(gs, ref, tol) do
    live = snapshot(gs)

    per_port =
      Map.new([:p1, :p2], fn port ->
        l = live[port]
        r = ref[port]

        {port,
         %{
           dx: Float.round(l.x - r.x, 2),
           dy: Float.round(l.y - r.y, 2),
           action_live: l.action,
           action_ref: r.action
         }}
      end)

    diverged =
      Enum.any?(per_port, fn {_p, d} ->
        abs(d.dx) > tol or abs(d.dy) > tol or
          not actions_equivalent?(d.action_live, d.action_ref)
      end)

    Map.put(per_port, :diverged, diverged)
  end

  # The analog->digital trigger conversion (see rec_input/1) cannot
  # reproduce shield SUBTYPE exactly (a digital press can powershield where
  # the source's analog ramp plain-shielded) — treat the shield family as
  # one state for drift purposes. Positions still must match.
  @shield_family MapSet.new(178..182)

  defp actions_equivalent?(a, a), do: true

  defp actions_equivalent?(a, b),
    do: MapSet.member?(@shield_family, a) and MapSet.member?(@shield_family, b)

  defp pad(n), do: String.pad_leading(to_string(n), 2, "0")
end

# ============================================================================
# CLI
# ============================================================================

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      character: :string,
      opponent_character: :string,
      driver: :string,
      fixture: :string,
      delay_id: :integer,
      live_af: :boolean,
      manifest: :string,
      types: :string,
      only: :string,
      runs: :integer,
      temperature: :float,
      window: :integer,
      input_offset: :integer,
      response_delay: :integer,
      prefix_history: :string,
      response_opponent: :string,
      trace_policy_inputs: :boolean,
      verify_input_timing: :boolean,
      reaction_delay: :integer,
      console_timeout: :float,
      drift_tolerance: :float,
      dolphin: :string,
      float_ports: :string,
      direct_inputs: :boolean,
      iso: :string,
      windowed: :boolean,
      trace_all: :boolean,
      audit_teacher_labels: :boolean,
      orphan_sweep: :boolean,
      slippi_port: :integer,
      out: :string,
      run_dir: :string,
      press_threshold: :float,
      release_threshold: :float,
      pipe_shim: :boolean,
      finalize: :boolean,
      quiet: :boolean,
      verbose: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)
float_ports = case opts[:float_ports] do
  nil -> []
  value -> value |> String.split(",") |> Enum.map(&String.to_integer/1) |> Enum.uniq() |> Enum.sort()
end
unless Enum.all?(float_ports, &(&1 in [1, 2])), do: raise("--float-ports must contain only 1 and/or 2")
direct_inputs = opts[:direct_inputs] || float_ports != []
if direct_inputs do
  ExPhil.Eval.FloatInputBuild.verify!(Path.expand(opts[:dolphin] || "~/.local/share/slippi/exi-ai-float/dolphin-emu-headless"))
end
opts = Keyword.put_new(opts, :pipe_shim, not direct_inputs)
if float_ports != [] and opts[:pipe_shim], do: raise("--float-ports requires --no-pipe-shim")

unless opts[:policy] != nil or (opts[:driver] || "policy") != "policy" do
  Output.error("--policy is required (unless --driver teacher|neutral)")
  System.halt(1)
end

ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")

# INVARIANTS item 12 (GOTCHA #115): this harness applies a decision
# --response-delay + 1 frames after the observed state. The rung is resolved
# ONCE here through ExPhil.Eval.HarnessRung: an explicit --response-delay is
# used as given (the Agent derives the delay-id from it, or checks an explicit
# --delay-id against it); otherwise the checkpoint's own smallest trained id
# (or --delay-id) picks its aligned response delay, so the suite can no
# longer be run one frame faster than the policy was trained by default.
{response_delay, reaction_delay} =
  cond do
    (opts[:driver] || "policy") != "policy" or opts[:policy] == nil ->
      rd = opts[:response_delay] || opts[:reaction_delay] || 0
      {rd, rd}

    true ->
      {:ok, %{config: cfg}} = ExPhil.Training.Checkpoint.load_policy(opts[:policy])

      # ONE knob: --reaction-delay k (== --response-delay k here, the suite's
      # pipeline is exactly 1 frame); --delay-id N picks that id's trained
      # rung; default = the checkpoint's smallest trained reaction delay.
      resolve_opts =
        cond do
          opts[:reaction_delay] != nil -> [reaction_delay: opts[:reaction_delay]]
          opts[:response_delay] != nil -> [reaction_delay: opts[:response_delay]]
          opts[:delay_id] != nil -> [reaction_delay: opts[:delay_id] |> then(fn id ->
            {offset, _} = ExPhil.Eval.HarnessRung.delay_id_reaction_offset(cfg)
            ExPhil.Data.LabelConvention.to_reaction(id, ExPhil.Data.LabelConvention.of(cfg)) + offset
          end)]
          true -> []
        end

      case ExPhil.Eval.HarnessRung.resolve(:scenario_suite, resolve_opts, cfg) do
        {:ok, r} ->
          Output.puts("reaction delay #{r.reaction_delay} (latency #{r.expected_latency}; --response-delay #{r.knob}; from #{r.source})")
          {r.knob, r.reaction_delay}

        {:error, msg} ->
          Output.error(msg)
          System.halt(1)
      end
  end

unless (opts[:response_opponent] || "replay") in ["replay", "neutral"],
  do: raise(ArgumentError, "--response-opponent must be replay or neutral")

suite_opts = [
  float_ports: float_ports,
  direct_inputs: direct_inputs,
  response_opponent: opts[:response_opponent] || "replay",
  verify_input_timing: Keyword.get(opts, :verify_input_timing, true),
  console_timeout: opts[:console_timeout],
  trace_policy_inputs: opts[:trace_policy_inputs] || false,
  prefix_history: opts[:prefix_history] || "applied",
  audit_teacher_labels: opts[:audit_teacher_labels] || false,
  dolphin: Path.expand(opts[:dolphin] || if(direct_inputs, do: "~/.local/share/slippi/exi-ai-float/dolphin-emu-headless", else: "~/.local/share/slippi/exi-ai/dolphin-emu-headless")),
  iso: Path.expand(opts[:iso] || "~/isos/melee.iso"),
  windowed: opts[:windowed] || false,
  trace_all: opts[:trace_all] || false,
  slippi_port: opts[:slippi_port] || 51480,
  input_offset: opts[:input_offset] || 1,
  response_delay: response_delay,
  drift_tolerance: opts[:drift_tolerance] || 3.0,
  window: opts[:window],
  finalize: opts[:finalize] || false,
  run_base: opts[:run_dir] || "logs/scenario_runs/#{ts}"
]

manifest_path = opts[:manifest] || "scenarios/manifest.json"
manifest = manifest_path |> File.read!() |> Jason.decode!()

# Whitelist decode (also forces the type atoms to exist before to_existing_atom)
type_by_name = Map.new(ScenarioScan.types(), fn t -> {to_string(t), t} end)

entries =
  manifest["entries"]
  |> Enum.map(fn e ->
    %{
      slp: e["slp"],
      frame: e["frame"],
      type: Map.fetch!(type_by_name, e["type"]),
      note: e["note"]
    }
  end)

entries =
  case opts[:types] do
    nil -> entries
    s ->
      wanted = s |> String.split(",") |> Enum.map(&Map.fetch!(type_by_name, &1))
      Enum.filter(entries, &(&1.type in wanted))
  end

entries =
  case opts[:only] do
    nil -> entries
    s ->
      idx = s |> String.split(",") |> Enum.map(&String.to_integer/1) |> MapSet.new()
      entries |> Enum.with_index() |> Enum.filter(fn {_, i} -> i in idx end) |> Enum.map(&elem(&1, 0))
  end

runs = opts[:runs] || 1
deterministic = opts[:temperature] == nil

Output.banner("ExPhil Scenario Suite")

Output.config([
  {"Policy", opts[:policy]},
  {"Manifest", manifest_path},
  {"Entries", length(entries)},
  {"Runs/entry", runs},
  {"Mode", if(deterministic, do: "deterministic", else: "temperature #{opts[:temperature]}")},
  {"Dolphin", suite_opts[:dolphin]},
  {"Headless", not suite_opts[:windowed]},
  {"Input offset", suite_opts[:input_offset]},
  {"Response delay", suite_opts[:response_delay]},
  {"Run dir", suite_opts[:run_base]}
])

if entries == [] do
  Output.error("No manifest entries after filtering")
  System.halt(1)
end

Output.step(1, 3, "Loading agent + parsing source replays")

# --driver (closed-loop correction validation, 2026-09-12): who drives port 1
# after the handoff. policy (default) = the checkpoint; teacher = the drill's
# scripted expert (MultishineExpert rules + table from --fixture), i.e. the
# CORRECTION the relabel would have written, executed for real; neutral = the
# control. Run the same manifest under each driver to promote only the
# corrections that actually restore the loop.
driver = String.to_atom(opts[:driver] || "policy")

unless suite_opts[:prefix_history] in ["applied", "committed", "cold"],
  do: raise(ArgumentError, "--prefix-history must be applied|committed|cold")

unless driver in [:policy, :teacher, :neutral],
  do: raise(ArgumentError, "--driver must be policy|teacher|neutral (got #{driver})")

if driver == :policy and suite_opts[:verify_input_timing] do
  Code.ensure_loaded!(Melee.Controller)
  unless function_exported?(Melee.Controller, :fix_pipe_analog_trigger, 1),
    do: raise("pipe_v2 verification requires the corrected libmelee_ex Controller; compile or reload it")
end

if opts[:audit_teacher_labels] && driver != :teacher,
  do: raise(ArgumentError, "--audit-teacher-labels requires --driver teacher")

agent =
  if driver == :policy do
    {:ok, agent} =
      Agent.start_link(
        policy_path: opts[:policy] || raise(ArgumentError, "--policy is required for --driver policy"),
        deterministic: deterministic,
        temperature: opts[:temperature] || 1.0,
        press_threshold: opts[:press_threshold] || 0.45,
        release_threshold: opts[:release_threshold] || 0.3,
        # --delay-id N: the rung to run a delay-conditioned checkpoint at in
        # THIS harness (frame-locked prefix loop); nil = the Agent derives it.
        delay_id: opts[:delay_id],
        allow_untrained_delay_id: opts[:delay_id] != nil,
        # INVARIANTS item 12: the harness declares itself + its knob; the Agent
        # derives (or checks) the delay-id through ExPhil.Eval.HarnessRung.
        harness: :scenario_suite,
        harness_knob: suite_opts[:response_delay],
        reaction_delay: reaction_delay,
        af_convention: ExPhil.Data.ActionFrameConvention.scenario_convention(opts)
      )

    case Agent.warmup(agent) do
      {:ok, ms} -> Output.success("Agent warmed up (#{ms}ms)")
      {:error, reason} -> Output.warning("Warmup failed: #{inspect(reason)}")
    end

    agent
  else
    nil
  end

expert =
  if driver == :teacher,
    do:
      ExPhil.Agents.MultishineExpert.from_fixture(
        opts[:fixture] || "test/fixtures/replays/fox_multishine_closed_d1.slp"
      ),
    else: nil

agent_runtime =
  if agent do
    agent
    |> :sys.get_state()
    |> Map.take([:delay_id, :reaction_delay, :harness, :harness_knob, :af_convention])
  end

agent_runtime = Map.merge(agent_runtime || %{}, %{float_ports: float_ports, input_transport: if(direct_inputs, do: "direct", else: "pipe")})
opts = Keyword.merge(opts, driver: driver, expert: expert)
suite_opts = Keyword.merge(suite_opts, driver: driver, expert: expert, character: opts[:character], opponent_character: opts[:opponent_character])

preps =
  entries
  |> Enum.map(& &1.slp)
  |> Enum.uniq()
  |> Map.new(fn slp ->
    # pipe_shim defaults ON: the bridge runs pipe inputs (EXI is opt-in
    # only — its analog RELEASE latches, see #66 addendum), and pipe mode
    # on the ExiAI build drops analog triggers, so recorded analog shield
    # holds must replay as digital presses. --no-pipe-shim for EXI runs.
    {slp, ScenarioSuite.prepare_replay(slp, pipe_shim: Keyword.get(opts, :pipe_shim, true), float_ports: float_ports)}
  end)

Output.success("Parsed #{map_size(preps)} source replay(s)")

Output.step(2, 3, "Running scenarios")

work = for entry <- entries, run_idx <- 1..runs, do: {entry, run_idx}

results =
  work
  |> Enum.with_index()
  |> Enum.map(fn {{entry, run_idx}, seq} ->
    Output.puts(
      "-- [#{seq + 1}/#{length(work)}] #{entry.type} @ #{Path.basename(entry.slp)}:#{entry.frame} (run #{run_idx})"
    )

    result =
      ScenarioSuite.run_one(entry, preps[entry.slp], agent, run_idx, seq, suite_opts)

    cond do
      result[:error] ->
        Output.error("   #{result[:error]}")

      result[:timing_valid] == false ->
        Output.warning("   INPUT TIMING INVALID: score withheld; #{inspect(result[:input_timing])}")

      result[:prefix_audit] && not result.prefix_audit.valid ->
        Output.warning("   EXACT PREFIX AUDIT FAILED: #{inspect(result.prefix_audit)}")

      result[:diverged] ->
        Output.warning(
          "   DIVERGED at handoff: #{inspect(Map.drop(result.drift, [:diverged]))} " <>
            "(score withheld from summary)"
        )

      true ->
        d = result.drift
        Output.success(
          "   score=#{result.score} pass=#{result.pass} " <>
            "drift p1=(#{d.p1.dx},#{d.p1.dy}) p2=(#{d.p2.dx},#{d.p2.dy}) " <>
            "#{result.wall_s}s | #{inspect(result.details)}"
        )
    end

    result
  end)

Output.step(3, 3, "Summary")

clean = Enum.filter(results, &(!&1[:error] and !&1[:diverged] and &1[:timing_valid] != false))
invalid_timing = Enum.filter(results, &(&1[:timing_valid] == false))
diverged = Enum.filter(results, & &1[:diverged])
errored = Enum.filter(results, & &1[:error])

summary =
  clean
  |> Enum.group_by(& &1.type)
  |> Map.new(fn {type, rs} ->
    {type,
     %{
       runs: length(rs),
       pass_rate: Float.round(Enum.count(rs, & &1.pass) / length(rs), 3),
       mean_score: Float.round(Enum.sum(Enum.map(rs, & &1.score)) / length(rs), 3),
       mean_wall_s: Float.round(Enum.sum(Enum.map(rs, & &1.wall_s)) / length(rs), 1)
     }}
  end)

Output.divider()

for {type, s} <- Enum.sort(summary) do
  Output.puts(
    "#{String.pad_trailing(to_string(type), 16)} runs=#{s.runs} pass=#{s.pass_rate} " <>
      "score=#{s.mean_score} wall=#{s.mean_wall_s}s"
  )
end

if diverged != [] do
  Output.warning("#{length(diverged)} run(s) DIVERGED at handoff (excluded from summary)")
end

if errored != [], do: Output.error("#{length(errored)} run(s) errored")
if invalid_timing != [], do: Output.warning("#{length(invalid_timing)} run(s) failed recorded-input timing validation")

out_path = opts[:out] || "logs/scenario_scores_#{ts}.json"
File.mkdir_p!(Path.dirname(out_path))

scoreboard = %{
  response_opponent: suite_opts[:response_opponent],
  console_timeout: suite_opts[:console_timeout],
  verify_input_timing: suite_opts[:verify_input_timing],
  invalid_timing_runs: length(invalid_timing),
  agent_runtime: agent_runtime,
  prefix_history: suite_opts[:prefix_history],
  driver: driver,
  fixture:
    if(driver == :teacher,
      do: opts[:fixture] || "test/fixtures/replays/fox_multishine_closed_d1.slp"
    ),
  audit_teacher_labels: opts[:audit_teacher_labels] || false,
  policy: opts[:policy],
  manifest: manifest_path,
  timestamp: ts,
  deterministic: deterministic,
  temperature: opts[:temperature],
  input_offset: suite_opts[:input_offset],
  response_delay: suite_opts[:response_delay],
  runs: results,
  summary: summary,
  diverged_runs: length(diverged),
  errored_runs: length(errored)
}

File.write!(out_path, Jason.encode!(scoreboard, pretty: true))
Output.success("Scoreboard written to #{out_path}")

if agent, do: GenServer.stop(agent)

# GOTCHAS #58/#63: sweep any orphaned Dolphin by exact PID. Safe here —
# the BEAM's own command line does not contain the pattern (the pkill -f
# self-match trap applies to shells that embed it).
case if(Keyword.get(opts, :orphan_sweep, true),
       do: System.cmd("pgrep", ["-f", "/tmp/libmelee_"]),
       else: {"", 1}
     ) do
  {out, 0} ->
    pids = out |> String.split("\n", trim: true)
    Enum.each(pids, fn pid -> System.cmd("kill", [pid]) end)
    if pids != [], do: Output.warning("Killed #{length(pids)} orphaned Dolphin(s)")

  _ ->
    :ok
end

if invalid_timing != [], do: System.halt(2)
