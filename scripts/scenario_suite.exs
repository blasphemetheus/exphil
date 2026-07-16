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
#   --dolphin PATH       Dolphin (default ~/.local/share/slippi/exi-ai/dolphin-emu-headless)
#   --iso PATH           Melee ISO (default ~/isos/melee.iso)
#   --windowed           Disable headless (debugging; needs the netplay build)
#   --slippi-port N      Base slippi port (default 51480, +1 per run)
#   --out PATH           Scoreboard JSON (default logs/scenario_scores_<ts>.json)
#   --run-dir PATH       Base dir for per-run replay dirs (default logs/scenario_runs/<ts>)
#   --press-threshold F / --release-threshold F   Button hysteresis (probe recipe: 0.45/0.3)
#   --quiet / --verbose
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

  # ~3 min of menu frames before declaring the boot stuck
  @menu_step_limit 12_000

  # ==========================================================================
  # Replay preparation (once per source .slp)
  # ==========================================================================

  def prepare_replay(slp) do
    {:ok, replay} = Peppi.parse(Path.expand(slp))

    {inputs, ref} =
      Enum.reduce(replay.frames, {%{}, %{}}, fn f, {inputs, ref} ->
        p1 = f.players[1]
        p2 = f.players[2]

        if p1 && p2 do
          {
            Map.put(inputs, f.frame_number, {rec_input(p1.controller), rec_input(p2.controller)}),
            Map.put(ref, f.frame_number, %{
              p1: ScenarioScan.player_summary(p1),
              p2: ScenarioScan.player_summary(p2)
            })
          }
        else
          {inputs, ref}
        end
      end)

    %{inputs: inputs, ref: ref}
  end

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
  @trigger_digital_threshold 0.31

  defp rec_input(c) do
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
        l: c.button_l or c.l_trigger > @trigger_digital_threshold,
        r: c.button_r or c.r_trigger > @trigger_digital_threshold,
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
      iso_path: opts[:iso],
      controller_port: 1,
      opponent_port: 2,
      character: :mewtwo,
      stage: :final_destination,
      online_delay: 0,
      dummy_mode: "external",
      dummy_character: "fox",
      dummy_cpu_level: 0,
      no_audio: true,
      headless: not opts[:windowed],
      replay_dir: run_dir,
      slippi_port: opts[:slippi_port] + seq
    }

    Agent.reset_buffer(agent)

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

    wall_s = (System.monotonic_time(:millisecond) - t0) / 1000

    Map.merge(
      %{
        type: type,
        slp: entry.slp,
        frame: entry.frame,
        note: entry.note,
        run: run_idx,
        window: window,
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

          :respond ->
            finish(%{st | truncated: :game_ended})

          :prefix ->
            %{error: "game ended during prefix (catastrophic divergence?)"}
        end

      {:postgame, _gs} ->
        case st.phase do
          :respond -> finish(%{st | truncated: :game_ended})
          :menu -> loop(bridge, %{st | menu_steps: st.menu_steps + 1})
          :prefix -> %{error: "game ended during prefix (catastrophic divergence?)"}
        end

      {:ok, gs} ->
        st = if st.phase == :menu, do: %{st | phase: :prefix}, else: st

        case st.phase do
          :prefix -> prefix_frame(bridge, gs, st)
          :respond -> respond_frame(bridge, gs, st)
        end

      {:game_ended, reason} ->
        if st.phase == :respond,
          do: finish(%{st | truncated: :game_ended}),
          else: %{error: "dolphin ended: #{reason}"}

      {:error, reason} ->
        %{error: "step error: #{inspect(reason)}"}
    end
  end

  defp prefix_frame(bridge, gs, st) do
    f = gs.frame

    st = track_drift(st, gs, f)

    if f >= st.handoff do
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
      case st.inputs[f + st.offset] do
        {p1, p2} ->
          MeleePort.send_controller(bridge, p1)
          MeleePort.send_controller(bridge, Map.put(p2, :port, 2))

        nil ->
          MeleePort.send_controller(bridge, @neutral)
          MeleePort.send_controller(bridge, Map.put(@neutral, :port, 2))
      end

      loop(bridge, st)
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
      finish(st)
    else
      # Port 1: the policy takes over.
      case Agent.get_controller(st.agent, gs, player_port: 1) do
        {:ok, controller} ->
          MeleePort.send_controller(bridge, controller_to_input(controller))

        {:error, reason} ->
          Logger.warning("[scenario] agent error at f=#{f}: #{inspect(reason)}")
          MeleePort.send_controller(bridge, @neutral)
      end

      # Port 2: keeps replaying the source game's inputs (deterministic,
      # non-reactive); neutral once the recording runs out.
      p2 =
        case st.inputs[f + st.offset] do
          {_p1, p2} -> p2
          nil -> @neutral
        end

      MeleePort.send_controller(bridge, Map.put(p2, :port, 2))

      loop(bridge, st)
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
      manifest: :string,
      types: :string,
      only: :string,
      runs: :integer,
      temperature: :float,
      window: :integer,
      input_offset: :integer,
      drift_tolerance: :float,
      dolphin: :string,
      iso: :string,
      windowed: :boolean,
      trace_all: :boolean,
      slippi_port: :integer,
      out: :string,
      run_dir: :string,
      press_threshold: :float,
      release_threshold: :float,
      quiet: :boolean,
      verbose: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

unless opts[:policy] do
  Output.error("--policy is required")
  System.halt(1)
end

ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")

suite_opts = [
  dolphin: Path.expand(opts[:dolphin] || "~/.local/share/slippi/exi-ai/dolphin-emu-headless"),
  iso: Path.expand(opts[:iso] || "~/isos/melee.iso"),
  windowed: opts[:windowed] || false,
  trace_all: opts[:trace_all] || false,
  slippi_port: opts[:slippi_port] || 51480,
  input_offset: opts[:input_offset] || 1,
  drift_tolerance: opts[:drift_tolerance] || 3.0,
  window: opts[:window],
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
  {"Run dir", suite_opts[:run_base]}
])

if entries == [] do
  Output.error("No manifest entries after filtering")
  System.halt(1)
end

Output.step(1, 3, "Loading agent + parsing source replays")

{:ok, agent} =
  Agent.start_link(
    policy_path: opts[:policy],
    deterministic: deterministic,
    temperature: opts[:temperature] || 1.0,
    press_threshold: opts[:press_threshold] || 0.45,
    release_threshold: opts[:release_threshold] || 0.3
  )

case Agent.warmup(agent) do
  {:ok, ms} -> Output.success("Agent warmed up (#{ms}ms)")
  {:error, reason} -> Output.warning("Warmup failed: #{inspect(reason)}")
end

preps =
  entries
  |> Enum.map(& &1.slp)
  |> Enum.uniq()
  |> Map.new(fn slp -> {slp, ScenarioSuite.prepare_replay(slp)} end)

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

clean = Enum.filter(results, &(!&1[:error] and !&1[:diverged]))
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

out_path = opts[:out] || "logs/scenario_scores_#{ts}.json"
File.mkdir_p!(Path.dirname(out_path))

scoreboard = %{
  policy: opts[:policy],
  manifest: manifest_path,
  timestamp: ts,
  deterministic: deterministic,
  temperature: opts[:temperature],
  input_offset: suite_opts[:input_offset],
  runs: results,
  summary: summary,
  diverged_runs: length(diverged),
  errored_runs: length(errored)
}

File.write!(out_path, Jason.encode!(scoreboard, pretty: true))
Output.success("Scoreboard written to #{out_path}")

GenServer.stop(agent)

# GOTCHAS #58/#63: sweep any orphaned Dolphin by exact PID. Safe here —
# the BEAM's own command line does not contain the pattern (the pkill -f
# self-match trap applies to shells that embed it).
case System.cmd("pgrep", ["-f", "/tmp/libmelee_"]) do
  {out, 0} ->
    pids = out |> String.split("\n", trim: true)
    Enum.each(pids, fn pid -> System.cmd("kill", [pid]) end)
    if pids != [], do: Output.warning("Killed #{length(pids)} orphaned Dolphin(s)")

  _ ->
    :ok
end
