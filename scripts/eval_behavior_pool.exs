#!/usr/bin/env elixir
# Parallel LIVE-SCORED behavior evals — the SessionPool + GameEvents
# wiring (libmelee_ex arc 2026-08-09, item "next wiring opportunity").
#
# N headless sessions run CONCURRENTLY in one BEAM, each scored live by
# ExPhil.Eval.LiveScorer (Melee.GameEvents fold: SD-vs-KO stock losses,
# shield breaks + per-frame diversity/damage/offstage) — no post-hoc
# replay parsing, and n>=8 protocol runs cost roughly the wall time of
# the slowest single run. Per-instance isolation follows
# Melee.SessionPool's recipe: a free UDP spectator port per worker
# (probed upward from --base-port) and a per-run replay dir; the GC
# adapter is never claimed (libmelee_ex declare_ports unplugs
# undeclared ports).
#
# Usage:
#   mix run scripts/eval_behavior_pool.exs \
#     --policy checkpoints/fox_il_v2_20260809_041730_best_policy.bin \
#     --runs 8 --parallel 4 --seconds 120 --temperature 0.3 \
#     --outdir eval_runs/0809_v2_pool \
#     2>&1 | tee eval_runs/0809_v2_pool.log
#
# Options:
#   --policy PATH        policy .bin (required)
#   --runs N             total runs [4]
#   --parallel N         concurrent sessions [2]
#   --seconds S          in-game seconds per run [120]
#   --temperature T      sampling temperature; omit for deterministic
#   --character NAME     bot character [fox]
#   --stage NAME         stage [final_destination]
#   --dummy-cpu-level N  opponent CPU level [1]
#   --dolphin PATH       Dolphin dir/executable [exi-ai headless build]
#   --iso PATH           Melee ISO [auto-detect ~/isos, ~/games]
#   --outdir DIR         replay/report dir (required)
#   --base-port N        UDP spectator port probe base [52000]
#   --emulation-speed F  0 = unthrottled (fast, sync-pacing regime) [0.0];
#                        1.0 = realtime — closer to the async-runner
#                        regime the 0809 baseline evals ran in
#
# PACING CAVEAT: pool workers run the sync blocking-input loop. That is
# a different effective-delay regime than eval_live_protocol's async
# runner at realtime — do NOT mix pool and async numbers in one
# comparison (the 0809 smoke showed regime-sized behavior deltas for a
# delay-untrained policy). Compare pool-vs-pool.

alias ExPhil.Agents.Agent
alias ExPhil.Bridge.MeleePort
alias ExPhil.Eval.LiveScorer
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      runs: :integer,
      parallel: :integer,
      seconds: :integer,
      temperature: :float,
      character: :string,
      stage: :string,
      dummy_cpu_level: :integer,
      dolphin: :string,
      iso: :string,
      outdir: :string,
      base_port: :integer,
      emulation_speed: :float,
      quiet: :boolean
    ]
  )

policy = opts[:policy] || raise "--policy is required"
outdir = opts[:outdir] || raise "--outdir is required"
runs = opts[:runs] || 4
parallel = opts[:parallel] || 2
seconds = opts[:seconds] || 120
temperature = opts[:temperature]
character = String.to_atom(opts[:character] || "fox")
stage = String.to_atom(opts[:stage] || "final_destination")
cpu_level = opts[:dummy_cpu_level] || 1
base_port = opts[:base_port] || 52_000

dolphin =
  opts[:dolphin] ||
    Path.expand("~/.local/share/slippi/exi-ai/dolphin-emu-headless")

iso =
  opts[:iso] ||
    Enum.find(
      [Path.expand("~/isos/melee.iso"), Path.expand("~/games/melee.iso")],
      &File.exists?/1
    ) || raise "no ISO found; pass --iso"

File.mkdir_p!(outdir)

Output.banner("Pool Behavior Eval")

Output.config([
  {"Policy", policy},
  {"Runs", "#{runs} (#{parallel} parallel)"},
  {"Seconds/run", seconds},
  {"Decode", if(temperature, do: "temperature #{temperature}", else: "deterministic")},
  {"Matchup", "#{character} vs level-#{cpu_level} CPU on #{stage}"},
  {"Outdir", outdir}
])

# Free UDP port at or above candidate (Melee.SessionPool's probe).
free_udp_port = fn candidate, probe ->
  Enum.find(candidate..(candidate + 500), fn p ->
    case :gen_udp.open(p, [:binary]) do
      {:ok, s} -> :gen_udp.close(s) == :ok
      {:error, _} -> false
    end
  end) || raise "no free UDP port near #{probe}"
end

# Non-blocking check for the MeleePort menu watchdog's notify message
# (the bridge sends it to this worker process once per stall episode).
menu_stuck? = fn ->
  receive do
    {:melee_port, :menu_stuck, _report} -> true
  after
    0 -> false
  end
end

run_one = fn agent, run_idx, slippi_port ->
  rundir = Path.join(outdir, "r#{run_idx}")
  File.mkdir_p!(rundir)

  {:ok, bridge} = MeleePort.start_link([])

  config = %{
    dolphin_path: dolphin,
    iso_path: iso,
    headless: true,
    blocking_input: true,
    emulation_speed: opts[:emulation_speed] || 0.0,
    slippi_port: slippi_port,
    replay_dir: rundir,
    character: character,
    stage: stage,
    dummy_mode: "cpu",
    dummy_character: character,
    dummy_cpu_level: cpu_level,
    # Menu watchdog: unthrottled menus take seconds, so 20s of zero
    # progress means wedged — abort the run (see loop) instead of
    # burning the wall deadline (the pre-fix steer_toward freeze cost
    # 3 runs x 7 minutes each).
    menu_stuck_frames: 1200,
    menu_stuck_notify: self()
  }

  {:ok, _} = MeleePort.init_console(bridge, config, 120_000)

  frame_cap = seconds * 60
  deadline = System.monotonic_time(:millisecond) + (seconds + 420) * 1000

  loop = fn loop, scorer, no_frame_streak ->
    cond do
      scorer.frames >= frame_cap ->
        {:cap, scorer}

      System.monotonic_time(:millisecond) > deadline ->
        {:timeout, scorer}

      menu_stuck?.() ->
        {:menu_stuck, scorer}

      true ->
        case MeleePort.step(bridge, auto_menu: true, poll: true) do
          {:ok, gs} ->
            controller =
              case Agent.get_controller(agent, gs) do
                {:ok, c} -> c
                _ -> nil
              end

            if controller, do: MeleePort.send_controller(bridge, controller)
            loop.(loop, LiveScorer.step(scorer, gs, controller), 0)

          {:menu, _gs} ->
            loop.(loop, scorer, 0)

          :no_frame ->
            if no_frame_streak > 600 do
              {:hung, scorer}
            else
              loop.(loop, scorer, no_frame_streak + 1)
            end

          {:postgame, _} ->
            {:game_end, scorer}

          {:game_ended, _} ->
            {:game_end, scorer}

          {:error, reason} ->
            {{:error, reason}, scorer}
        end
    end
  end

  {outcome, scorer} = loop.(loop, LiveScorer.new(1), 0)

  MeleePort.stop(bridge)

  {run_idx, outcome, LiveScorer.report(scorer)}
end

start = System.monotonic_time(:millisecond)

# One worker per parallel slot; each warms ONE agent (JIT is the
# per-run startup cost — ~minutes — so it must amortize across the
# worker's whole share of runs) and drives its runs sequentially with a
# fresh Dolphin per run.
worker = fn worker_idx ->
  agent_opts = [
    policy_path: policy,
    name: :"pool_agent_#{worker_idx}",
    deterministic: temperature == nil
  ]

  agent_opts = if temperature, do: agent_opts ++ [temperature: temperature], else: agent_opts

  {:ok, agent} = Agent.start_link(agent_opts)
  Agent.warmup(agent)

  my_runs = Enum.filter(1..runs, &(rem(&1 - 1, parallel) == worker_idx - 1))

  results =
    for run_idx <- my_runs do
      port = free_udp_port.(base_port + (worker_idx - 1) * 3, base_port)

      try do
        run_one.(agent, run_idx, port)
      catch
        kind, reason -> {run_idx, {:crashed, kind, reason}, nil}
      end
    end

  GenServer.stop(agent, :normal, 10_000)
  results
end

results =
  1..parallel
  |> Task.async_stream(worker,
    max_concurrency: parallel,
    timeout: div(runs + parallel - 1, parallel) * (seconds + 480) * 1000 + 300_000,
    on_timeout: :kill_task
  )
  |> Enum.flat_map(fn
    {:ok, worker_results} -> worker_results
    {:exit, reason} -> [{:worker_crash, reason, nil}]
  end)
  |> Enum.sort()

elapsed_s = div(System.monotonic_time(:millisecond) - start, 1000)

keys = [
  :seconds, :stocks_lost, :sd_deaths, :ko_deaths, :stocks_taken, :damage_dealt,
  :shield_pct, :shieldbreaks, :distinct_actions, :inputs_per_min, :offstage_pct
]

header = ~w(run outcome secs stk_lost SD KO stk_taken dmg shield% breaks actions inp/min offstage%)

Output.puts("")
Output.puts("  " <> Enum.join(header, "  "))

rows =
  for result <- results do
    case result do
      {i, outcome, report} when is_map(report) ->
        Output.puts(
          "  r#{i}  #{inspect(outcome)}  " <> Enum.map_join(keys, "  ", &"#{report[&1]}")
        )

        report

      other ->
        Output.warning("worker failed: #{inspect(other, limit: 3)}")
        nil
    end
  end
  |> Enum.reject(&is_nil/1)
  |> Enum.reject(&(&1.seconds == 0.0))

if rows != [] do
  means =
    Map.new(keys, fn k ->
      {k, Float.round(Enum.sum(Enum.map(rows, &(&1[k] * 1.0))) / length(rows), 1)}
    end)

  Output.puts("  MEAN(#{length(rows)})  -  " <> Enum.map_join(keys, "  ", &"#{means[&1]}"))
end

Output.success("#{length(rows)}/#{runs} runs scored in #{div(elapsed_s, 60)}m #{rem(elapsed_s, 60)}s")
