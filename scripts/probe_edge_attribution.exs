# Does the policy READ its own position when deciding to keep dashing?
#
# The edge-SD failure (fox dashdances, runs too far, runs off) is
# covariate shift — but the FIX depends on whether position/ledge_dist
# gates the dash decision at all:
#
#   position READ, data missing  -> DAgger edge corrections / ledge
#                                   oversampling (coverage fix)
#   position UNREAD              -> auxiliary edge-danger head / feature
#                                   work first (representation fix) —
#                                   more data won't help what isn't read
#
# (The P4 lesson: never assume a feature is read. ledge_dist is DERIVED
# from own-x at embed time — player.ex embed_ledge_distance — so patching
# x moves x + ledge_dist + offstage features consistently: a well-posed
# position counterfactual.)
#
# Method: three sequential passes per replay through a deterministic
# Agent (real / x patched OUTWARD toward the near edge / x patched
# INWARD toward center; the patch applies to every frame, so the GRU's
# history stays self-consistent — the absorber lesson). At grounded
# dash/run sites, compare the main-stick command's toward-edge
# component across passes. If shoving the fox 25 units closer to the
# edge doesn't change the stick, position is unread for this decision.
#
#   mix run scripts/probe_edge_attribution.exs \
#     [--policy checkpoints/fox_il_v2_20260809_041730_best_policy.bin] \
#     [--replays eval_runs/0809_regime_async2] [--shift 25] [--limit-replays 4]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Agents.Agent
alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, replays: :string, shift: :float, limit_replays: :integer]
  )

policy = opts[:policy] || "checkpoints/fox_il_v2_20260809_041730_best_policy.bin"
replay_dir = opts[:replays] || "eval_runs/0809_regime_async2"
shift = opts[:shift] || 25.0
limit = opts[:limit_replays] || 4

# libmelee Action enum: DASHING 0x14, RUNNING 0x15
dash_actions = MapSet.new([0x14, 0x15])
window_burn = 60

replays = Path.wildcard(Path.join(replay_dir, "**/r*.slp")) |> Enum.sort() |> Enum.take(limit)
if replays == [], do: raise("no replays under #{replay_dir}")

Output.banner("Edge attribution probe (position counterfactual)")
Output.puts("policy: #{Path.basename(policy)}")
Output.puts("replays: #{length(replays)} from #{replay_dir}, shift ±#{shift}")

# x-patches. Outward: toward the NEAR edge (same side), capped onstage.
# Inward: toward center, never crossing it (side flips would flip the
# "toward edge" frame of reference mid-history).
patch = fn frames, dir ->
  Enum.map(frames, fn f ->
    p = f.game_state.players[1]

    if p == nil or not is_number(p.x) do
      f
    else
      side = if p.x >= 0, do: 1.0, else: -1.0

      new_x =
        case dir do
          :outward -> side * min(abs(p.x) + shift, 84.0)
          :inward -> side * max(abs(p.x) - shift, 1.0)
        end

      players = Map.put(f.game_state.players, 1, %{p | x: new_x})
      %{f | game_state: %{f.game_state | players: players}}
    end
  end)
end

{:ok, agent} = Agent.start_link(policy_path: policy, deterministic: true)
Agent.warmup(agent)

run_pass = fn frames ->
  Agent.reset_buffer(agent)

  Enum.map(frames, fn f ->
    case Agent.get_controller(agent, f.game_state, player_port: 1) do
      {:ok, c} -> c.main_stick.x
      _ -> nil
    end
  end)
end

results =
  Enum.flat_map(replays, fn path ->
    {:ok, replay} = Peppi.parse(path, player_port: 1)

    frames =
      replay
      |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
      |> Enum.reject(&(&1.game_state.frame < 0))

    real = run_pass.(frames)
    outward = run_pass.(patch.(frames, :outward))
    inward = run_pass.(patch.(frames, :inward))

    # Sites from the REAL frames: grounded dash/run, past window burn-in
    frames
    |> Enum.with_index()
    |> Enum.filter(fn {f, i} ->
      p = f.game_state.players[1]

      i >= window_burn and p != nil and p.on_ground == true and
        MapSet.member?(dash_actions, trunc((p.action || -1) * 1.0)) and
        is_number(p.x) and Enum.at(real, i) != nil and
        Enum.at(outward, i) != nil and Enum.at(inward, i) != nil
    end)
    |> Enum.map(fn {f, i} ->
      p = f.game_state.players[1]
      side = if p.x >= 0, do: 1.0, else: -1.0

      # stick x in [0,1], 0.5 center; toward-edge component is signed
      # by which side the fox is on
      toward = fn stick_x -> side * (stick_x - 0.5) end

      %{
        x_abs: abs(p.x),
        near_edge: abs(p.x) > 55.0,
        real: toward.(Enum.at(real, i)),
        outward: toward.(Enum.at(outward, i)),
        inward: toward.(Enum.at(inward, i))
      }
    end)
  end)

GenServer.stop(agent, :normal, 10_000)

report = fn label, sites ->
  n = length(sites)

  if n < 20 do
    Output.puts("#{label}: only #{n} sites — skipping")
  else
    mean = fn key -> Float.round(Enum.sum(Enum.map(sites, & &1[key])) / n, 4) end
    rate = fn key -> Float.round(Enum.count(sites, &(&1[key] > 0.1)) / n * 100, 1) end

    mad = fn key ->
      Float.round(Enum.sum(Enum.map(sites, &abs(&1[key] - &1.real))) / n, 4)
    end

    Output.puts("")
    Output.puts("#{label} (#{n} sites):")
    Output.puts("  mean toward-edge stick:  real #{mean.(:real)}  outward #{mean.(:outward)}  inward #{mean.(:inward)}")
    Output.puts("  toward-edge command %:   real #{rate.(:real)}  outward #{rate.(:outward)}  inward #{rate.(:inward)}")
    Output.puts("  mean |Δstick| vs real:   outward #{mad.(:outward)}  inward #{mad.(:inward)}")
  end
end

report.("ALL dash/run sites", results)
report.("NEAR-EDGE sites (|x| > 55)", Enum.filter(results, & &1.near_edge))
report.("MID-STAGE sites (|x| <= 55)", Enum.reject(results, & &1.near_edge))

all_mad =
  if results != [],
    do:
      Enum.sum(Enum.map(results, &(abs(&1.outward - &1.real) + abs(&1.inward - &1.real)))) /
        (2 * length(results)),
    else: 0.0

Output.puts("")

if all_mad < 0.02 do
  Output.warning(
    "VERDICT: position looks UNREAD for dash steering (mean |Δstick| " <>
      "#{Float.round(all_mad, 4)} under ±#{shift}-unit patches) — coverage fixes " <>
      "(DAgger/oversampling) won't bite; aux edge-danger head first."
  )
else
  Output.success(
    "VERDICT: position IS read (mean |Δstick| #{Float.round(all_mad, 4)} under " <>
      "±#{shift}-unit patches) — the SD loop is a COVERAGE gap; DAgger edge " <>
      "corrections / ledge oversampling are the right lever."
  )
end
