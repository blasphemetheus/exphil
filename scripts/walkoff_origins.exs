# Walk-off origins (2026-10-25): HOW does the bot leave the stage? The 10-24
# testbed read: --neutral-weight 0.5 tripled SDs while recovery per trip was
# unchanged — the bot simply went offstage 3x more often against an IDLE
# opponent. This script runs the same closed-loop rollout as
# sim_closed_loop.exs and classifies every offstage TRIP START:
#   hit_off      in hitstun at the crossing, or damage taken within 20 f
#   special_off  crossed in side-B / up-B / shine (350..369)
#   jump_off     left the ground by a jump within 30 f before crossing
#   walk_off     last grounded action was a walk / dash / run / turn / wait and
#                no jump: fell off the ledge on the ground
#   aerial_drift airborne over the stage for 30+ f (after an aerial / land
#                cancel) and drifted out
# plus, per origin: the stick-toward-edge share on the 10 frames before the
# crossing, the x speed, and the outcome (recovered / died). Idle opponent, so
# every trip is the bot's own.
#
#   mix run scripts/walkoff_origins.exs --policy P --label L [--envs 32] [--frames 1800]
#     [--seed 1001] [--out FILE.json]
# Run with EDIFICE_FUSED_CUSTOM_CALL=0 (the fused handlers crash Agent warmup
# on crypto load since Edifice 6d5b0cc, 10-24).

alias ExPhil.Agents.Agent
alias ExPhil.Sim.{Env, Drill, GA}
alias ExPhil.Training.{Checkpoint, Output}
alias ExPhil.Interp.ActionNames

{opts, _, bad} = OptionParser.parse(System.argv(), strict: [policy: :string, label: :string, envs: :integer, frames: :integer, seed: :integer, out: :string, stateful_step: :boolean])
if bad != [], do: raise("invalid options: #{inspect(bad)}")
policy = opts[:policy] || raise("--policy required")
label = opts[:label] || Path.basename(Path.dirname(policy))
n = opts[:envs] || 32
frames = opts[:frames] || 1800
stage = "final_destination"
edge = GA.stage_edge(32)
neutral = Drill.neutral()

{:ok, export} = Checkpoint.load_policy(policy)
contract = ExPhil.Networks.Policy.ExecutionContract.load(export.config)
{:ok, actor} =
  Agent.start_link(policy_path: policy, deterministic: false, temperature: 1.0, af_convention: :parsed,
    frame_delay: 0, reaction_delay: 0, harness: :sync_runner,
    stateful_step: opts[:stateful_step] || contract.recurrent_state == :carried_zero)
{:ok, _} = Agent.warmup(actor)
:ok = Agent.batch_init(actor, n)

Output.banner("Walk-off origins: #{label}")
players = [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]
{:ok, sim} = Env.start(:nif, stage: stage, players: players, batch_size: n, seed: opts[:seed] || 1001)
{:ok, _, _} = Env.observe(sim)
{:ok, gs0} = Env.frames(sim)
starts = for i <- 0..(n - 1), into: %{}, do: ({:ok, blob} = Env.save(sim, i); {i, blob})

offstage? = fn p -> not p.on_ground and (abs(p.x) > edge or p.y < -5.0) end
jumpish? = fn a -> a in 24..34 end            # KNEE_BEND .. aerial jumps
groundmove? = fn a -> a in 14..23 or a in 38..41 end  # wait/walk/turn/dash/run (+ squat family)
special? = fn a -> a >= 350 and a <= 369 end
act = fn p -> trunc(p.action || 0) end

# per env: a 30-frame history of {player, controller}, the last grounded frame's action, the open trip
env0 = %{hist: [], last_hit: -10_000, off: false, trip: nil}
acc0 = %{trips: [], frames: 0, games: 0}

classify = fn e, p1, t ->
  recent = Enum.take(e.hist, 30)                       # newest first: {p, c, t}
  hit_recent = t - e.last_hit <= 20
  jumped = Enum.any?(recent, fn {p, _, _} -> jumpish?.(act.(p)) end)
  airborne_run = Enum.take_while(recent, fn {p, _, _} -> not p.on_ground end) |> length()
  last_ground = Enum.find(recent, fn {p, _, _} -> p.on_ground end)
  origin =
    cond do
      (p1.hitstun_frames_left || 0) > 0.5 or hit_recent -> :hit_off
      special?.(act.(p1)) -> :special_off
      jumped and airborne_run <= 30 -> :jump_off
      airborne_run >= 30 -> :aerial_drift
      last_ground != nil and groundmove?.(act.(elem(last_ground, 0))) -> :walk_off
      true -> :other
    end
  side = if p1.x > 0, do: 1, else: -1
  last10 = Enum.take(recent, 10)
  toward = Enum.count(last10, fn {_, c, _} -> (c.main_stick.x - 0.5) * side > 0.2 end) / max(length(last10), 1)
  held_out = Enum.count(last10, fn {_, c, _} -> (c.main_stick.x - 0.5) * side > 0.45 end) / max(length(last10), 1)
  %{origin: origin, t: t, x: Float.round(p1.x * 1.0, 1), y: Float.round(p1.y * 1.0, 1), vx: Float.round(((p1.speed_ground_x_self || 0.0) + (p1.speed_air_x_self || 0.0)) * 1.0, 2),
    last_ground_action: last_ground && ActionNames.name(act.(elem(last_ground, 0))), action: ActionNames.name(act.(p1)),
    airborne_run: airborne_run, stick_toward_share: Float.round(toward, 2), stick_full_out_share: Float.round(held_out, 2), jumps_left: p1.jumps_left}
end

t0 = System.monotonic_time(:millisecond)
{acc, _, _} =
  Enum.reduce(1..frames, {acc0, List.duplicate(env0, n), gs0}, fn t, {acc, envs, states} ->
    {:ok, c1s} = Agent.batch_get_controllers(actor, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1)
    {:ok, nexts, terms} = Env.step(sim, Enum.zip_with(c1s, List.duplicate(neutral, n), fn a, b -> [a || neutral, b] end))

    {acc, envs} =
      Enum.zip([states, nexts, c1s, envs])
      |> Enum.reduce({acc, []}, fn {s, nx, c, e}, {acc, out} ->
        p0 = s.players[1]; p1 = nx.players[1]; c = c || neutral
        hit? = p1.percent > p0.percent
        last_hit = if hit?, do: t, else: e.last_hit
        died? = (p1.stock || 0) < (p0.stock || 0)
        now_off = offstage?.(p1) and not died?
        hist = Enum.take([{p0, c, t} | e.hist], 40)
        {trip, acc} =
          cond do
            now_off and not e.off -> {classify.(%{e | last_hit: last_hit, hist: hist}, p1, t), acc}
            e.off and died? and e.trip -> {nil, %{acc | trips: [Map.put(e.trip, :outcome, :died) | acc.trips]}}
            e.off and not now_off and e.trip -> {nil, %{acc | trips: [Map.put(e.trip, :outcome, :recovered) | acc.trips]}}
            true -> {e.trip, acc}
          end
        {%{acc | frames: acc.frames + 1}, [%{e | hist: hist, last_hit: last_hit, off: now_off, trip: trip} | out]}
      end)
    envs = Enum.reverse(envs)

    finished = terms |> Enum.with_index() |> Enum.filter(fn {term, _} -> (term["done"] || 0) == 1 end) |> Enum.map(&elem(&1, 1))
    {nexts, envs, acc} =
      if finished == [] do
        {nexts, envs, acc}
      else
        for i <- finished, do: {:ok, _} = Env.restore(sim, i, starts[i])
        :ok = Agent.batch_reset_rows(actor, finished)
        {:ok, reset} = Env.frames(sim)
        {reset, envs |> Enum.with_index() |> Enum.map(fn {e, i} -> if i in finished, do: env0, else: e end), %{acc | games: acc.games + length(finished)}}
      end
    if rem(t, 600) == 0, do: Output.puts("  frame #{t}/#{frames}  trips #{length(acc.trips)}  (#{round((System.monotonic_time(:millisecond) - t0) / t)} ms/step)")
    {acc, envs, nexts}
  end)

trips = Enum.reverse(acc.trips)
minutes = acc.frames / 3600
by = Enum.group_by(trips, & &1.origin)
mean = fn xs -> if xs == [], do: nil, else: Float.round(Enum.sum(xs) / length(xs), 2) end
summary =
  Enum.map([:walk_off, :jump_off, :aerial_drift, :special_off, :hit_off, :other], fn o ->
    ts = by[o] || []
    %{origin: o, n: length(ts), per_min: Float.round(length(ts) / max(minutes, 1.0e-3), 2),
      died_share: (if ts == [], do: nil, else: Float.round(Enum.count(ts, &(&1.outcome == :died)) / length(ts), 2)),
      stick_toward: mean.(Enum.map(ts, & &1.stick_toward_share)), stick_full_out: mean.(Enum.map(ts, & &1.stick_full_out_share)),
      jumps_left: mean.(Enum.map(ts, &(&1.jumps_left || 0))),
      last_ground: ts |> Enum.map(& &1.last_ground_action) |> Enum.reject(&is_nil/1) |> Enum.frequencies() |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.take(3) |> Enum.map(fn {k, v} -> [k, v] end)}
  end)

Output.puts("RESULT #{label} walk-off origins: #{length(trips)} trips in #{Float.round(minutes, 1)} env-min (#{Float.round(length(trips) / max(minutes, 1.0e-3), 2)}/min)")
Enum.each(summary, fn s ->
  Output.puts("RESULT #{label}   #{String.pad_trailing(to_string(s.origin), 13)} n #{String.pad_leading(to_string(s.n), 3)}  #{s.per_min}/min  died #{inspect(s.died_share)}  stick toward edge (10 f before) #{inspect(s.stick_toward)}  full-out #{inspect(s.stick_full_out)}  jumps left #{inspect(s.jumps_left)}  last ground action #{inspect(s.last_ground)}")
end)
out = opts[:out] || "eval_runs/walkoff/#{label}.json"
File.mkdir_p!(Path.dirname(out))
File.write!(out, Jason.encode!(%{label: label, policy: policy, envs: n, frames: frames, env_minutes: minutes, summary: summary, trips: trips}))
Output.puts("RESULT #{label} -> #{out}")
