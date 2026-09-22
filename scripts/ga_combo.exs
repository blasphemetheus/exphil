# GA for the longest combo from one start (SIM_USES_CHECKLIST "GA longest combo", v0).
#
#   devenv shell -- mix run scripts/ga_combo.exs --population 512 --generations 200 \
#     --horizon 90 --out eval_runs/0922_ga/fd_idle [--policy CKPT] [--seed 1] [--start-seed 1] \
#     [--start-percent 0] [--trace-every 5]
#
# Start: Fox (P1, attacker) vs Fox (P2) on FD at --start-percent, one random-walk
# start from Drill.build_pool (seeded). Defender: idle, or the prior (--policy)
# driving P2 (batched agent). Writes OUT/run.json (per-generation stats + elite
# genomes), OUT/gen<N>.msltrace.json for the elite every --trace-every
# generations (and the final best as best.msltrace.json), OUT/start.bin (the
# savestate) so the run can be replayed or extended.

alias ExPhil.Agents.Agent
alias ExPhil.Sim.{Drill, Env, GA, Trace}
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [move_bonus: :float, kill_bonus: :float, population: :integer, generations: :integer, horizon: :integer, out: :string, policy: :string, seed: :integer, start_seed: :integer, start_percent: :integer, trace_every: :integer, elite: :integer, mutation: :float, crossover: :float, stage: :string, max_hold: :integer]
  )

pop = opts[:population] || 256
gens = opts[:generations] || 50
horizon = opts[:horizon] || 90
out = opts[:out] || raise("--out required")
seed = opts[:seed] || 1
stage = opts[:stage] || "final_destination"
trace_every = opts[:trace_every] || 5
File.mkdir_p!(out)

Output.banner("GA longest combo v0")
Output.config([{"Population", pop}, {"Generations", gens}, {"Horizon", horizon}, {"Stage", stage}, {"Defender", opts[:policy] || "idle"}, {"Seed", seed}, {"Out", out}])

players = [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]

# 1. one start state from a batch-1 sim
{:ok, sim1} = Env.start(:nif, stage: stage, players: players, batch_size: 1, seed: seed)
[entry] = Drill.build_pool(sim1, 1, seed: opts[:start_seed] || seed, percent: opts[:start_percent] || 0, stage: stage, warm: 30)

# Settle the start: the random walk leaves players mid-animation (run 4 started with P1 in a jump,
# P2 mid-dash — a quarter of the window was dead time, Bradley 2026-09-22). Hold neutral until
# both stand in WAIT (action 14) on the ground, then save THAT as the start (cap 240 frames).
neutral = Drill.neutral()
{:ok, _} = Env.restore(sim1, 0, entry.blob)

settled =
  Enum.reduce_while(1..240, nil, fn _, _ ->
    {:ok, [gs], _} = Env.step(sim1, [[neutral, neutral]])
    p1 = gs.players[1]
    p2 = gs.players[2]
    if p1.action == 14 and p2.action == 14 and p1.on_ground and p2.on_ground, do: {:halt, gs}, else: {:cont, gs}
  end)

{:ok, settled_blob} = Env.save(sim1, 0)
walk_frame = entry.frame
entry = %{entry | blob: settled_blob, frame: settled.frame, summary: %{p1: ExPhil.Eval.ScenarioScan.player_summary(settled.players[1]), p2: ExPhil.Eval.ScenarioScan.player_summary(settled.players[2])}}
Output.puts("settled start: +#{settled.frame - walk_frame} f → frame #{settled.frame}, p1 action #{settled.players[1].action} p2 action #{settled.players[2].action}, distance #{Float.round(abs(settled.players[1].x - settled.players[2].x), 1)}")
Env.stop(sim1)
File.write!(Path.join(out, "start.bin"), entry.blob)
Output.puts("start: frame #{entry.frame}  p1 #{inspect(entry.summary.p1 |> Map.take([:x, :y, :action, :percent]))}  p2 #{inspect(entry.summary.p2 |> Map.take([:x, :y, :action, :percent]))}")

# 2. the population sim
{:ok, sim} = Env.start(:nif, stage: stage, players: players, batch_size: pop, seed: seed + 1)
{:ok, id} = Env.upload(sim, entry.blob)

defender =
  if p = opts[:policy] do
    Output.puts("⏳ loading the defender policy (JIT on first use)…")
    {:ok, a} = Agent.start_link(policy_path: p, deterministic: false, temperature: 1.0, af_convention: :parsed, frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true)
    {:ok, _} = Agent.warmup(a)
    a
  else
    :idle
  end

stage_id = %{"final_destination" => 32, "battlefield" => 31, "yoshis_story" => 8, "dreamland" => 28, "pokemon_stadium" => 3, "fountain_of_dreams" => 2}[stage] || 32
write_trace = fn states, name, label -> Trace.from_game_states(states, chars: [1, 1], stage: stage_id, label: label) |> Trace.write!(Path.join(out, name)) end

t0 = System.monotonic_time(:millisecond)

result =
  GA.run(sim, {:id, id},
    population: pop, generations: gens, horizon: horizon, seed: seed, defender: defender, history: entry.history,
    move_bonus: opts[:move_bonus] || 20.0, kill_bonus: opts[:kill_bonus] || 1000.0,
    elite: opts[:elite] || 8, mutation: opts[:mutation] || 0.15, crossover: opts[:crossover] || 0.7, max_hold: opts[:max_hold] || 12,
    on_generation: fn s ->
      bar = String.duplicate("█", min(20, s.best_chain * 2)) |> String.pad_trailing(20, "░")
      Output.puts("gen #{String.pad_leading(Integer.to_string(s.generation), 3)}  best #{:io_lib.format("~7.1f", [s.best])}  mean #{:io_lib.format("~7.1f", [s.mean])}  chain #{s.best_chain} moves (#{s.best_hits} hits) #{Float.round(s.best_chain_damage, 1)}% in-chain / #{Float.round(s.best_damage, 1)}% total#{if s.best_stocks > 0, do: " ★KILL", else: ""}  #{bar}  best-so-far #{Float.round(s.best_so_far, 1)}  (#{s.ms} ms)")
      if rem(s.generation, trace_every) == 0 or s.generation == 1 do
        write_trace.(s.elite.states, "gen#{s.generation}.msltrace.json", "gen #{s.generation} elite: #{s.best_chain} moves, #{Float.round(s.best_damage, 1)}% (fitness #{Float.round(s.best, 1)})")
      end
    end)

ms = System.monotonic_time(:millisecond) - t0
best = result.best
write_trace.(best.states, "best.msltrace.json", "best of run (gen #{best.generation}): #{best.score.chain} moves, #{Float.round(best.score.damage, 1)}%")

File.write!(
  Path.join(out, "run.json"),
  Jason.encode!(
    %{
      population: pop, n_generations: gens, horizon: horizon, stage: stage_id, defender: opts[:policy] || "idle", seed: seed, ms: ms,
      start: %{frame: entry.frame, p1: entry.summary.p1, p2: entry.summary.p2},
      best: %{generation: best.generation, fitness: best.score.fitness, chain: best.score.chain, hits: best.score.hits, chain_damage: best.score.chain_damage, stocks_taken: best.score.stocks_taken, damage: best.score.damage, alive: best.score.alive?, genome: Enum.map(best.genome, fn {n, h} -> [n, h] end), trace: "best.msltrace.json"},
      generations:
        Enum.map(result.generations, fn s ->
          %{generation: s.generation, best: s.best, mean: s.mean, median: s.median, best_chain: s.best_chain, best_hits: s.best_hits, best_chain_damage: s.best_chain_damage, best_stocks: s.best_stocks, best_damage: s.best_damage, best_so_far: s.best_so_far, chains: s.chains, ms: s.ms,
            elite_genome: Enum.map(s.elite_genome, fn {n, h} -> [n, h] end),
            trace: if(rem(s.generation, trace_every) == 0 or s.generation == 1, do: "gen#{s.generation}.msltrace.json", else: nil)}
        end)
    },
    pretty: true
  )
)

Output.success("#{gens} generations × #{pop} in #{Float.round(ms / 1000, 1)} s; best gen #{best.generation}: #{best.score.chain} moves (#{best.score.hits} hits), #{Float.round(best.score.damage, 1)}%, fitness #{Float.round(best.score.fitness, 1)} → #{out}/run.json")
Output.puts("best genome: " <> Enum.map_join(best.genome, " ", fn {n, h} -> "#{n}×#{h}" end))
Env.stop(sim)
