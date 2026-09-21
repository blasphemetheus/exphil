# SIM_INTEGRATION.md step 6: the Fox fair-conversion drill — baseline of the
# frozen prior on a fixed start pool. Builds (or loads) a pool of randomized
# starts, rolls the attacker (port 1) vs the defender (port 2) for `horizon`
# frames from each, scores with FairConversion + AerialChain, writes the
# pool (reusable by search-as-teacher) and a results summary.
#
#   devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.3 EXPHIL_EXLA_PRECISION=highest \
#     mix run scripts/sim_drill.exs --policy checkpoints/.../model_best_policy.bin \
#       --starts 300 --horizon 120 --defender self|idle --out eval_runs/0921_sim_drill/v0 [--seed 1]

alias ExPhil.Agents.Agent
alias ExPhil.Bridge.SimPort
alias ExPhil.Sim.Drill
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [policy: :string, starts: :integer, horizon: :integer, defender: :string, out: :string, seed: :integer, temperature: :float, pool: :string, max_distance: :float])
policy = opts[:policy] || raise("--policy required")
starts = opts[:starts] || 100
horizon = opts[:horizon] || 120
defender_kind = opts[:defender] || "self"
out = opts[:out] || raise("--out DIR required")
seed = opts[:seed] || 1
File.mkdir_p!(out)

Output.banner("Fair-conversion drill v0 (step 6)")
Output.config([{"Policy", policy}, {"Starts", starts}, {"Horizon", horizon}, {"Defender", defender_kind}, {"Pool", opts[:pool] || "play"}, {"Seed", seed}, {"Out", out}])

agent_opts = [policy_path: policy, deterministic: false, temperature: opts[:temperature] || 1.0, af_convention: :parsed, frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true]

Output.step(1, 3, "Agents + sim")
{:ok, attacker} = Agent.start_link(agent_opts)
{:ok, _} = Agent.warmup(attacker)
defender = if defender_kind == "self", do: (fn -> {:ok, d} = Agent.start_link(agent_opts); {:ok, _} = Agent.warmup(d); d end).(), else: :idle
{:ok, sim} = SimPort.start_link(stage: "final_destination", players: [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}], length: 256, seed: seed)

Output.step(2, 3, "Building start pool (#{starts})")
t0 = System.monotonic_time(:millisecond)
pool =
  case opts[:pool] || "play" do
    "walk" -> Drill.build_pool(sim, starts, seed: seed, max_distance: opts[:max_distance] || 50.0)
    "play" -> Drill.build_pool_from_play(sim, attacker, defender, starts, seed: seed, max_distance: opts[:max_distance] || 50.0)
  end
Output.puts("pool: #{length(pool)} starts in #{System.monotonic_time(:millisecond) - t0} ms; sample #{inspect(Enum.at(pool, 0).summary)}")
Drill.pool_to_disk(pool, Path.join(out, "pool.jsonl"))

Output.step(3, 3, "Rolling out")
t1 = System.monotonic_time(:millisecond)
io = File.open!(Path.join(out, "rollouts.jsonl"), [:write, :utf8])

results =
  pool
  |> Enum.with_index(1)
  |> Enum.map(fn {entry, i} ->
    r = Drill.rollout(sim, entry, attacker, defender, horizon: horizon)
    IO.write(io, Jason.encode!(%{id: entry.id, start: entry.summary, contact: r.contact?, converted: r.converted?, damage: r.damage, fair: r.fair, chain: %{openings: r.chain.openings, mean_connected_aerials: r.chain.mean_connected_aerials}}) <> "\n")
    if rem(i, 25) == 0, do: Output.puts("  #{i}/#{length(pool)} rollouts")
    r
  end)

File.close(io)
agg = Drill.aggregate(results)
ms = System.monotonic_time(:millisecond) - t1
Output.puts("rollouts: #{length(results)} × #{horizon} frames in #{div(ms, 1000)} s (#{Float.round(length(results) * horizon / max(ms, 1) * 1000, 0)} fps)")
Output.puts("BASELINE #{defender_kind}: contact rate #{Float.round(agg.contact_rate, 3)}  conversion rate #{Float.round(agg.conversion_rate, 3)}  conversion|contact #{Float.round(agg.conversion_given_contact, 3)}  mean damage #{Float.round(agg.mean_damage, 1)}  mean connected aerials/opening #{Float.round(agg.mean_connected_aerials, 2)}")
Output.puts("outcomes: #{inspect(agg.outcome_kinds)}")
File.write!(Path.join(out, "summary.json"), Jason.encode!(Map.merge(agg, %{policy: policy, defender: defender_kind, horizon: horizon, seed: seed, starts: length(pool)}), pretty: true))
SimPort.stop(sim)
Output.success("wrote #{out}/{pool.jsonl,rollouts.jsonl,summary.json}")
