# SIM_INTEGRATION.md step 7: search-as-teacher v0 (random shooting) over a
# drill start pool. Reports the ORACLE's fair-conversion / contact rates on
# the same starts as the prior's baseline (scripts/sim_drill.exs) and writes
# the winning programs as labels.
#
#   devenv shell -- env EXPHIL_GPU=0 mix run scripts/sim_search.exs \
#     --pool eval_runs/0921_sim_drill/self_n200/pool.jsonl --defender idle \
#     --n 64 --horizon 90 --out eval_runs/0921_sim_search/idle_n200 [--seed 1] [--starts 200]
#
#   # defender = the frozen prior (needs a policy; builds a fresh play pool in-process)
#   devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.3 EXPHIL_EXLA_PRECISION=highest \
#     mix run scripts/sim_search.exs --policy checkpoints/.../model_best_policy.bin --defender self \
#     --starts 100 --n 32 --horizon 90 --out eval_runs/0921_sim_search/self_n100

alias ExPhil.Agents.Agent
alias ExPhil.Sim.Env
alias ExPhil.Sim.{Drill, Search}
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [pool: :string, policy: :string, defender: :string, n: :integer, horizon: :integer, out: :string, seed: :integer, starts: :integer, max_hold: :integer, batched: :boolean, backend: :string, pool_term: :string, episodes_out: :string])
defender_kind = opts[:defender] || "idle"
n = opts[:n] || 64
horizon = opts[:horizon] || 90
out = opts[:out] || raise("--out DIR required")
seed = opts[:seed] || 1
File.mkdir_p!(out)

Output.banner("Search-as-teacher v0 (step 7)")
Output.config([{"Pool", opts[:pool] || "(built from play)"}, {"Defender", defender_kind}, {"Candidates/start", n}, {"Horizon", horizon}, {"Seed", seed}, {"Out", out}])

batched = Keyword.get(opts, :batched, true)
backend = String.to_atom(opts[:backend] || "nif")
{:ok, sim} = Env.start(backend, stage: "final_destination", players: [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}], length: 256, seed: seed)
# the batched pool builder and the batched search both need the sim at batch n from the start
if batched, do: {:ok, _} = Env.reinit(sim, %{stage: "final_destination", players: [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}], batch_size: n, length: 256, seed: seed})

{pool, defender} =
  case defender_kind do
    "idle" ->
      path = opts[:pool] || opts[:pool_term] || raise("--pool or --pool-term required for --defender idle")
      pool =
        if opts[:pool_term] do
          Drill.pool_from_file(opts[:pool_term])
        else
          path |> File.stream!() |> Stream.map(&Jason.decode!/1) |> Enum.map(fn r -> %{id: r["id"], blob: Base.decode64!(r["blob"]), frame: r["frame"], summary: r["summary"], history: []} end)
        end
      {pool, :idle}

    "self" ->
      policy = opts[:policy] || raise("--policy required for --defender self")
      agent_opts = [policy_path: policy, deterministic: false, temperature: 1.0, af_convention: :parsed, frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true]
      {:ok, a} = Agent.start_link(agent_opts); {:ok, _} = Agent.warmup(a)
      {:ok, d} = Agent.start_link(agent_opts); {:ok, _} = Agent.warmup(d)
      Output.step(1, 2, "Building play pool (#{opts[:starts] || 100})")
      pool = if batched, do: Drill.build_pool_from_play_batch(sim, a, d, opts[:starts] || 100, n, seed: seed, max_distance: 50.0), else: Drill.build_pool_from_play(sim, a, d, opts[:starts] || 100, seed: seed, max_distance: 50.0)
      Drill.pool_to_disk(pool, Path.join(out, "pool.jsonl"))
      {pool, d}
  end

Output.puts("pool: #{length(pool)} starts")
Output.step(2, 2, "Shooting #{n} × #{horizon} frames per start (#{if batched, do: "batched: #{n} envs", else: "sequential"})")
if batched, do: {:ok, _} = Env.reinit(sim, %{stage: "final_destination", players: [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}], batch_size: n, length: 256, seed: seed})
labels_io = File.open!(Path.join(out, "labels.jsonl"), [:write, :utf8])
episodes = :ets.new(:episodes, [:ordered_set, :public])
results_io = File.open!(Path.join(out, "results.jsonl"), [:write, :utf8])
t0 = System.monotonic_time(:millisecond)

results =
  pool
  |> Enum.with_index(1)
  |> Enum.map(fn {entry, i} ->
    shoot = if batched, do: &Search.shoot_batch/4, else: &Search.shoot/4
    r = shoot.(sim, entry, defender, n: n, horizon: horizon, seed: seed, max_hold: opts[:max_hold] || 12)
    b = r.best

    IO.write(results_io, Jason.encode!(%{id: entry.id, start: entry.summary, tried: r.tried, n_converted: r.n_converted, n_contact: r.n_contact, best: %{score: b.score, converted: b.converted?, contact: b.contact?, damage: b.damage, alive: b.alive?, program: b.program, aerials: b.chain.mean_connected_aerials, openings: Enum.map(b.openings, &Map.take(&1, [:frame, :opener, :family, :hits, :damage, :converted?])), outcomes: b.fair.outcomes}}) <> "\n")

    if b.converted? and b.alive? and opts[:episodes_out] do
      # Training episode (causal pairs: state[t] with the input issued from it): the warm
      # history the policy actually played (its own labels), then the oracle program.
      warm = Enum.map(entry.history, fn {gs, c1, _c2} -> %{game_state: gs, controller: c1, player_tag: nil} end)
      prog = Enum.zip(Enum.take(b.states, length(b.controllers)), b.controllers) |> Enum.map(fn {gs, c} -> %{game_state: gs, controller: c, player_tag: nil} end)
      :ets.insert(episodes, {entry.id, warm ++ prog})
    end

    if b.contact? and b.alive? do
      IO.write(labels_io, Jason.encode!(%{id: entry.id, converted: b.converted?, damage: b.damage, frames: Enum.zip(b.frames, [nil | b.controllers]) |> Enum.map(fn {f, c} -> %{frame: f.frame, p1: f.p1, p2: f.p2, ctrl: c && Search.controller_json(c)} end)}) <> "\n")
    end

    if rem(i, 10) == 0 do
      done = Enum.take(pool, i)
      el = System.monotonic_time(:millisecond) - t0
      Output.puts("  #{i}/#{length(pool)} starts (#{div(el, 1000)} s, #{Float.round(i * n * horizon / max(el, 1) * 1000, 0)} fps)")
      _ = done
    end

    r
  end)

File.close(labels_io)
File.close(results_io)

starts = length(results)
conv = Enum.count(results, & &1.converted_any?)
contact = Enum.count(results, & &1.contact_any?)
best_conv = Enum.count(results, &(&1.best.converted? and &1.best.alive?))
mean_dmg = results |> Enum.map(& &1.best.damage) |> then(&(if starts == 0, do: 0.0, else: Enum.sum(&1) / starts))
per_cand_conv = results |> Enum.map(& &1.n_converted) |> Enum.sum()
ms = System.monotonic_time(:millisecond) - t0

Output.puts("ORACLE #{defender_kind}: starts #{starts}  any-candidate converted #{Float.round(conv / max(starts, 1), 3)}  any-candidate contact #{Float.round(contact / max(starts, 1), 3)}  best converted+alive #{Float.round(best_conv / max(starts, 1), 3)}  mean best damage #{Float.round(mean_dmg, 1)}  candidate-level conversion #{Float.round(per_cand_conv / max(starts * n, 1), 4)}  (#{div(ms, 1000)} s)")

File.write!(Path.join(out, "summary.json"), Jason.encode!(%{defender: defender_kind, starts: starts, n: n, horizon: horizon, seed: seed, any_converted_rate: conv / max(starts, 1), any_contact_rate: contact / max(starts, 1), best_converted_alive_rate: best_conv / max(starts, 1), mean_best_damage: mean_dmg, candidate_conversion_rate: per_cand_conv / max(starts * n, 1), elapsed_ms: ms}, pretty: true))
Env.stop(sim)
Output.success("wrote #{out}/{results.jsonl,labels.jsonl,summary.json}")

if opts[:episodes_out] do
  lists = :ets.tab2list(episodes) |> Enum.map(&elem(&1, 1))
  File.write!(opts[:episodes_out], :erlang.term_to_binary(%{
    expert: "search_oracle_v0",
    exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
    action_delay: 0,
    label_convention: ExPhil.Data.LabelConvention.current(),
    frame_lists: lists,
    defender: defender_kind, n: n, horizon: horizon, seed: seed
  }, [:compressed]))
  Output.success("wrote #{length(lists)} oracle episodes (#{Enum.sum(Enum.map(lists, &length/1))} frames) to #{opts[:episodes_out]}")
end
