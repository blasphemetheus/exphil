# Where does a policy-in-the-loop sim frame spend its time? (2026-09-21, before
# any long run). Measures, with the real V3.1 agent:
#   1. Agent.get_controller latency, batch 1 (the 9 ms suspect)
#   2. K agents in parallel processes -> decisions/s (does inference parallelize?)
#   3. SimPort.step round trip at env batch 1 vs 8 (JSON growth)
#   4. restore by blob vs by cached state_id
#
#   devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.5 EXPHIL_EXLA_PRECISION=highest \
#     mix run scripts/sim_profile.exs --policy checkpoints/.../model_best_policy.bin

alias ExPhil.Agents.Agent
alias ExPhil.Bridge.SimPort
alias ExPhil.Sim.Drill
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [policy: :string, agents: :integer])
policy = opts[:policy] || raise("--policy required")
agent_opts = [policy_path: policy, deterministic: false, temperature: 1.0, af_convention: :parsed, frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true]

pct = fn xs, p -> s = Enum.sort(xs); Enum.at(s, min(length(s) - 1, trunc(p * length(s)))) end
stats = fn us -> "mean #{div(Enum.sum(us), max(length(us), 1))} us  p50 #{pct.(us, 0.5)}  p95 #{pct.(us, 0.95)}  max #{Enum.max(us)}" end

Output.banner("Sim loop profile")
{:ok, sim} = SimPort.start_link(stage: "final_destination", players: [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}], length: 256, seed: 1)
for _ <- 1..130, do: {:ok, _, _} = SimPort.step(sim)
{:ok, [gs]} = SimPort.frames(sim)

# 1. single agent latency
{:ok, a} = Agent.start_link(agent_opts)
{:ok, _} = Agent.warmup(a)
for _ <- 1..20, do: {:ok, _} = Agent.get_controller(a, gs, player_port: 1)
us = for i <- 1..200 do
  g = %{gs | frame: gs.frame + i}
  {t, {:ok, _}} = :timer.tc(fn -> Agent.get_controller(a, g, player_port: 1) end)
  t
end
Output.puts("1. Agent.get_controller (batch 1): #{stats.(us)}")

# 2. K agents in parallel
for k <- [1, 2, 4, 8] |> Enum.filter(&(&1 <= (opts[:agents] || 8))) do
  agents = for _ <- 1..k, do: (fn -> {:ok, p} = Agent.start_link(agent_opts); {:ok, _} = Agent.warmup(p); p end).()
  for p <- agents, _ <- 1..10, do: {:ok, _} = Agent.get_controller(p, gs, player_port: 1)
  n = 100
  {t, _} = :timer.tc(fn ->
    for i <- 1..n do
      agents
      |> Task.async_stream(fn p -> {:ok, _} = Agent.get_controller(p, %{gs | frame: gs.frame + i}, player_port: 1) end, max_concurrency: k, timeout: 30_000, ordered: false)
      |> Stream.run()
    end
  end)
  Output.puts("2. #{k} agents in parallel: #{Float.round(k * n / (t / 1.0e6), 0)} decisions/s (#{div(t, n)} us per frame of #{k} decisions)")
  Enum.each(agents, &GenServer.stop/1)
end

# 3. sim step round trip at batch 1 vs 8
for b <- [1, 8] do
  {:ok, s} = SimPort.start_link(stage: "final_destination", players: [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}], batch_size: b, length: 256, seed: 2)
  for _ <- 1..20, do: {:ok, _, _} = SimPort.step(s)
  ctrl = for _ <- 1..b, do: [Drill.neutral(), Drill.neutral()]
  us = for _ <- 1..200 do
    {t, {:ok, _, _}} = :timer.tc(fn -> SimPort.step(s, ctrl) end)
    t
  end
  Output.puts("3. SimPort.step batch #{b}: #{stats.(us)}  (#{Float.round(b * 1.0e6 / (Enum.sum(us) / length(us)), 0)} env-frames/s)")
  SimPort.stop(s)
end

# 4. restore by blob vs cached id
{:ok, blob, sid} = SimPort.save(sim, 0, keep: true)
us_blob = for _ <- 1..50 do
  {t, {:ok, _}} = :timer.tc(fn -> SimPort.restore(sim, 0, blob) end)
  t
end
us_id = for _ <- 1..50 do
  {t, {:ok, _}} = :timer.tc(fn -> SimPort.restore(sim, 0, {:id, sid}) end)
  t
end
Output.puts("4. restore by blob (#{byte_size(blob)} bytes): #{stats.(us_blob)}")
Output.puts("4. restore by cached id: #{stats.(us_id)}")
SimPort.stop(sim)
