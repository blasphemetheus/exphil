# RL_ON_PRIOR R2 — critic on the frozen trunk.
#
#   devenv shell -- mix run scripts/critic_r2.exs --policy CKPT --envs 64 --frames 1800 \
#     --rounds 4 --gamma 0.995 --out eval_runs/0922_r2/v1
#
# Rolls the prior vs itself in the sim (`envs` games of `frames` frames per round,
# fresh games each round), records trunk features + reward v1 per frame from
# port 1's side, fits an MLP value head on rounds 1..R-1 and reports explained
# variance on round R (held-out games). Gate: EV > 0.3. Writes OUT/critic.bin
# (params + model config), OUT/r2.json (metrics), OUT/data_round<k>.bin.

alias ExPhil.Agents.Agent
alias ExPhil.Sim.{Critic, Env}
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [policy: :string, envs: :integer, frames: :integer, rounds: :integer, gamma: :float, out: :string, epochs: :integer, hidden: :integer, stage: :string, seed: :integer])
policy = opts[:policy] || raise("--policy required")
n = opts[:envs] || 64
frames = opts[:frames] || 1800
rounds = opts[:rounds] || 4
gamma = opts[:gamma] || 0.995
out = opts[:out] || raise("--out required")
stage = opts[:stage] || "final_destination"
File.mkdir_p!(out)

Output.banner("R2 — critic on the frozen trunk")
Output.config([{"Policy", policy}, {"Envs", n}, {"Frames", frames}, {"Rounds", rounds}, {"Gamma", gamma}, {"Stage", stage}, {"Out", out}])

players = [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]
{:ok, sim} = Env.start(:nif, stage: stage, players: players, batch_size: n, seed: opts[:seed] || 11)

agent_opts = [policy_path: policy, deterministic: false, temperature: 1.0, af_convention: :parsed, frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true]
Output.puts("⏳ loading two agents (JIT on first use)…")
{:ok, a1} = Agent.start_link(agent_opts)
{:ok, a2} = Agent.start_link(agent_opts)
{:ok, _} = Agent.warmup(a1)
{:ok, _} = Agent.warmup(a2)

t0 = System.monotonic_time(:millisecond)

data =
  for r <- 1..rounds do
    tr = System.monotonic_time(:millisecond)
    {:ok, _} = Env.reinit(sim, %{stage: stage, players: players, length: 256, seed: (opts[:seed] || 11) * 1000 + r})
    d = Critic.collect(sim, {a1, a2}, n, frames, on_frame: fn t -> if rem(t, 300) == 0, do: Output.puts("  round #{r}: frame #{t}/#{frames}") end)
    ret = Critic.returns(d.rewards, d.dones, gamma)
    ms = System.monotonic_time(:millisecond) - tr
    fps = round(n * frames / max(1, ms) * 1000)
    Output.puts("round #{r}: #{n} envs × #{frames} f in #{Float.round(ms / 1000, 1)} s (#{fps} env-frames/s); total reward mean #{Float.round(Enum.sum(d.total_reward) / n, 3)}, stocks events #{Nx.sum(Nx.abs(Nx.round(d.rewards))) |> Nx.to_number() |> round()}")
    File.write!(Path.join(out, "data_round#{r}.bin"), Nx.serialize(%{features: d.features, rewards: d.rewards, dones: d.dones, returns: ret}))
    %{features: d.features, returns: ret, d: d.d}
  end

flat = fn ds -> {Nx.concatenate(Enum.map(ds, &Nx.reshape(&1.features, {:auto, &1.d}))), Nx.concatenate(Enum.map(ds, &Nx.reshape(&1.returns, {:auto})))} end
{train_x, train_y} = flat.(Enum.take(data, rounds - 1))
{test_x, test_y} = flat.([List.last(data)])
Output.puts("train #{Nx.axis_size(train_x, 0)} states (d=#{hd(data).d}), held-out #{Nx.axis_size(test_x, 0)}; return mean #{Float.round(Nx.mean(train_y) |> Nx.to_number(), 3)} sd #{Float.round(Nx.standard_deviation(train_y) |> Nx.to_number(), 3)}")

fit = Critic.fit({train_x, train_y}, {test_x, test_y}, epochs: opts[:epochs] || 20, hidden: opts[:hidden] || 256)
for h <- fit.history, do: Output.puts("  epoch #{h.epoch}: held-out EV #{Float.round(h.ev_test, 3)}")

ms = System.monotonic_time(:millisecond) - t0
verdict = if fit.ev > 0.3, do: "R2 PASSED (EV > 0.3)", else: "R2 NOT PASSED (EV ≤ 0.3)"
Output.puts("held-out explained variance #{Float.round(fit.ev, 3)} (train #{Float.round(fit.ev_train, 3)}); MSE #{Float.round(fit.mse, 4)} vs mean-baseline #{Float.round(fit.baseline_mse, 4)} → #{verdict}")

File.write!(Path.join(out, "critic.bin"), Nx.serialize(%{params: ExPhil.Training.PPO.to_binary_backend(fit.params), d: hd(data).d, hidden: opts[:hidden] || 256, gamma: gamma, policy: policy}))
File.write!(Path.join(out, "r2.json"), Jason.encode!(%{ev: fit.ev, ev_train: fit.ev_train, mse: fit.mse, baseline_mse: fit.baseline_mse, history: fit.history, envs: n, frames: frames, rounds: rounds, gamma: gamma, d: hd(data).d, policy: policy, ms: ms, verdict: verdict}, pretty: true))
Output.success("#{verdict} → #{out}/r2.json (#{Float.round(ms / 1000, 1)} s)")
Env.stop(sim)
