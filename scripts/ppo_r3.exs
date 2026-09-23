# RL_ON_PRIOR R3 — head-only PPO on the frozen trunk, Fox ditto in the sim.
#
#   devenv shell -- mix run scripts/ppo_r3.exs --policy CKPT --critic eval_runs/0923_r2/refit/critic_best.bin \
#     --envs 64 --frames 600 --iters 200 --out eval_runs/0923_ppo/v1
#
# Each iteration: roll `envs` × `frames` with the CURRENT head on port 1 and the
# FROZEN prior on port 2; value the states with the critic; GAE; `--epochs`
# passes of minibatch PPO on the head only; push the new head into the actor.
# The trunk never trains, so the update is a small MLP over stored features.
#
# Guards: the KL to the frozen prior is both a loss term and a tripwire
# (`--kl-stop`), and every `--save-every` iterations the head is written out so
# an overnight run always leaves usable artifacts.

alias ExPhil.Agents.Agent
alias ExPhil.Sim.{Env, PPO}
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string, critic: :string, out: :string, envs: :integer, frames: :integer,
      iters: :integer, epochs: :integer, minibatch: :integer, lr: :float, clip: :float,
      kl_coef: :float, ent_coef: :float, gamma: :float, lambda: :float, seed: :integer,
      stage: :string, save_every: :integer, kl_stop: :float, vf_lr: :float
    ]
  )

policy = opts[:policy] || raise("--policy required")
out = opts[:out] || raise("--out required")
n = opts[:envs] || 64
frames = opts[:frames] || 600
iters = opts[:iters] || 100
epochs = opts[:epochs] || 2
mb = opts[:minibatch] || 4096
lr = opts[:lr] || 3.0e-5
gamma = opts[:gamma] || 0.995
lambda = opts[:lambda] || 0.95
kl_stop = opts[:kl_stop] || 0.5
save_every = opts[:save_every] || 10
stage = opts[:stage] || "final_destination"
seed = opts[:seed] || 31
File.mkdir_p!(out)

Output.banner("R3 — head-only PPO on the frozen trunk")
Output.config([
  {"Policy (prior)", policy}, {"Critic init", opts[:critic] || "fresh"},
  {"Envs × frames", "#{n} × #{frames}"}, {"Iterations", iters},
  {"PPO epochs/iter", epochs}, {"Minibatch", mb}, {"LR", lr},
  {"gamma/lambda", "#{gamma}/#{lambda}"}, {"KL coef", opts[:kl_coef] || 0.05},
  {"KL stop", kl_stop}, {"Out", out}
])

players = [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]
{:ok, sim} = Env.start(:nif, stage: stage, players: players, batch_size: n, seed: seed)

agent_opts = [policy_path: policy, deterministic: false, temperature: 1.0, af_convention: :parsed,
              frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true]
Output.puts("⏳ loading actor + frozen opponent (JIT on first use)…")
{:ok, actor} = Agent.start_link(agent_opts)
{:ok, opponent} = Agent.start_link(agent_opts)
{:ok, _} = Agent.warmup(actor)
{:ok, _} = Agent.warmup(opponent)

# ---- head params: prior (frozen anchor) and theta (trained)
prior_map = PPO.head_params(policy)
d = Nx.axis_size(prior_map["ar_residual_proj"]["kernel"], 0)
model = PPO.head_model(d)
{_init, predict_fn} = Axon.build(model, mode: :inference)
# params travel as plain maps (see PPO.predict/3)
prior_params = PPO.to_backend(prior_map)
theta = PPO.to_backend(prior_map)

{opt_init, opt_update} = Polaris.Optimizers.adam(learning_rate: lr)
opt_state = opt_init.(theta)

# ---- critic on the same features
critic_cfg =
  case opts[:critic] do
    nil -> %{hidden: 256, params: nil}
    path -> path |> File.read!() |> :erlang.binary_to_term()
  end

vhidden = Map.get(critic_cfg, :hidden, 256)
vmodel =
  Axon.input("features", shape: {nil, d})
  |> Axon.dense(vhidden, activation: :relu, name: "v1")
  |> Axon.dense(vhidden, activation: :relu, name: "v2")
  |> Axon.dense(1, name: "value")

{vinit, vpredict} = Axon.build(vmodel, mode: :inference)

vparams =
  case Map.get(critic_cfg, :params) do
    nil -> vinit.(Nx.template({1, d}, :f32), Axon.ModelState.empty())
    p -> PPO.to_backend(p)
  end

{vopt_init, vopt_update} = Polaris.Optimizers.adam(learning_rate: opts[:vf_lr] || 3.0e-4)
vopt_state = vopt_init.(vparams)

Output.puts("head d=#{d}; #{map_size(prior_map)} ar_* layers; critic hidden #{vhidden}#{if Map.get(critic_cfg, :params), do: " (loaded, EV #{Float.round(Map.get(critic_cfg, :ev, 0.0), 3)})", else: ""}")

t0 = System.monotonic_time(:millisecond)
log = []

{_theta, _opt, _v, _vopt, log} =
  Enum.reduce_while(1..iters, {theta, opt_state, vparams, vopt_state, log}, fn iter, {theta, opt_state, vparams, vopt_state, log} ->
    ti = System.monotonic_time(:millisecond)
    {:ok, _} = Env.reinit(sim, %{stage: stage, players: players, length: 256, seed: seed * 1000 + iter})
    roll = PPO.collect(sim, actor, opponent, n, frames)
    flat = PPO.flatten(roll) |> PPO.to_backend()
    total = n * frames

    # values, GAE, normalized advantages
    values_flat = vpredict.(vparams, %{"features" => flat.features}) |> Nx.squeeze(axes: [1])
    values = Nx.reshape(values_flat, {n, frames})
    {adv, returns} = PPO.gae(PPO.to_backend(roll.rewards), values, PPO.to_backend(roll.dones), gamma, lambda)
    adv_flat = Nx.reshape(adv, {total})
    adv_norm = Nx.divide(Nx.subtract(adv_flat, Nx.mean(adv_flat)), Nx.add(Nx.standard_deviation(adv_flat), 1.0e-8))
    ret_flat = Nx.reshape(returns, {total})

    # logp under the collecting params and the prior's logits (the KL anchor)
    old_logits = PPO.logits_of(predict_fn, theta, flat.features, flat.action)
    logp_old = PPO.logp(old_logits, flat.action)
    prior_logits = PPO.logits_of(predict_fn, prior_params, flat.features, flat.action)

    nb = max(1, div(total, mb))

    {theta, opt_state, ms_acc} =
      Enum.reduce(1..epochs, {theta, opt_state, []}, fn _ep, {theta, opt_state, acc} ->
        order = Nx.tensor(Enum.shuffle(0..(total - 1)))

        Enum.reduce(0..(nb - 1), {theta, opt_state, acc}, fn b, {theta, opt_state, acc} ->
          idx = Nx.slice_along_axis(order, b * mb, mb, axis: 0)

          batch = %{
            features: Nx.take(flat.features, idx),
            action: Map.new(flat.action, fn {k, v} -> {k, Nx.take(v, idx)} end),
            advantages: Nx.take(adv_norm, idx),
            logp_old: Nx.take(logp_old, idx),
            prior_logits: prior_logits |> Tuple.to_list() |> Enum.map(&Nx.take(&1, idx)) |> List.to_tuple()
          }

          {theta, opt_state, m} =
            PPO.update_step(predict_fn, theta, opt_state, opt_update, batch,
              clip: opts[:clip] || 0.2, kl_coef: opts[:kl_coef] || 0.05, ent_coef: opts[:ent_coef] || 0.001)

          {theta, opt_state, [m | acc]}
        end)
      end)

    # critic: one pass of MSE to the GAE returns
    {vparams, vopt_state, vloss} =
      Enum.reduce(0..(nb - 1), {vparams, vopt_state, 0.0}, fn b, {vp, vo, _} ->
        idx = Nx.slice_along_axis(Nx.tensor(Enum.shuffle(0..(total - 1))), b * mb, mb, axis: 0)
        xb = Nx.take(flat.features, idx)
        yb = Nx.take(ret_flat, idx)

        {l, g} =
          Nx.Defn.value_and_grad(vp, fn p ->
            pred = vpredict.(p, %{"features" => xb}) |> Nx.squeeze(axes: [1])
            Nx.mean(Nx.pow(Nx.subtract(pred, yb), 2))
          end)

        {u, vo} = vopt_update.(g, vo, vp)
        {Polaris.Updates.apply_updates(vp, u), vo, Nx.to_number(l)}
      end)

    # push the trained head into the actor so the next rollout is on-policy
    :ok = Agent.put_head_params(actor, theta)

    m = Enum.reduce(ms_acc, %{}, fn m, acc -> Map.merge(acc, m, fn _k, a, b -> a + b end) end)
    cnt = max(1, length(ms_acc))
    mean = fn k -> Map.get(m, k, 0.0) / cnt end
    reward_sum = Nx.sum(roll.rewards) |> Nx.to_number()
    stocks = Nx.sum(Nx.as_type(Nx.greater(Nx.abs(roll.rewards), 0.5), :s64)) |> Nx.to_number()
    ev = 1.0 - (Nx.variance(Nx.subtract(ret_flat, values_flat)) |> Nx.to_number()) / max(1.0e-9, Nx.variance(ret_flat) |> Nx.to_number())
    ms = System.monotonic_time(:millisecond) - ti

    row = %{iter: iter, reward: reward_sum / n, stocks: stocks, kl: mean.(:kl), entropy: mean.(:entropy),
            clip_frac: mean.(:clip_frac), pg: mean.(:pg), vloss: vloss, value_ev: ev, ms: ms}

    Output.puts(
      "iter #{String.pad_leading(Integer.to_string(iter), 3)}  reward/env #{:io_lib.format("~7.3f", [row.reward])}  stocks #{String.pad_leading(Integer.to_string(stocks), 3)}  " <>
        "KL #{:io_lib.format("~6.4f", [row.kl])}  H #{:io_lib.format("~5.2f", [row.entropy])}  clip #{:io_lib.format("~4.2f", [row.clip_frac])}  " <>
        "vEV #{:io_lib.format("~5.2f", [ev])}  (#{ms} ms)"
    )

    log = [row | log]

    if rem(iter, save_every) == 0 or iter == iters do
      File.write!(Path.join(out, "head_iter#{iter}.bin"), :erlang.term_to_binary(%{
        ar: ExPhil.Training.PPO.to_binary_backend(theta), iter: iter, policy: policy, d: d
      }))
      File.write!(Path.join(out, "log.json"), Jason.encode!(%{config: %{envs: n, frames: frames, iters: iters, lr: lr, gamma: gamma, lambda: lambda, policy: policy}, log: Enum.reverse(log)}, pretty: true))
    end

    cond do
      row.kl > kl_stop ->
        Output.error("KL to the prior #{Float.round(row.kl, 3)} > #{kl_stop} — stopping (the head is drifting off the imitation manifold)")
        {:halt, {theta, opt_state, vparams, vopt_state, log}}

      true ->
        {:cont, {theta, opt_state, vparams, vopt_state, log}}
    end
  end)

ms = System.monotonic_time(:millisecond) - t0
File.write!(Path.join(out, "log.json"), Jason.encode!(%{config: %{envs: n, frames: frames, iters: iters, lr: lr, gamma: gamma, lambda: lambda, policy: policy}, log: Enum.reverse(log)}, pretty: true))
Output.success("#{length(log)} iterations in #{Float.round(ms / 1000, 1)} s → #{out}/log.json")
Env.stop(sim)
