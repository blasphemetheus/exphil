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
alias ExPhil.Training.SelfPlay.OpponentPool

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string, critic: :string, out: :string, envs: :integer, frames: :integer,
      iters: :integer, epochs: :integer, minibatch: :integer, lr: :float, clip: :float,
      kl_coef: :float, ent_coef: :float, gamma: :float, lambda: :float, seed: :integer,
      stage: :string, save_every: :integer, kl_stop: :float, vf_lr: :float,
      character: :string, opponent_heads: :string
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
# ditto matchup; validated here so an unknown name fails before the sim boots
character = opts[:character] || "fox"
_ = ExPhil.Bridge.SimBatch.character_id(character)
for {name, value} <- [envs: n, frames: frames, iters: iters, epochs: epochs, minibatch: mb, save_every: save_every] do
  if value < 1, do: raise(ArgumentError, "--#{name} must be positive")
end
:rand.seed(:exsss, {seed, seed + 1, seed + 2})
File.mkdir_p!(out)
if File.exists?(Path.join(out, "log.json")), do: raise("Output already contains a run: #{out}")

Output.banner("R3 — head-only PPO on the frozen trunk")
Output.config([
  {"Policy (prior)", policy}, {"Critic init", opts[:critic] || "fresh"},
  {"Envs × frames", "#{n} × #{frames}"}, {"Iterations", iters},
  {"PPO epochs/iter", epochs}, {"Minibatch", mb}, {"LR", lr},
  {"gamma/lambda", "#{gamma}/#{lambda}"}, {"KL coef", opts[:kl_coef] || 0.05},
  {"KL stop", kl_stop}, {"Out", out},
  {"Character (ditto)", character}, {"Stage", stage}, {"Seed", seed}
])

players = [%{character: character, costume: 1}, %{character: character, costume: 0}]
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
# Optional same-trunk opponent pool. Keep the imitation prior as an opponent
# AND as the unchanged KL anchor. A pool of Mewtwo heads is not a cross-character
# league; every loaded head must belong to this exact prior and character.
{:ok, opponent_pool} = OpponentPool.new(config: %{current: 0.0, historical: 0.0, cpu: 0.0, random: 1.0}, max_historical: 100)
opponent_pool = opponent_pool |> OpponentPool.set_current(prior_map) |> OpponentPool.snapshot("prior")
opponent_paths = if opts[:opponent_heads], do: opts[:opponent_heads] |> String.split(",", trim: true) |> Enum.uniq(), else: []
if length(opponent_paths) > 98, do: raise("too many opponent heads")
{opponent_pool, opponent_manifest} = Enum.reduce(opponent_paths, {opponent_pool, []}, fn path, {pool, manifest} ->
  saved = path |> File.read!() |> :erlang.binary_to_term()
  unless saved[:policy] && Path.expand(saved.policy) == Path.expand(policy) && saved[:character] == character,
    do: raise("Opponent #{path} has a different prior or character")
  unless is_map(saved[:ar]) && Enum.sort(Map.keys(saved.ar)) == Enum.sort(Map.keys(prior_map)),
    do: raise("Opponent #{path} has incompatible head parameters")
  for {layer, params} <- prior_map, {key, tensor} <- params do
    unless Nx.shape(saved.ar[layer][key]) == Nx.shape(tensor), do: raise("Opponent shape mismatch: #{path} #{layer}/#{key}")
  end
  item = %{path: path, version: path, iteration: saved.iter,
    sha256: Base.encode16(:crypto.hash(:sha256, File.read!(path)), case: :lower)}
  {pool |> OpponentPool.set_current(saved.ar) |> OpponentPool.snapshot(path), manifest ++ [item]}
end)
File.write!(Path.join(out, "opponents.json"), Jason.encode!(%{sampling: "uniform over prior plus listed frozen heads; one opponent per rollout batch",
  prior: policy, prior_sha256: Base.encode16(:crypto.hash(:sha256, File.read!(policy)), case: :lower),
  character: character, heads: opponent_manifest}, pretty: true))
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

# A critic is a value head on ONE trunk's features. Reusing Fox's on Mewtwo is
# silently wrong wherever the widths happen to match, so refuse it by name.
case Map.get(critic_cfg, :character) do
  nil -> if opts[:critic], do: Output.warning("--critic #{opts[:critic]} carries no character provenance; cannot verify it matches #{character}")
  ^character -> :ok
  other -> raise ArgumentError, "critic was fit on #{other} but this run is #{character}; fit a new critic"
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
    # Preserve historical fixed-prior RNG behavior when no pool was requested.
    opponent_version = if opponent_paths == [] do
      "prior"
    else
      {_, selected} = OpponentPool.sample(opponent_pool)
      :ok = Agent.put_head_params(opponent, PPO.to_backend(selected.params))
      selected.version
    end
    if opponent_paths != [], do: Output.puts("iteration #{iter} opponent: #{opponent_version}")
    {:ok, _} = Env.reinit(sim, %{stage: stage, players: players, length: 256, seed: seed * 1000 + iter})
    roll = PPO.collect(sim, actor, opponent, n, frames)
    flat = PPO.flatten(roll) |> PPO.to_backend()
    total = n * frames

    # values, GAE, normalized advantages
    values_flat = vpredict.(vparams, %{"features" => flat.features}) |> Nx.squeeze(axes: [1])
    values = Nx.reshape(values_flat, {n, frames})
    bootstrap = vpredict.(vparams, %{"features" => PPO.to_backend(roll.bootstrap_features)}) |> Nx.squeeze(axes: [1])
    {adv, returns} = PPO.gae(PPO.to_backend(roll.rewards), values, PPO.to_backend(roll.dones), gamma, lambda, bootstrap)
    adv_flat = Nx.reshape(adv, {total})
    adv_norm = Nx.divide(Nx.subtract(adv_flat, Nx.mean(adv_flat)), Nx.add(Nx.standard_deviation(adv_flat), 1.0e-8))
    ret_flat = Nx.reshape(returns, {total})

    # logp under the collecting params and the prior's logits (the KL anchor)
    old_logits = PPO.logits_of(predict_fn, theta, flat.features, flat.action)
    logp_old = PPO.logp(old_logits, flat.action)
    prior_logits = PPO.logits_of(predict_fn, prior_params, flat.features, flat.action)

    nb = div(total + mb - 1, mb)

    {theta, opt_state, ms_acc} =
      Enum.reduce(1..epochs, {theta, opt_state, []}, fn _ep, {theta, opt_state, acc} ->
        order = Nx.tensor(Enum.shuffle(0..(total - 1)))

        Enum.reduce(0..(nb - 1), {theta, opt_state, acc}, fn b, {theta, opt_state, acc} ->
          idx = Nx.slice_along_axis(order, b * mb, min(mb, total - b * mb), axis: 0)

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
    vorder = Nx.tensor(Enum.shuffle(0..(total - 1)))
    {vparams, vopt_state, vloss} =
      Enum.reduce(0..(nb - 1), {vparams, vopt_state, 0.0}, fn b, {vp, vo, _} ->
        idx = Nx.slice_along_axis(vorder, b * mb, min(mb, total - b * mb), axis: 0)
        xb = Nx.take(flat.features, idx)
        yb = Nx.take(ret_flat, idx)

        {l, g} =
          Nx.Defn.jit(fn p, x, y ->
            Nx.Defn.value_and_grad(p, fn pp ->
              pred = vpredict.(pp, %{"features" => x}) |> Nx.squeeze(axes: [1])
              Nx.mean(Nx.pow(Nx.subtract(pred, y), 2))
            end)
          end).(vp, xb, yb)

        {u, vo} = vopt_update.(g, vo, vp)
        {PPO.apply_updates(vp, u), vo, Nx.to_number(l)}
      end)

    # push the trained head into the actor so the next rollout is on-policy
    :ok = Agent.put_head_params(actor, theta)

    m = Enum.reduce(ms_acc, %{}, fn m, acc -> Map.merge(acc, m, fn _k, a, b -> a + b end) end)
    cnt = max(1, length(ms_acc))
    mean = fn k -> Map.get(m, k, 0.0) / cnt end
    final_logits = PPO.logits_of(predict_fn, theta, flat.features, flat.action)
    final_kl = PPO.kl(prior_logits, final_logits) |> Nx.mean() |> Nx.to_number()
    reward_sum = Nx.sum(roll.rewards) |> Nx.to_number()
    stocks = Nx.sum(Nx.as_type(Nx.greater(Nx.abs(roll.rewards), 0.5), :s64)) |> Nx.to_number()
    ev = 1.0 - (Nx.variance(Nx.subtract(ret_flat, values_flat)) |> Nx.to_number()) / max(1.0e-9, Nx.variance(ret_flat) |> Nx.to_number())
    ms = System.monotonic_time(:millisecond) - ti

    row = %{iter: iter, opponent: opponent_version, reward: reward_sum / n, stocks: stocks, kl: final_kl, update_kl: mean.(:kl), entropy: mean.(:entropy),
            clip_frac: mean.(:clip_frac), pg: mean.(:pg), vloss: vloss, value_ev: ev, ms: ms}

    Output.puts(
      "iter #{String.pad_leading(Integer.to_string(iter), 3)}  reward/env #{:io_lib.format("~7.3f", [row.reward])}  stocks #{String.pad_leading(Integer.to_string(stocks), 3)}  " <>
        "KL #{:io_lib.format("~6.4f", [row.kl])}  H #{:io_lib.format("~5.2f", [row.entropy])}  clip #{:io_lib.format("~4.2f", [row.clip_frac])}  " <>
        "vEV #{:io_lib.format("~5.2f", [ev])}  (#{ms} ms)"
    )

    log = [row | log]
    File.write!(Path.join(out, "metrics.jsonl"), Jason.encode!(row) <> "\n", [:append])

    stop_requested = File.exists?(Path.join(out, "STOP"))
    if iter == 1 or rem(iter, save_every) == 0 or iter == iters or row.kl > kl_stop or stop_requested do
      File.write!(Path.join(out, "head_iter#{iter}.bin"), :erlang.term_to_binary(%{
        ar: ExPhil.Training.PPO.to_binary_backend(theta), iter: iter, policy: policy, d: d,
        character: character, stage: stage, seed: seed
      }))
      File.write!(Path.join(out, "trainer_iter#{iter}.bin"), :erlang.term_to_binary(%{
        theta: PPO.to_backend(theta, Nx.BinaryBackend), opt_state: PPO.to_backend(opt_state, Nx.BinaryBackend),
        critic: PPO.to_backend(vparams, Nx.BinaryBackend), critic_opt_state: PPO.to_backend(vopt_state, Nx.BinaryBackend),
        iter: iter, policy: policy, opts: opts, rng: :rand.export_seed()
      }))
      File.write!(Path.join(out, "log.json"), Jason.encode!(%{config: %{envs: n, frames: frames, iters: iters, lr: lr, gamma: gamma, lambda: lambda, policy: policy,
                character: character, stage: stage, seed: seed, kl_coef: opts[:kl_coef] || 0.05, kl_stop: kl_stop,
                clip: opts[:clip] || 0.2, ent_coef: opts[:ent_coef] || 0.001, minibatch: mb, epochs: epochs,
                critic: opts[:critic] || "fresh"}, log: Enum.reverse(log)}, pretty: true))
    end

    cond do
      stop_requested ->
        Output.puts("STOP file found; saved iteration #{iter} and stopping cleanly")
        {:halt, {theta, opt_state, vparams, vopt_state, log}}

      row.kl > kl_stop ->
        Output.error("KL to the prior #{Float.round(row.kl, 3)} > #{kl_stop} — stopping (the head is drifting off the imitation manifold)")
        {:halt, {theta, opt_state, vparams, vopt_state, log}}

      true ->
        {:cont, {theta, opt_state, vparams, vopt_state, log}}
    end
  end)

ms = System.monotonic_time(:millisecond) - t0
File.write!(Path.join(out, "log.json"), Jason.encode!(%{config: %{envs: n, frames: frames, iters: iters, lr: lr, gamma: gamma, lambda: lambda, policy: policy,
                character: character, stage: stage, seed: seed, kl_coef: opts[:kl_coef] || 0.05, kl_stop: kl_stop,
                clip: opts[:clip] || 0.2, ent_coef: opts[:ent_coef] || 0.001, minibatch: mb, epochs: epochs,
                critic: opts[:critic] || "fresh"}, log: Enum.reverse(log)}, pretty: true))
Output.success("#{length(log)} iterations in #{Float.round(ms / 1000, 1)} s → #{out}/log.json")
Env.stop(sim)
