defmodule ExPhil.Sim.Critic do
  @moduledoc """
  RL_ON_PRIOR **R2 — critic learns**: a value head on the FROZEN imitation trunk.

  The prior's batched agent computes trunk features every frame before its
  autoregressive heads sample (`Agent.batch_get_controllers/3` with
  `return_features: true`). This module rolls the prior against itself in
  the sim, records `(features, reward)` per env per frame from port 1's
  side, turns rewards into discounted returns, and fits a small MLP
  `features → value`. The R2 gate: explained variance on held-out envs
  > 0.3 against the per-state mean.

  Reward v1 (RL_ON_PRIOR design decision): stock differential + 0.01 ×
  damage differential, per frame, no shaping.

  What this buys: (a) the first learned value of a Melee state on our own
  trunk (the coach's "what the engine thinks" without 16 rollouts); (b) the
  baseline R3 PPO needs; (c) a check that the trunk features carry outcome
  information at all.
  """

  alias ExPhil.Agents.Agent
  alias ExPhil.Sim.{Drill, Env}

  @neutral Drill.neutral()

  @doc """
  Roll `n` envs (the sim's batch size) for `frames` frames from a fresh game
  (or from `:starts`, a list of `{:id, id}` / blobs to restore per env) with
  the prior on both ports. Returns `%{features: [n][frames] of f32 lists,
  rewards: [n][frames], dones: [n][frames], d: feature width}` plus per-env
  totals for sanity. Options: `:gamma` is not applied here (see `returns/3`).
  """
  def collect(sim, {a1, a2}, n, frames, opts \\ []) do
    for i <- 0..(n - 1) do
      case Keyword.get(opts, :starts) do
        nil -> :ok
        starts -> {:ok, _} = Env.restore(sim, i, Enum.at(starts, rem(i, length(starts))), frames: false)
      end
    end

    if Keyword.get(opts, :starts) == nil, do: {:ok, _} = Env.reset(sim)
    {:ok, _, _} = Env.observe(sim)
    for a <- [a1, a2], do: ensure_batch(a, n)
    {:ok, gs0s} = Env.frames(sim)
    on_frame = Keyword.get(opts, :on_frame, fn _ -> :ok end)

    {acc, _states} =
      Enum.reduce(1..frames, {[], gs0s}, fn t, {acc, states} ->
        {:ok, c1s, feats} = Agent.batch_get_controllers(a1, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1, return_features: true)
        {:ok, c2s} = Agent.batch_get_controllers(a2, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)

        case Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a || @neutral, b || @neutral] end)) do
          {:ok, nexts, terms} ->
            rewards = Enum.zip_with(states, nexts, &reward/2)
            dones = Enum.map(terms, fn term -> (term["done"] || 0) == 1 end)
            on_frame.(t)
            {[{feats, rewards, dones} | acc], nexts}

          {:error, reason} ->
            raise "sim step failed: #{inspect(reason)}"
        end
      end)

    steps = Enum.reverse(acc)
    d = steps |> hd() |> elem(0) |> Nx.axis_size(1)
    # features: stack per step [n, d] → [frames, n, d] → per env [n][frames][d]
    feats = steps |> Enum.map(&elem(&1, 0)) |> Nx.stack() |> Nx.transpose(axes: [1, 0, 2])
    rewards = steps |> Enum.map(&elem(&1, 1)) |> Nx.tensor(type: :f32) |> Nx.transpose()
    dones = steps |> Enum.map(fn {_, _, ds} -> Enum.map(ds, &if(&1, do: 1, else: 0)) end) |> Nx.tensor(type: :u8) |> Nx.transpose()

    %{features: feats, rewards: rewards, dones: dones, d: d, n: n, frames: frames, total_reward: Nx.sum(rewards, axes: [1]) |> Nx.to_flat_list()}
  end

  @doc "Reward v1 for port 1: stocks taken − lost + 0.01 × (damage dealt − taken), per frame."
  def reward(a, b) do
    p1a = a.players[1]
    p1b = b.players[1]
    p2a = a.players[2]
    p2b = b.players[2]
    taken = max(0, (p2a.stock || 0) - (p2b.stock || 0))
    lost = max(0, (p1a.stock || 0) - (p1b.stock || 0))
    dealt = max(0.0, p2b.percent - p2a.percent)
    took = max(0.0, p1b.percent - p1a.percent)
    taken - lost + 0.01 * (dealt - took)
  end

  @doc "Discounted returns per env ([n, T] rewards, [n, T] dones) with `gamma`; a done resets the tail."
  def returns(rewards, dones, gamma) do
    r = Nx.to_batched(rewards, 1) |> Enum.map(&Nx.to_flat_list/1)
    d = Nx.to_batched(dones, 1) |> Enum.map(&Nx.to_flat_list/1)

    Enum.zip(r, d)
    |> Enum.map(fn {rs, ds} ->
      {out, _} =
        Enum.zip(rs, ds)
        |> Enum.reverse()
        |> Enum.reduce({[], 0.0}, fn {rw, dn}, {acc, g} ->
          g = rw + if(dn == 1, do: 0.0, else: gamma * g)
          {[g | acc], g}
        end)

      out
    end)
    |> Nx.tensor(type: :f32)
  end

  @doc """
  Fit `features [N, d] → value`; `fit/3` validates and tests on the same set
  (kept for the smoke path). Prefer `fit/4`: `val` selects the epoch (early
  stopping), `test` is only ever measured, so the reported EV is honest.
  """
  def fit(train, test, opts \\ []), do: fit(train, test, test, opts)

  @doc """
  Fit with an explicit validation split. Options: `:hidden` (256), `:epochs`
  (20), `:batch` (1024), `:lr` (1e-3), `:weight_decay` (0 → Adam, else AdamW),
  `:dropout` (0). Returns `%{params, model, ev, ev_val, ev_train, best_epoch,
  mse, baseline_mse, history}` where `params` are the best-validation epoch's.
  """
  def fit({x, y}, {xv, yv}, {xt, yt}, opts) do
    hidden = Keyword.get(opts, :hidden, 256)
    epochs = Keyword.get(opts, :epochs, 20)
    batch = Keyword.get(opts, :batch, 1024)
    lr = Keyword.get(opts, :lr, 1.0e-3)
    wd = Keyword.get(opts, :weight_decay, 0.0)
    dropout = Keyword.get(opts, :dropout, 0.0)
    d = Nx.axis_size(x, 1)

    model =
      Axon.input("features", shape: {nil, d})
      |> Axon.dense(hidden, activation: :relu, name: "v1")
      |> then(&if(dropout > 0.0, do: Axon.dropout(&1, rate: dropout), else: &1))
      |> Axon.dense(hidden, activation: :relu, name: "v2")
      |> then(&if(dropout > 0.0, do: Axon.dropout(&1, rate: dropout), else: &1))
      |> Axon.dense(1, name: "value")

    {_init_fn, predict_fn} = Axon.build(model, mode: :inference)
    n = Nx.axis_size(x, 0)
    nb = div(n, batch)

    # one epoch of shuffled minibatches, as a stream the Axon loop consumes
    batches = fn ->
      perm = Nx.tensor(Enum.shuffle(0..(n - 1)))
      xs = Nx.take(x, perm)
      ys = Nx.take(y, perm)

      Stream.map(0..(nb - 1), fn i ->
        {%{"features" => Nx.slice_along_axis(xs, i * batch, batch, axis: 0)}, Nx.slice_along_axis(ys, i * batch, batch, axis: 0) |> Nx.new_axis(1)}
      end)
    end

    optimizer =
      if wd > 0.0,
        do: Polaris.Optimizers.adamw(learning_rate: lr, decay: wd),
        else: Polaris.Optimizers.adam(learning_rate: lr)

    # keep the best-validation epoch's params: the 09-23 run peaked at epoch 1
    # (EV 0.24) and decayed to 0.04 by epoch 60 — training EV 0.91 was memorization
    {_last, best, history} =
      Enum.reduce(1..epochs, {nil, nil, []}, fn ep, {params, best, hist} ->
        loop = Axon.Loop.trainer(model, :mean_squared_error, optimizer, log: 0)
        state = Axon.Loop.run(loop, batches.(), params || Axon.ModelState.empty(), epochs: 1, compiler: EXLA)

        params =
          case state do
            %Axon.ModelState{} -> state
            %{model_state: ms} -> ms
            %{step_state: %{model_state: ms}} -> ms
          end

        ev_v = explained_variance(predict_fn.(params, %{"features" => xv}) |> Nx.squeeze(axes: [1]), yv)
        best = if best == nil or ev_v > elem(best, 1), do: {params, ev_v, ep}, else: best
        {params, best, [%{epoch: ep, ev_val: ev_v} | hist]}
      end)

    {params, ev_val, best_epoch} = best
    pred_t = predict_fn.(params, %{"features" => xt}) |> Nx.squeeze(axes: [1])
    pred_tr = predict_fn.(params, %{"features" => x}) |> Nx.squeeze(axes: [1])

    %{
      params: params,
      model: model,
      ev: explained_variance(pred_t, yt),
      ev_val: ev_val,
      best_epoch: best_epoch,
      ev_train: explained_variance(pred_tr, y),
      mse: Nx.mean(Nx.pow(Nx.subtract(pred_t, yt), 2)) |> Nx.to_number(),
      baseline_mse: Nx.variance(yt) |> Nx.to_number(),
      history: Enum.reverse(history)
    }
  end

  @doc "1 − Var(y − ŷ) / Var(y): 0 = no better than the mean, 1 = perfect."
  def explained_variance(pred, y) do
    vy = Nx.variance(y) |> Nx.to_number()
    if vy <= 0.0, do: 0.0, else: 1.0 - (Nx.variance(Nx.subtract(y, pred)) |> Nx.to_number()) / vy
  end

  defp ensure_batch(agent, n) do
    case Agent.batch_reset_rows(agent, Enum.to_list(0..(n - 1))) do
      :ok -> :ok
      {:error, :batch_not_initialized} -> :ok = Agent.batch_init(agent, n)
      {:error, other} -> raise "batch reset failed: #{inspect(other)}"
    end
  end
end
