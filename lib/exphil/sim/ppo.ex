defmodule ExPhil.Sim.PPO do
  @moduledoc """
  RL_ON_PRIOR **R3 — PPO on the autoregressive heads, trunk frozen**.

  The imitation prior is a GRU trunk + a 6-component autoregressive head
  (buttons → main_x → main_y → c_x → c_y → shoulder, each conditioned on the
  sampled prefix through a residual stream). RL here trains ONLY the head:

    * the trunk stays exactly as imitation left it, so the policy keeps the
      representation the whole style program depends on, and the update never
      pays BPTT — the rollout already produced the per-frame trunk features;
    * the head graph is rebuilt standalone on `Axon.input("trunk")` with the
      teacher-forced inputs (`ExPhil.Networks.Policy.Heads`), so one forward
      over stored `(features, action)` gives the conditional logits of every
      component at once;
    * three parameter sets appear in the loss: `theta` (training), `old`
      (the params that collected the batch — PPO's ratio denominator) and
      `prior` (the frozen imitation head — the KL anchor that keeps the
      policy human-shaped, `RL_ON_PRIOR.md` design decision 1).

  Loss per sample:

      -min(r·A, clip(r, 1±eps)·A) + vf_coef·(V(s) − R)² + kl_coef·KL(prior ‖ theta) − ent_coef·H(theta)

  with `r = exp(logp_theta − logp_old)`, A from GAE on the critic's values,
  and the component KL/entropy summed over the six heads (Bernoulli for the
  8 buttons, categorical for the four stick axes and the shoulder).

  The value function is the R2 critic (`ExPhil.Sim.Critic`): an MLP on the
  same trunk features, updated online with the same minibatches.
  """

  alias ExPhil.Agents.Agent
  alias ExPhil.Networks.Policy.Heads
  alias ExPhil.Sim.{Drill, Env}

  @neutral Drill.neutral()
  @components [:buttons, :main_x, :main_y, :c_x, :c_y, :shoulder]

  # ------------------------------------------------------------------ models

  @doc """
  The autoregressive head as a standalone model over trunk features.
  Inputs: `"trunk"` `[b, d]` plus the teacher-forced `tf_*` of the SAME frame.
  Outputs the six logit tensors as an Axon container.
  """
  def head_model(d, opts \\ []) do
    Axon.input("trunk", shape: {nil, d})
    |> Heads.build_autoregressive_head(
      axis_buckets: Keyword.get(opts, :axis_buckets, 16),
      shoulder_buckets: Keyword.get(opts, :shoulder_buckets, 4),
      residual_size: Keyword.get(opts, :residual_size, 128),
      component_hidden: Keyword.get(opts, :component_hidden, 64)
    )
  end

  @doc "The `ar_*` layers of a policy checkpoint, as a plain params map."
  def head_params(policy_path) do
    {:ok, export} = ExPhil.Training.Checkpoint.load_policy(policy_path)

    data =
      case export.params do
        %Axon.ModelState{data: d} -> d
        %{data: d} when is_map(d) -> d
        m when is_map(m) -> m
      end

    Map.filter(data, fn {k, _} -> is_binary(k) and String.starts_with?(k, "ar_") end)
  end

  @doc "Wrap a plain params map as an Axon.ModelState the head model can run."
  def as_model_state(map), do: Axon.ModelState.new(map)

  @doc """
  Build the teacher-forced input map for a batch: trunk features plus the
  stored action as conditioning (`buttons` multi-hot f32, axes as indices).
  """
  def tf_inputs(features, action) do
    %{
      "trunk" => features,
      "tf_buttons" => Nx.as_type(action.buttons, :f32),
      "tf_main_x" => Nx.as_type(action.main_x, :s64),
      "tf_main_y" => Nx.as_type(action.main_y, :s64),
      "tf_c_x" => Nx.as_type(action.c_x, :s64),
      "tf_c_y" => Nx.as_type(action.c_y, :s64)
    }
  end

  # ------------------------------------------------------- log-probs / KL / H

  @doc """
  Log π(a|s), entropy and per-component logits for a batch, under `params`.
  `logits` is the container output `{buttons, main_x, main_y, c_x, c_y, shoulder}`.
  Buttons are 8 independent Bernoulli; the rest are categorical.
  """
  def logp(logits, action) do
    {b, mx, my, cx, cy, sh} = logits

    button_lp =
      Nx.multiply(Nx.as_type(action.buttons, :f32), Nx.log(Nx.sigmoid(b) |> clamp()))
      |> Nx.add(
        Nx.multiply(
          Nx.subtract(1.0, Nx.as_type(action.buttons, :f32)),
          Nx.log(Nx.subtract(1.0, Nx.sigmoid(b)) |> clamp())
        )
      )
      |> Nx.sum(axes: [1])

    cat_lp =
      [{mx, action.main_x}, {my, action.main_y}, {cx, action.c_x}, {cy, action.c_y}, {sh, action.shoulder}]
      |> Enum.map(fn {l, a} -> categorical_logp(l, a) end)
      |> Enum.reduce(&Nx.add/2)

    Nx.add(button_lp, cat_lp)
  end

  @doc "Entropy of the factorized head distribution, summed over components."
  def entropy(logits) do
    {b, mx, my, cx, cy, sh} = logits
    p = Nx.sigmoid(b) |> clamp()
    bern = Nx.negate(Nx.add(Nx.multiply(p, Nx.log(p)), Nx.multiply(Nx.subtract(1.0, p), Nx.log(clamp(Nx.subtract(1.0, p)))))) |> Nx.sum(axes: [1])

    [mx, my, cx, cy, sh]
    |> Enum.map(fn l ->
      lp = log_softmax(l)
      Nx.negate(Nx.sum(Nx.multiply(Nx.exp(lp), lp), axes: [1]))
    end)
    |> Enum.reduce(bern, &Nx.add/2)
  end

  @doc "KL(p ‖ q) between two head distributions on the same conditioning, per sample."
  def kl(logits_p, logits_q) do
    {bp, mxp, myp, cxp, cyp, shp} = logits_p
    {bq, mxq, myq, cxq, cyq, shq} = logits_q

    pp = Nx.sigmoid(bp) |> clamp()
    qq = Nx.sigmoid(bq) |> clamp()

    bern =
      Nx.add(
        Nx.multiply(pp, Nx.subtract(Nx.log(pp), Nx.log(qq))),
        Nx.multiply(Nx.subtract(1.0, pp), Nx.subtract(Nx.log(clamp(Nx.subtract(1.0, pp))), Nx.log(clamp(Nx.subtract(1.0, qq)))))
      )
      |> Nx.sum(axes: [1])

    Enum.zip([mxp, myp, cxp, cyp, shp], [mxq, myq, cxq, cyq, shq])
    |> Enum.map(fn {lp_l, lq_l} ->
      lp = log_softmax(lp_l)
      lq = log_softmax(lq_l)
      Nx.sum(Nx.multiply(Nx.exp(lp), Nx.subtract(lp, lq)), axes: [1])
    end)
    |> Enum.reduce(bern, &Nx.add/2)
  end

  defp categorical_logp(logits, idx) do
    lp = log_softmax(logits)
    Nx.take_along_axis(lp, Nx.new_axis(Nx.as_type(idx, :s64), 1), axis: 1) |> Nx.squeeze(axes: [1])
  end

  defp log_softmax(l) do
    m = Nx.reduce_max(l, axes: [1], keep_axes: true)
    z = Nx.subtract(l, m)
    Nx.subtract(z, Nx.log(Nx.sum(Nx.exp(z), axes: [1], keep_axes: true)))
  end

  defp clamp(t), do: Nx.clip(t, 1.0e-6, 1.0 - 1.0e-6)

  # ------------------------------------------------------------------ rollout

  @doc """
  Roll `n` envs for `frames` frames: `actor` on port 1 (the policy being
  trained), `opponent` on port 2 (frozen). Returns stacked
  `%{features: [n, T, d], action: %{k => [n, T(, 8)]}, rewards: [n, T],
  dones: [n, T]}`. Reward is `ExPhil.Sim.Critic.reward/2` (stock diff +
  0.01 × damage diff) from port 1's side.
  """
  def collect(sim, actor, opponent, n, frames, opts \\ []) do
    on_frame = Keyword.get(opts, :on_frame, fn _ -> :ok end)
    {:ok, _, _} = Env.observe(sim)
    for a <- [actor, opponent], do: ensure_batch(a, n)
    {:ok, gs0s} = Env.frames(sim)

    {acc, _} =
      Enum.reduce(1..frames, {[], gs0s}, fn t, {acc, states} ->
        {:ok, c1s, sample} = Agent.batch_get_controllers(actor, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1, return_sample: true)
        {:ok, c2s} = Agent.batch_get_controllers(opponent, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)

        case Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a || @neutral, b || @neutral] end)) do
          {:ok, nexts, terms} ->
            rewards = Enum.zip_with(states, nexts, &ExPhil.Sim.Critic.reward/2)
            dones = Enum.map(terms, fn term -> if((term["done"] || 0) == 1, do: 1, else: 0) end)
            on_frame.(t)
            {[{sample, rewards, dones} | acc], nexts}

          {:error, reason} ->
            raise "sim step failed: #{inspect(reason)}"
        end
      end)

    steps = Enum.reverse(acc)

    features = steps |> Enum.map(fn {s, _, _} -> s.features end) |> Nx.stack() |> Nx.transpose(axes: [1, 0, 2])

    action =
      Map.new(@components, fn k ->
        stacked = steps |> Enum.map(fn {s, _, _} -> squeeze_row(s.action[k], k) end) |> Nx.stack()
        {k, if(k == :buttons, do: Nx.transpose(stacked, axes: [1, 0, 2]), else: Nx.transpose(stacked))}
      end)

    %{
      features: features,
      action: action,
      rewards: steps |> Enum.map(fn {_, r, _} -> r end) |> Nx.tensor(type: :f32) |> Nx.transpose(),
      dones: steps |> Enum.map(fn {_, _, d} -> d end) |> Nx.tensor(type: :u8) |> Nx.transpose(),
      n: n,
      frames: frames,
      d: Nx.axis_size(features, 2)
    }
  end

  # the agent hands back [n, 8] for buttons and [n, 1] (or [n]) for the axes
  defp squeeze_row(t, :buttons), do: Nx.reshape(t, {Nx.axis_size(t, 0), 8})
  defp squeeze_row(t, _), do: Nx.reshape(t, {:auto})

  @doc "GAE(λ) advantages and returns from `[n, T]` rewards/values/dones."
  def gae(rewards, values, dones, gamma, lambda) do
    t = Nx.axis_size(rewards, 1)

    {adv_rev, _} =
      Enum.reduce((t - 1)..0//-1, {[], Nx.broadcast(0.0, {Nx.axis_size(rewards, 0)})}, fn i, {acc, carry} ->
        r = Nx.slice_along_axis(rewards, i, 1, axis: 1) |> Nx.squeeze(axes: [1])
        v = Nx.slice_along_axis(values, i, 1, axis: 1) |> Nx.squeeze(axes: [1])
        done = Nx.slice_along_axis(dones, i, 1, axis: 1) |> Nx.squeeze(axes: [1]) |> Nx.as_type(:f32)
        not_done = Nx.subtract(1.0, done)

        v_next =
          if i == t - 1,
            do: Nx.broadcast(0.0, Nx.shape(v)),
            else: Nx.slice_along_axis(values, i + 1, 1, axis: 1) |> Nx.squeeze(axes: [1])

        delta = Nx.add(r, Nx.subtract(Nx.multiply(Nx.multiply(gamma, v_next), not_done), v))
        a = Nx.add(delta, Nx.multiply(Nx.multiply(gamma * lambda, not_done), carry))
        {[a | acc], a}
      end)

    advantages = Nx.stack(adv_rev, axis: 1)
    {advantages, Nx.add(advantages, values)}
  end

  defp ensure_batch(agent, n) do
    case Agent.batch_reset_rows(agent, Enum.to_list(0..(n - 1))) do
      :ok -> :ok
      {:error, :batch_not_initialized} -> :ok = Agent.batch_init(agent, n)
      {:error, other} -> raise "batch reset failed: #{inspect(other)}"
    end
  end
end
