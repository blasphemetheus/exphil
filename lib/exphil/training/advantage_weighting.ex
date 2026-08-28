defmodule ExPhil.Training.AdvantageWeighting do
  @moduledoc """
  Per-frame LOSS weights from observed outcomes — the AWBC (advantage-
  weighted behavioral cloning) entry rung of offline RL (OFFLINE_RL_SPEC
  F5, adapted to the multishine specialist first per HANDOFF_2026-08-12).

  Plain BC imitates every demonstrated frame equally, mistakes included.
  AWBC reweights the SAME imitation loss by how well things went next:

      r_t   = shine-chain reward (below)
      R_t   = sum_{k=0..H} gamma^k * r_{t+k}      (return-to-go, within-list)
      A_t   = R_t - mean(R)                        (per-list mean baseline, v1)
      w_t   = clip(exp(A_t / beta), w_lo, w_hi)    then mean-normalized to 1.0

  ## Reward (multishine)

  `+1.0` on each frame that ENTERS the grounded-reflector family
  (`ShineChain.family/1`) — exactly one per completed multishine cycle.
  The return-to-go is then a discounted count of upcoming shines: frames
  inside (and leading into) long unbroken chains carry high advantage,
  frames before a break carry low advantage, with no separate break
  penalty needed. `:air_shine_reward` optionally credits aerial-shine
  entries too (default 0.0).

  ## Beta (percentile rule)

  When `:beta` is not given it is set from the advantage spread so that
  the p90/p10 WEIGHT ratio lands at `:weight_ratio` (default 7, the
  spec's 5-10 band): `beta = (A_p90 - A_p10) / ln(ratio)`. This keeps
  the sharpness data-scaled instead of hand-tuned.

  ## Distinct from the sampling-weight family

  ConversionSampling / OpenerSampling / MarginSampling produce POOL
  SAMPLING weights (how often a frame is drawn per epoch). These are
  LOSS weights: they multiply the per-frame imitation loss via the
  batch `:frame_weights` channel (`Data` composes them with the
  neutral/transition weights at batch assembly). Mean-normalization to
  1.0 keeps the overall loss scale comparable to plain BC (B1 vs B2).

  ## The B3 control

  `shuffle: true` permutes the computed weights WITHIN each frame list
  (deterministically, from `:seed`): identical weight distribution, zero
  outcome information. Any B2 gain that B3 reproduces is placebo — the
  pre-registered control from the spec.

  Frame lists are the SHIFTED lists (labels aligned to states), same
  contract as the sampling-weight modules. Returns-to-go never cross
  list boundaries.
  """

  alias ExPhil.Eval.ShineChain
  alias ExPhil.Rewards

  @default_gamma 0.997
  @default_horizon 300
  @default_clip {0.2, 5.0}
  @default_weight_ratio 7.0

  @doc """
  Compute per-frame AWBC loss weights over shifted frame lists.

  Returns `{weights, stats}` where `weights` is a flat list aligned with
  `List.flatten(frame_lists)` and `stats` is a map with `:frames`,
  `:shine_entries`, `:beta`, `:p10_w`, `:p90_w`, `:weight_ratio`,
  `:flat_lists` (lists with zero advantage spread — no signal there).

  Options:
    * `:beta` — advantage temperature; default: percentile rule
    * `:weight_ratio` — target p90/p10 weight ratio for the rule (#{@default_weight_ratio})
    * `:gamma` — discount (#{@default_gamma})
    * `:horizon` — return-to-go horizon in frames (#{@default_horizon})
    * `:clip` — `{lo, hi}` weight clip (#{inspect(@default_clip)})
    * `:air_shine_reward` — reward for aerial-shine entry (0.0)
    * `:port` — bot port in the frame lists (1)
    * `:shuffle` — B3 control: permute weights within each list (false)
    * `:seed` — determinism for `:shuffle` (0)
  """
  def frame_weights(frame_lists, opts \\ []) do
    gamma = Keyword.get(opts, :gamma, @default_gamma)
    horizon = Keyword.get(opts, :horizon, @default_horizon)
    {clip_lo, clip_hi} = Keyword.get(opts, :clip, @default_clip)
    air_reward = Keyword.get(opts, :air_shine_reward, 0.0)
    port = Keyword.get(opts, :port, 1)
    reward = Keyword.get(opts, :reward, :shine)

    rewards_per_list =
      case reward do
        :standard -> Enum.map(frame_lists, &standard_rewards(&1, port))
        _ -> Enum.map(frame_lists, &rewards(&1, port, air_reward))
      end

    advantages_per_list =
      Enum.map(rewards_per_list, fn rs ->
        returns = return_to_go(rs, gamma, horizon)
        n = length(returns)
        mean = if n > 0, do: Enum.sum(returns) / n, else: 0.0
        Enum.map(returns, &(&1 - mean))
      end)

    flat_adv = List.flatten(advantages_per_list)

    beta =
      Keyword.get(opts, :beta) ||
        percentile_beta(flat_adv, Keyword.get(opts, :weight_ratio, @default_weight_ratio))

    raw_per_list =
      Enum.map(advantages_per_list, fn advs ->
        Enum.map(advs, fn a ->
          :math.exp(a / beta) |> min(clip_hi) |> max(clip_lo)
        end)
      end)

    # Mean-normalize to 1.0 so the overall loss scale matches plain BC
    flat_raw = List.flatten(raw_per_list)
    mean_w = if flat_raw == [], do: 1.0, else: Enum.sum(flat_raw) / length(flat_raw)

    normed_per_list =
      Enum.map(raw_per_list, fn ws -> Enum.map(ws, &(&1 / mean_w)) end)

    normed_per_list =
      if Keyword.get(opts, :shuffle, false) do
        seed = Keyword.get(opts, :seed, 0)

        Enum.with_index(normed_per_list, fn ws, i ->
          :rand.seed(:exsss, {seed, i, 424_242})
          Enum.shuffle(ws)
        end)
      else
        normed_per_list
      end

    weights = List.flatten(normed_per_list)
    sorted = Enum.sort(weights)
    n = length(sorted)
    q = fn p -> if n > 0, do: Enum.at(sorted, min(trunc(p * n), n - 1)), else: 1.0 end
    p10 = q.(0.10)
    p90 = q.(0.90)

    stats = %{
      frames: n,
      reward: reward,
      shine_entries: rewards_per_list |> List.flatten() |> Enum.count(&(&1 > 0.0)),
      beta: beta,
      p10_w: p10,
      p90_w: p90,
      weight_ratio: if(p10 > 0, do: p90 / p10, else: nil),
      flat_lists:
        Enum.count(advantages_per_list, fn advs ->
          advs == [] or Enum.all?(advs, &(abs(&1) < 1.0e-9))
        end)
    }

    {weights, stats}
  end

  @doc """
  Per-frame shine rewards for one frame list: `+1.0` on entry into the
  grounded-reflector family, `air_shine_reward` on entry into the aerial
  one.
  """
  def rewards(frames, port \\ 1, air_reward \\ 0.0) do
    frames
    |> Enum.map(fn f ->
      p = f.game_state.players[port]
      ShineChain.family(trunc((p && p.action) || 0))
    end)
    |> Enum.map_reduce(:other, fn fam, prev ->
      r =
        cond do
          fam == :ground_reflect and prev != :ground_reflect -> 1.0
          fam == :air_reflect and prev != :air_reflect -> air_reward
          true -> 0.0
        end

      {r, fam}
    end)
    |> elem(0)
  end

  @doc """
  Per-frame STANDARD rewards (OFFLINE_RL_SPEC generalist arm): the stock +
  damage + win transition reward `r_t = reward(state_t, state_{t+1})` from
  `Rewards.Standard`, combined with `Rewards.default_config/0`'s standard
  weights (stock 1.0, damage 0.01, win 5.0). The last frame has no transition
  and gets 0.0. Same length as `frames`, so return-to-go stays within-list.
  """
  def standard_rewards(frames, port \\ 1) do
    transitions =
      frames
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [prev, curr] ->
        s = Rewards.Standard.compute(prev.game_state, curr.game_state, player_port: port)
        s.stock * 1.0 + s.damage * 0.01 + s.win * 5.0
      end)

    transitions ++ [0.0]
  end

  @doc """
  Bounded-horizon discounted return-to-go:
  `R_t = sum_{k=0..H} gamma^k * r_{t+k}`, computed with the reverse
  recurrence `R_t = r_t + gamma * R_{t+1} - gamma^(H+1) * r_{t+H+1}`.
  """
  def return_to_go(rewards, gamma, horizon) when is_list(rewards) do
    arr = List.to_tuple(rewards)
    n = tuple_size(arr)
    tail_discount = :math.pow(gamma, horizon + 1)

    Enum.reduce((n - 1)..0//-1, [], fn t, acc ->
      next = if acc == [], do: 0.0, else: hd(acc)
      falls_off = if t + horizon + 1 < n, do: elem(arr, t + horizon + 1), else: 0.0
      [elem(arr, t) + gamma * next - tail_discount * falls_off | acc]
    end)
  end

  # beta such that exp((p90A - p10A)/beta) = ratio. Degenerate spread
  # (all-equal advantages) falls back to 1.0 — weights come out uniform,
  # which is the honest answer for a signal-free pool.
  defp percentile_beta(advantages, ratio) do
    sorted = Enum.sort(advantages)
    n = length(sorted)

    if n == 0 do
      1.0
    else
      p10 = Enum.at(sorted, min(trunc(0.10 * n), n - 1))
      p90 = Enum.at(sorted, min(trunc(0.90 * n), n - 1))
      spread = p90 - p10
      if spread < 1.0e-9, do: 1.0, else: spread / :math.log(ratio)
    end
  end
end
