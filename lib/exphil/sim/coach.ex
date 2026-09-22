defmodule ExPhil.Sim.Coach do
  @moduledoc """
  Chess-review v0 for a real replay (COACH_STYLE_PRODUCTS, the "eval bar +
  blunder marks" experience, 2026-09-21).

  Pipeline: seed the sim at every `every`-th frame of the replay
  (`ExPhil.Sim.Seed`, bit-exact on local games), then from each seed point
  roll `samples` self-play continuations of `horizon` frames with the prior
  on BOTH ports (batched agents, one restore per env), and score each from
  the subject's side:

      value = stocks taken − stocks lost + (damage dealt − damage taken) / 100

  The mean over samples is the EVAL at that moment (what the prior expects
  to happen from here); the same metric over the replay's own next
  `horizon` frames is what ACTUALLY happened. A BLUNDER window is one where
  actual − expected < −`blunder` (default 0.5 = half a stock's worth worse
  than the prior's expectation). For each point the best sample (by value)
  is kept as the "best line" and exported as a viewer trace on request.

  Caveats (v0): the prior is a Fox generalist, so the opponent port is
  played by a Fox policy driving whatever character it is; values are the
  prior's expectation, not ground truth (a weak prior under-rates strong
  play); horizon-limited (kills beyond the window are invisible).
  """

  alias ExPhil.Agents.Agent
  alias ExPhil.Sim.{Env, Seed}

  @neutral %{main_x: 0.5, main_y: 0.5, c_x: 0.5, c_y: 0.5, shoulder: 0.0, buttons: %{}}

  @doc """
  Options: `:subject` (port, default 1), `:every` (120), `:horizon` (240),
  `:samples` (16), `:blunder` (0.5), `:agents` (`{p1_agent, p2_agent}`,
  required), `:start` (first frame, default 0), `:stop` (last frame),
  `:on_point` (fn point -> any, progress hook).

  Returns `%{points: [...], blunders: [...], divergence, replay, frames}`.
  Each point: `%{frame, expected, sd, actual, delta, best, lines: %{best, typical, worst}, samples: [values], state}`
  (each line `%{value, states}`; typical = the median sample).
  """
  def review(path, opts) do
    subject = Keyword.get(opts, :subject, 1)
    every = Keyword.get(opts, :every, 120)
    horizon = Keyword.get(opts, :horizon, 240)
    n = Keyword.get(opts, :samples, 16)
    blunder = Keyword.get(opts, :blunder, 0.5)
    {a1, a2} = Keyword.fetch!(opts, :agents)
    on_point = Keyword.get(opts, :on_point, fn _ -> :ok end)

    {:ok, meta} = ExPhil.Data.Peppi.metadata(path)
    {:ok, replay} = ExPhil.Data.Peppi.parse(path)
    by_frame = Map.new(replay.frames, &{&1.frame_number, &1})
    last_frame = replay.frames |> Enum.map(& &1.frame_number) |> Enum.max()
    start = Keyword.get(opts, :start, 0)
    stop = min(Keyword.get(opts, :stop, last_frame - horizon), last_frame - 1)
    points = Enum.to_list(start..stop//every)

    # 1. one replay pass, a savestate at every point
    {:ok, seed} = Seed.from_replay(path, frame: stop, frames: points, warm: 30)
    Env.stop(seed.sim)

    # 2. a rollout sim with `samples` envs, same match config
    {:ok, sim} = Env.start(:nif, stage: meta.stage, players: seed.players, batch_size: n, seed: 7, ucf_cardinals: 1)
    ensure_batch(a1, n)
    ensure_batch(a2, n)
    opp = other(subject)

    points =
      seed.saves
      |> Enum.filter(&MapSet.member?(MapSet.new(points), &1.frame))
      |> Enum.map(fn save ->
        rollouts = rollouts(sim, save, {a1, a2}, n, horizon)
        values = Enum.map(rollouts, &value(&1, subject, opp))
        expected = mean(values)
        actual_states = for f <- save.frame..min(save.frame + horizon, last_frame), by_frame[f], do: by_frame[f]
        actual = if length(actual_states) > 1, do: value(actual_states, subject, opp), else: nil
        ranked = Enum.zip(rollouts, values) |> Enum.sort_by(&elem(&1, 1))
        {worst_states, worst_value} = hd(ranked)
        {best_states, best_value} = List.last(ranked)
        {typical_states, typical_value} = Enum.at(ranked, div(length(ranked), 2))

        point = %{
          frame: save.frame,
          expected: expected,
          sd: sd(values, expected),
          actual: actual,
          delta: actual && actual - expected,
          samples: values,
          best: %{value: best_value, states: best_states},
          lines: %{best: %{value: best_value, states: best_states}, typical: %{value: typical_value, states: typical_states}, worst: %{value: worst_value, states: worst_states}},
          state: save.state,
          diverged?: save.diverged?
        }

        on_point.(point)
        point
      end)

    Env.stop(sim)

    # points after the seed diverged carry a bogus state (dead players, zero rollouts): never blunders
    blunders = Enum.filter(points, fn p -> not p.diverged? and p.delta != nil and p.delta < -blunder end)
    %{points: points, blunders: blunders, divergence: seed.divergence, replay: path, frames: last_frame, subject: subject, horizon: horizon, every: every, samples: n}
  end

  # `n` continuations from one savestate: restore into every env, warm both agents on the history, roll.
  defp rollouts(sim, save, {a1, a2}, n, horizon) do
    {:ok, id} = Env.upload(sim, save.blob)
    for i <- 0..(n - 1), do: {:ok, _} = Env.restore(sim, i, {:id, id}, frames: false)
    {:ok, _, _} = Env.observe(sim)
    ensure_batch(a1, n)
    ensure_batch(a2, n)

    for s <- save.history do
      :ok = Agent.batch_observe(a1, List.duplicate(%{s | own_port: 1}, n), player_port: 1)
      :ok = Agent.batch_observe(a2, List.duplicate(%{s | own_port: 2}, n), player_port: 2)
    end

    {:ok, gs0s} = Env.frames(sim)

    {history, _} =
      Enum.reduce(1..horizon, {[gs0s], gs0s}, fn _, {acc, states} ->
        {:ok, c1s} = Agent.batch_get_controllers(a1, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1)
        {:ok, c2s} = Agent.batch_get_controllers(a2, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)

        case Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a || @neutral, b || @neutral] end)) do
          {:ok, nexts, _} -> {[nexts | acc], nexts}
          {:error, reason} -> raise "sim step failed: #{inspect(reason)}"
        end
      end)

    history |> Enum.reverse() |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
  end

  @doc "Subject-side value of a state sequence: stocks taken − lost + (damage dealt − taken)/100."
  def value(states, subject, opp) do
    {dealt, taken, s_lost, o_lost} =
      states
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.reduce({0.0, 0.0, 0, 0}, fn [a, b], {d, t, sl, ol} ->
        pa = a.players
        pb = b.players
        d = d + max(0.0, (pb[opp].percent - pa[opp].percent) * 1.0)
        t = t + max(0.0, (pb[subject].percent - pa[subject].percent) * 1.0)
        sl = sl + max(0, (pa[subject].stock || 0) - (pb[subject].stock || 0))
        ol = ol + max(0, (pa[opp].stock || 0) - (pb[opp].stock || 0))
        {d, t, sl, ol}
      end)

    o_lost - s_lost + (dealt - taken) / 100.0
  end

  @doc false
  def other_port(p), do: other(p)

  defp other(1), do: 2
  defp other(2), do: 1

  defp ensure_batch(agent, n) do
    case Agent.batch_reset_rows(agent, Enum.to_list(0..(n - 1))) do
      :ok -> :ok
      {:error, :batch_not_initialized} -> :ok = Agent.batch_init(agent, n)
      {:error, other} -> raise "batch reset failed: #{inspect(other)}"
    end
  end

  defp mean([]), do: 0.0
  defp mean(xs), do: Enum.sum(xs) / length(xs)
  defp sd(xs, _m) when length(xs) < 2, do: 0.0
  defp sd(xs, m), do: :math.sqrt(Enum.sum(Enum.map(xs, &((&1 - m) * (&1 - m)))) / (length(xs) - 1))
end
