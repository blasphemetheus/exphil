defmodule ExPhil.Eval.Elo do
  @moduledoc """
  Elo ratings + round-robin pairing for the checkpoint ladder (task #19).

  Pure math — the ladder scripts own game orchestration and winner
  extraction; this module owns pairings, rating folds, and standings so
  they are unit-testable without Dolphin.

  Ratings fold deterministically in match order. With the small match
  counts a ladder night produces, the K-factor dominates smoothing —
  keep `:k` fixed across a ladder so runs are comparable, and prefer
  re-folding the full match ledger (`rate_all/2`) over incremental
  updates when matches arrive out of order.
  """

  @initial 1000.0
  @default_k 32.0

  @doc "Expected score of `a` against `b` (logistic, 400-point scale)."
  def expected(ra, rb), do: 1.0 / (1.0 + :math.pow(10.0, (rb - ra) / 400.0))

  @doc """
  Fold one match into the ratings map. `score_a` is 1.0 (a wins),
  0.5 (draw), or 0.0 (b wins). Unknown players start at #{trunc(@initial)}.
  """
  def update(ratings, a, b, score_a, opts \\ []) when score_a >= 0.0 and score_a <= 1.0 do
    k = Keyword.get(opts, :k, @default_k)
    ra = Map.get(ratings, a, @initial)
    rb = Map.get(ratings, b, @initial)
    ea = expected(ra, rb)

    ratings
    |> Map.put(a, ra + k * (score_a - ea))
    |> Map.put(b, rb + k * (1.0 - score_a - (1.0 - ea)))
  end

  @doc """
  Fold a full match ledger (list of `{a, b, score_a}`) from fresh
  ratings. Deterministic in ledger order.
  """
  def rate_all(matches, opts \\ []) do
    Enum.reduce(matches, %{}, fn {a, b, score_a}, acc -> update(acc, a, b, score_a, opts) end)
  end

  @doc """
  All unordered pairs of `items`, each repeated `games_per_pair` times
  with alternating order (so neither side always gets port 1 / first
  pick — port asymmetries exist, e.g. mainline respawn nuances).
  """
  def round_robin(items, games_per_pair \\ 1) do
    pairs =
      for {a, i} <- Enum.with_index(items),
          {b, j} <- Enum.with_index(items),
          i < j,
          do: {a, b}

    Enum.flat_map(pairs, fn {a, b} ->
      Enum.map(0..(games_per_pair - 1), fn g ->
        if rem(g, 2) == 0, do: {a, b}, else: {b, a}
      end)
    end)
  end

  @doc """
  Standings from a match ledger: list of
  `%{player, rating, wins, losses, draws, games}` sorted by rating
  (descending).
  """
  def standings(matches, opts \\ []) do
    ratings = rate_all(matches, opts)

    records =
      Enum.reduce(matches, %{}, fn {a, b, score_a}, acc ->
        {aw, al, ad, bw, bl, bd} =
          cond do
            score_a == 1.0 -> {1, 0, 0, 0, 1, 0}
            score_a == 0.0 -> {0, 1, 0, 1, 0, 0}
            true -> {0, 0, 1, 0, 0, 1}
          end

        acc
        |> Map.update(a, {aw, al, ad}, fn {w, l, d} -> {w + aw, l + al, d + ad} end)
        |> Map.update(b, {bw, bl, bd}, fn {w, l, d} -> {w + bw, l + bl, d + bd} end)
      end)

    ratings
    |> Enum.map(fn {p, r} ->
      {w, l, d} = Map.get(records, p, {0, 0, 0})
      %{player: p, rating: Float.round(r, 1), wins: w, losses: l, draws: d, games: w + l + d}
    end)
    |> Enum.sort_by(&(-&1.rating))
  end
end
