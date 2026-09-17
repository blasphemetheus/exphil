defmodule ExPhil.Interp.StyleMatcher do
  @moduledoc """
  STYLE_IDENTITY.md S3a: closed-set Bayesian identity matching.

  For a game (query vector + costume), the posterior over known entities is

      P(e | game) ∝ P(e) · P(costume | e) · LR(d(q, c_e))

  * `P(e)`         — share of tagged games (a frequent player is the likelier
                      guess; `:flat_prior` disables this),
  * `P(costume|e)` — add-one-smoothed histogram of the costumes that entity's
                      tagged games used (YETI_SCENE_PRIORS.md: costume is a
                      data point, never a filter — everyone occasionally
                      picks another colour),
  * `LR(d)`        — likelihood ratio p_same(d) / p_diff(d) of the query's
                      distance to the entity centroid, with both densities
                      fitted (1-D Gaussians) on the gallery's own
                      leave-one-out game-to-centroid distances: same-player
                      vs other-player. This is S2's calibration used
                      directly, in the embedding space the rows carry
                      (`:vec` from `StyleCalibration.embed/2`).

  An explicit UNKNOWN hypothesis (tag `"?"`, prior `:unknown_prior` as a
  multiple of the mean entity prior, LR = 1) keeps the closed set honest: a
  game whose distance to every entity looks like an other-player distance
  is assigned to `"?"` rather than to its least-far player. A game is *assigned* when the top
  posterior is a real entity clearing `threshold`; otherwise it stays
  anonymous. `evaluate/2` runs leave-one-out over the tagged games
  and reports accuracy AND coverage at the threshold — the two numbers the
  threshold trades between.
  """

  @type entity :: %{tag: String.t(), n: pos_integer(), centroid: [float()], costumes: %{integer() => float()}}

  @doc """
  Build the entity gallery from tagged rows (grouped by tag, rows carrying
  `:vec` and `:costume`). `:costume_slots` is the number of costume slots
  for the character (Fox 4), used for add-one smoothing.
  """
  @spec gallery(%{String.t() => [map()]}, keyword()) :: %{entities: [entity()], lr: map(), slots: pos_integer()}
  def gallery(by_tag, opts \\ []) do
    slots = Keyword.get(opts, :costume_slots, 4)
    {same, diff} = loo_distances(by_tag)
    lr = %{same: gauss_fit(same), diff: gauss_fit(diff)}

    entities =
      Enum.map(by_tag, fn {tag, rows} ->
        %{
          tag: tag,
          n: length(rows),
          centroid: mean_vec(Enum.map(rows, & &1.vec)),
          costumes: costume_dist(rows, slots)
        }
      end)

    %{entities: entities, lr: lr, slots: slots}
  end

  @doc "Posterior over entities for one query row; sorted desc, as `[{tag, p}]`."
  @spec posterior(map(), map(), keyword()) :: [{String.t(), float()}]
  def posterior(row, %{entities: entities, lr: lr, slots: slots}, opts \\ []) do
    flat = Keyword.get(opts, :flat_prior, false)
    unknown_prior = Keyword.get(opts, :unknown_prior, 1.0)
    use_costume = Keyword.get(opts, :costume, true)
    exclude = Keyword.get(opts, :exclude_self, false)
    total = entities |> Enum.map(& &1.n) |> Enum.sum()

    scores =
      Enum.flat_map(entities, fn e ->
        # leave-one-out: drop the query from its own entity's centroid
        {centroid, n} =
          if exclude and e.tag == row.tag and e.n > 1 do
            {Enum.zip(e.centroid, row.vec) |> Enum.map(fn {c, x} -> (c * e.n - x) / (e.n - 1) end), e.n - 1}
          else
            {e.centroid, e.n}
          end

        if n == 0 do
          []
        else
          d2 = Enum.zip(row.vec, centroid) |> Enum.reduce(0.0, fn {a, b}, s -> s + (a - b) * (a - b) end)
          log_prior = if flat, do: 0.0, else: :math.log(n / total)
          log_costume = if use_costume and is_integer(row[:costume]), do: :math.log(Map.get(e.costumes, row.costume, 1.0 / (e.n + slots))), else: 0.0
          [{e.tag, log_prior + log_costume + log_lr(:math.sqrt(d2), lr)}]
        end
      end)

    # unknown: prior = unknown_prior × the mean entity prior, LR = 1.
    scores =
      if unknown_prior > 0 and entities != [],
        do: [{"?", :math.log(unknown_prior / length(entities))} | scores],
        else: scores

    m = scores |> Enum.map(&elem(&1, 1)) |> Enum.max(fn -> 0.0 end)
    z = scores |> Enum.map(fn {_, s} -> :math.exp(s - m) end) |> Enum.sum()
    scores |> Enum.map(fn {t, s} -> {t, :math.exp(s - m) / z} end) |> Enum.sort_by(&(-elem(&1, 1)))
  end

  @doc """
  Leave-one-out evaluation over the gallery's own tagged rows: for each
  threshold, accuracy among assigned games and coverage (share assigned).
  """
  @spec evaluate(%{String.t() => [map()]}, map(), keyword()) :: map()
  def evaluate(by_tag, gallery, opts \\ []) do
    thresholds = Keyword.get(opts, :thresholds, [0.5, 0.7, 0.8, 0.9, 0.95])

    tops =
      by_tag
      |> Enum.flat_map(fn {tag, rows} ->
        Enum.map(rows, fn r ->
          [{best, p} | _] = posterior(r, gallery, Keyword.put(opts, :exclude_self, true))
          {best == tag, p}
        end)
      end)

    n = length(tops)

    %{
      queries: n,
      top1: Enum.count(tops, &elem(&1, 0)) / n,
      at_threshold:
        Map.new(thresholds, fn t ->
          assigned = Enum.filter(tops, fn {_, p} -> p >= t end)
          {t, %{coverage: length(assigned) / n, accuracy: if(assigned == [], do: nil, else: Enum.count(assigned, &elem(&1, 0)) / length(assigned))}}
        end)
    }
  end

  @doc "Assign pseudo-tags to untagged rows: `[{path, tag, p}]` for those clearing `threshold`."
  @spec assign([map()], map(), float(), keyword()) :: [%{path: String.t(), port: integer(), tag: String.t(), p: float()}]
  def assign(rows, gallery, threshold, opts \\ []) do
    Enum.flat_map(rows, fn r ->
      case posterior(r, gallery, opts) do
        [{tag, p} | _] when p >= threshold and tag != "?" -> [%{path: r.path, port: r.port, tag: tag, p: p}]
        _ -> []
      end
    end)
  end

  # Leave-one-out game-to-centroid distances: to the game's own entity
  # (same) and to every other entity (different).
  defp loo_distances(by_tag) do
    cents = Map.new(by_tag, fn {t, rows} -> {t, {vsum(Enum.map(rows, & &1.vec)), length(rows)}} end)

    Enum.reduce(by_tag, {[], []}, fn {tag, rows}, {same, diff} ->
      Enum.reduce(rows, {same, diff}, fn r, {same, diff} ->
        same =
          case cents[tag] do
            {sum, n} when n > 1 -> [dist(r.vec, Enum.zip(sum, r.vec) |> Enum.map(fn {s, x} -> (s - x) / (n - 1) end)) | same]
            _ -> same
          end

        diff =
          Enum.reduce(cents, diff, fn
            {^tag, _}, acc -> acc
            {_, {sum, n}}, acc -> [dist(r.vec, Enum.map(sum, &(&1 / n))) | acc]
          end)

        {same, diff}
      end)
    end)
  end

  defp gauss_fit([]), do: %{mean: 0.0, std: 1.0}

  defp gauss_fit(xs) do
    m = Enum.sum(xs) / length(xs)
    v = Enum.sum(Enum.map(xs, &((&1 - m) * (&1 - m)))) / length(xs)
    %{mean: m, std: max(:math.sqrt(v), 1.0e-6)}
  end

  defp log_gauss(d, %{mean: m, std: s}), do: -:math.log(s) - (d - m) * (d - m) / (2 * s * s)

  defp log_lr(d, %{same: same, diff: diff}), do: log_gauss(d, same) - log_gauss(d, diff)

  defp dist(a, b), do: :math.sqrt(Enum.zip(a, b) |> Enum.reduce(0.0, fn {x, y}, s -> s + (x - y) * (x - y) end))

  defp vsum([first | rest]), do: Enum.reduce(rest, first, fn v, acc -> Enum.zip(acc, v) |> Enum.map(fn {a, b} -> a + b end) end)

  defp costume_dist(rows, slots) do
    counts = rows |> Enum.map(& &1[:costume]) |> Enum.filter(&is_integer/1) |> Enum.frequencies()
    n = rows |> Enum.count(&is_integer(&1[:costume]))
    Map.new(0..(slots - 1), fn s -> {s, (Map.get(counts, s, 0) + 1) / (n + slots)} end)
  end

  defp mean_vec(vs) do
    n = length(vs)
    vs |> Enum.reduce(fn v, acc -> Enum.zip(acc, v) |> Enum.map(fn {a, b} -> a + b end) end) |> Enum.map(&(&1 / n))
  end
end
