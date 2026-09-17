defmodule ExPhil.Interp.StyleCalibration do
  @moduledoc """
  STYLE_IDENTITY.md step 2: calibrate the fingerprint matcher on tagged
  games, which are free ground truth.

  Inputs are fingerprint rows (`scripts/style_fingerprint.exs` JSONL:
  `%{tag, features, corpus?}`). Outputs answer three questions:

    * how far apart are same-player games vs different-player games
      (`pair_distributions/2`);
    * given a game, does its nearest tag centroid name the right player
      (`retrieval/3`, leave-one-out so a game never votes for itself);
    * what distance threshold buys a chosen false-match rate
      (`threshold_at_fmr/2`).

  Pure functions over lists; no Nx, so the script runs on any box and the
  tests need no GPU. Distances come from `StyleFingerprint.distance/2`
  (full feature set) or `invariant_distance/2` (controller micro only —
  the cross-character metric).
  """

  alias ExPhil.Interp.StyleFingerprint

  @type row :: %{tag: String.t() | nil, features: %{atom() => float()}}

  @doc "Rows whose tag is non-nil and whose tag has at least `min_games` rows."
  @spec tagged(list(row()), pos_integer()) :: %{String.t() => [row()]}
  def tagged(rows, min_games \\ 3) do
    rows
    |> Enum.reject(&is_nil(&1.tag))
    |> Enum.group_by(& &1.tag)
    |> Enum.filter(fn {_, rs} -> length(rs) >= min_games end)
    |> Map.new()
  end

  @doc """
  Per-tag centroid = per-feature mean, in the same space `distance/2`
  reads (the fingerprint maps themselves; compression happens inside
  the distance).
  """
  @spec centroid([row()]) :: %{atom() => float()}
  def centroid(rows) do
    n = length(rows)

    feats =
      StyleFingerprint.keys()
      |> Map.new(fn k -> {k, Enum.sum(Enum.map(rows, &Map.get(&1.features, k, 0.0))) / n} end)

    case rows do
      [%{vec: v} | _] when is_list(v) ->
        d = length(v)
        vec = for i <- 0..(d - 1), do: Enum.sum(Enum.map(rows, &Enum.at(&1.vec, i))) / n
        %{features: feats, vec: vec}

      _ ->
        %{features: feats}
    end
  end

  @doc """
  Same-tag and different-tag pairwise distances. `metric` is `:full` or
  `:invariant`. Different-tag pairs are sampled (`max_pairs`) so a corpus
  of thousands stays tractable; same-tag pairs are exhaustive.
  """
  @spec pair_distributions(%{String.t() => [row()]}, keyword()) :: %{same: [float()], different: [float()]}
  def pair_distributions(by_tag, opts \\ []) do
    metric = Keyword.get(opts, :metric, :full)
    max_pairs = Keyword.get(opts, :max_pairs, 20_000)
    seed = Keyword.get(opts, :seed, 17)

    same =
      Enum.flat_map(by_tag, fn {_, rs} ->
        for {a, i} <- Enum.with_index(rs), {b, j} <- Enum.with_index(rs), i < j, do: dist(a, b, metric)
      end)

    :rand.seed(:exsss, {seed, seed, seed})
    tags = Map.keys(by_tag)

    different =
      if length(tags) < 2 do
        []
      else
        Stream.repeatedly(fn ->
          [ta, tb] = Enum.take_random(tags, 2)
          dist(Enum.random(by_tag[ta]), Enum.random(by_tag[tb]), metric)
        end)
        |> Enum.take(min(max_pairs, length(same) * 4 + 1000))
      end

    %{same: same, different: different}
  end

  @doc """
  Leave-one-out nearest-centroid retrieval: for every tagged game, rank
  all tag centroids (its own tag's centroid recomputed WITHOUT it) by
  distance; report top-1 / top-5 accuracy and per-tag top-1.

  With `:gallery` given (another corpus's rows grouped by tag), centroids
  come from the gallery and queries from `by_tag` — the cross-corpus
  test; only tags present in both are scored.

  Vectorized (09-17): rows are embedded once (`:raw` when they carry no
  `:vec`; the invariant metric uses the invariant subset), centroids are
  per-tag sums so leave-one-out is `(sum - q) / (n - 1)`, and all
  query-centroid distances are one Nx matmul. The naive version rebuilt
  every centroid per query — ~100 s per experiment on 3k games.
  """
  @spec retrieval(%{String.t() => [row()]}, keyword()) :: map()
  def retrieval(by_tag, opts \\ []) do
    metric = Keyword.get(opts, :metric, :full)
    gallery = Keyword.get(opts, :gallery)
    vec = fn r -> metric_vector(r, metric) end

    {queries, gallery_tags, sums, counts} =
      case gallery do
        nil ->
          tags = by_tag |> Map.keys() |> Enum.sort()
          sums = Map.new(tags, fn t -> {t, by_tag[t] |> Enum.map(vec) |> vsum()} end)
          counts = Map.new(tags, fn t -> {t, length(by_tag[t])} end)
          {Map.take(by_tag, Enum.filter(tags, &(counts[&1] >= 2))), tags, sums, counts}

        g ->
          tags = g |> Map.keys() |> Enum.sort()
          sums = Map.new(tags, fn t -> {t, g[t] |> Enum.map(vec) |> vsum()} end)
          counts = Map.new(tags, fn t -> {t, length(g[t])} end)
          {Map.take(by_tag, tags), tags, sums, counts}
      end

    if map_size(queries) == 0 do
      %{queries: 0, tags: 0, gallery_size: length(gallery_tags), top1: nil, top5: nil, per_tag: %{}}
    else
      tag_index = gallery_tags |> Enum.with_index() |> Map.new()
      d = sums |> Map.values() |> hd() |> length()
      t = length(gallery_tags)

      # Base centroid matrix {T, d}; leave-one-out adjusts one row per query.
      cent = Nx.tensor(Enum.map(gallery_tags, &Enum.map(sums[&1], fn s -> s / counts[&1] end)), type: :f32)

      {qs, own, adj} =
        queries
        |> Enum.flat_map(fn {tag, rs} -> Enum.map(rs, &{tag, vec.(&1)}) end)
        |> Enum.reduce({[], [], []}, fn {tag, v}, {qs, own, adj} ->
          i = tag_index[tag]
          # leave-one-out centroid for the query's own tag (within-corpus only)
          loo =
            if is_nil(gallery) and counts[tag] > 1,
              do: Enum.zip(sums[tag], v) |> Enum.map(fn {s, x} -> (s - x) / (counts[tag] - 1) end),
              else: Enum.map(sums[tag], &(&1 / counts[tag]))

          {[v | qs], [i | own], [loo | adj]}
        end)

      q = Nx.tensor(Enum.reverse(qs), type: :f32)
      own = Enum.reverse(own)
      adj = Nx.tensor(Enum.reverse(adj), type: :f32)
      n = Nx.axis_size(q, 0)

      # squared distances to every base centroid, then overwrite the own-tag column
      qq = Nx.sum(Nx.multiply(q, q), axes: [1]) |> Nx.new_axis(1)
      cc = Nx.sum(Nx.multiply(cent, cent), axes: [1]) |> Nx.new_axis(0)
      d2 = Nx.subtract(Nx.add(qq, cc), Nx.multiply(2, Nx.dot(q, Nx.transpose(cent))))
      diff = Nx.subtract(q, adj)
      own_d2 = Nx.sum(Nx.multiply(diff, diff), axes: [1])
      own_idx = Nx.tensor(own, type: :s64)
      onehot = Nx.equal(Nx.new_axis(own_idx, 1), Nx.iota({1, t}))
      d2 = Nx.select(onehot, Nx.new_axis(own_d2, 1), d2)

      # rank of the own tag = number of centroids strictly closer
      ranks = Nx.sum(Nx.as_type(Nx.less(d2, Nx.new_axis(own_d2, 1)), :s64), axes: [1]) |> Nx.to_flat_list()
      tags_of = queries |> Enum.flat_map(fn {tag, rs} -> List.duplicate(tag, length(rs)) end)

      results =
        Enum.zip(tags_of, ranks)
        |> Enum.map(fn {tag, rank} -> %{tag: tag, rank: rank, top1: rank == 0, top5: rank < 5} end)

      _ = d

      %{
        queries: n,
        tags: map_size(queries),
        gallery_size: t,
        top1: Enum.count(results, & &1.top1) / n,
        top5: Enum.count(results, & &1.top5) / n,
        per_tag:
          results
          |> Enum.group_by(& &1.tag)
          |> Map.new(fn {tg, rs} -> {tg, %{n: length(rs), top1: Enum.count(rs, & &1.top1) / length(rs)}} end)
      }
    end
  end

  # The vector `dist/3` would compare for this metric.
  defp metric_vector(%{vec: v}, metric) when metric != :invariant and is_list(v), do: v

  defp metric_vector(row, :invariant) do
    StyleFingerprint.invariant_keys() |> Enum.map(&compress(Map.get(row.features, &1, 0.0)))
  end

  defp metric_vector(row, _), do: raw_vector(row)

  defp vsum([first | rest]), do: Enum.reduce(rest, first, fn v, acc -> Enum.zip(acc, v) |> Enum.map(fn {a, b} -> a + b end) end)

  @doc """
  Distance threshold such that at most `fmr` of different-player pairs
  fall below it, with the fraction of same-player pairs it accepts.
  """
  @spec threshold_at_fmr(%{same: [float()], different: [float()]}, float()) :: %{threshold: float(), same_accepted: float(), fmr: float()}
  def threshold_at_fmr(%{same: same, different: different}, fmr) do
    sorted = Enum.sort(different)
    idx = max(trunc(fmr * length(sorted)) - 1, 0)
    threshold = Enum.at(sorted, idx, 0.0)

    %{
      threshold: threshold,
      fmr: fmr,
      same_accepted: if(same == [], do: nil, else: Enum.count(same, &(&1 <= threshold)) / length(same))
    }
  end

  @doc "Quantile summary of a distance list."
  @spec summary([float()]) :: map()
  def summary([]), do: %{n: 0}

  def summary(xs) do
    s = Enum.sort(xs)
    n = length(s)
    q = fn p -> Enum.at(s, min(trunc(p * n), n - 1)) end
    %{n: n, mean: Enum.sum(s) / n, p10: q.(0.1), p50: q.(0.5), p90: q.(0.9)}
  end

  @doc "Parse a fingerprint JSONL file into rows with atom feature keys."
  @spec load_jsonl(Path.t(), String.t()) :: [row()]
  def load_jsonl(path, corpus) do
    known = MapSet.new(StyleFingerprint.keys())

    path
    |> File.stream!()
    |> Stream.map(&Jason.decode!/1)
    |> Enum.map(fn r ->
      feats =
        r["features"]
        |> Enum.flat_map(fn {k, v} ->
          a = String.to_atom(k)
          if MapSet.member?(known, a) and is_number(v), do: [{a, v * 1.0}], else: []
        end)
        |> Map.new()

      %{
        tag: r["tag"],
        path: r["path"],
        port: r["port"],
        corpus: corpus,
        # identity evidence beyond the features (S3 matcher)
        costume: r["costume"],
        costume_name: r["costume_name"],
        character: r["character"],
        stage: r["stage"],
        started_at: r["started_at"],
        features: feats
      }
    end)
  end

  # Rows carrying a `:vec` (from `embed/2`) are compared in that space for
  # every metric except :invariant; otherwise fall back to the raw
  # feature-map distances.
  defp dist(%{vec: va}, %{vec: vb}, metric) when metric != :invariant and is_list(va) and is_list(vb),
    do: euclid(va, vb)

  defp dist(a, b, :full), do: StyleFingerprint.distance(a.features, b.features)
  defp dist(a, b, :invariant), do: StyleFingerprint.invariant_distance(a.features, b.features)
  defp dist(a, b, _), do: StyleFingerprint.distance(a.features, b.features)

  defp euclid(a, b) do
    Enum.zip(a, b) |> Enum.map(fn {x, y} -> (x - y) * (x - y) end) |> Enum.sum() |> :math.sqrt()
  end

  @doc """
  Compressed raw vector for a row (log1p on rates, proportions raw) — the
  space `StyleFingerprint.distance/2` works in.
  """
  @spec raw_vector(row()) :: [float()]
  def raw_vector(row) do
    row.features |> StyleFingerprint.vector() |> Enum.map(&compress/1)
  end

  defp compress(v) when v > 1.0, do: :math.log(1.0 + v)
  defp compress(v), do: v

  @doc "Per-feature mean/std of the compressed vectors (std floored at 1e-6)."
  @spec zscore_stats([row()]) :: %{mean: [float()], std: [float()]}
  def zscore_stats(rows) do
    vs = Enum.map(rows, &raw_vector/1)
    n = length(vs)
    d = length(hd(vs))
    mean = for i <- 0..(d - 1), do: Enum.sum(Enum.map(vs, &Enum.at(&1, i))) / n
    std = for {m, i} <- Enum.with_index(mean), do: max(:math.sqrt(Enum.sum(Enum.map(vs, fn v -> (Enum.at(v, i) - m) ** 2 end)) / n), 1.0e-6)
    %{mean: mean, std: std}
  end

  @doc """
  Attach `:vec` to every row: `:raw` (compressed features), `{:zscore, stats}`,
  or `{:project, stats, fun}` where `fun` maps a z-scored list to a
  projected list (the learned metric).
  """
  @spec embed([row()], :raw | {:zscore, map()} | {:project, map(), ([float()] -> [float()])}) :: [row()]
  def embed(rows, :raw), do: Enum.map(rows, &Map.put(&1, :vec, raw_vector(&1)))

  def embed(rows, {:zscore, %{mean: m, std: s}}) do
    Enum.map(rows, fn r ->
      z = Enum.zip([raw_vector(r), m, s]) |> Enum.map(fn {x, mu, sd} -> (x - mu) / sd end)
      Map.put(r, :vec, z)
    end)
  end

  def embed(rows, {:project, stats, fun}) do
    rows |> embed({:zscore, stats}) |> Enum.map(&Map.put(&1, :vec, fun.(&1.vec)))
  end
end
