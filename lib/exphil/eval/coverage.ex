defmodule ExPhil.Eval.Coverage do
  @moduledoc """
  Situation-occupancy buckets: where does play GO? (DATA_FLYWHEEL_DESIGN
  2026-07-23, stage A3.)

  The r15→r16 lesson — "converts fine, never initiates" — restated as
  measurable occupancy: bucket every frame of the bot's games and of the
  human corpus into a coarse situation key, then diff the distributions.
  Buckets the bot underpopulates relative to humans are exactly the
  training-data gaps (e.g. a gate-10≈0 bot never visits the close-range,
  both-actionable, facing-toward buckets humans live in).

  Deliberately cheap — hand features, no NN embedding — so it runs over the
  95k-replay corpus without waiting on the streaming-embed work (#33). The
  embedding-ANN version (B1 v2) is the richer successor.

  Consumes `ExPhil.Eval.ScenarioScan` slim frames
  (`%{frame:, p1:, p2:}`, each player a `player_summary/1`
  `%{x, y, action, facing, on_ground, stock, percent}`).

  ## Bucket key

  Five features joined by `|`, e.g. `"d15-30|zmid|aboth|p40-80|ftoward"`:
  - `d<band>` horizontal distance |p2.x - p1.x|: `0-15|15-30|30-50|50+`
  - `z<band>` p1 FD zone by |x|: `center|mid|edge|off` (0-30/30-60/60-85.5/85.5+)
  - `a<band>` who is actionable (not in hitstun/knockdown): `both|p1|p2|none`
  - `p<band>` p1 percent: `0-40|40-80|80-120|120+`
  - `f<band>` p1 facing vs p2: `toward|away`

  Port convention: p1 = the subject (the bot in bot ledgers; the player of
  interest in corpus ledgers). Action sets mirror `ExPhil.Interp.ReplayStats`.
  """

  @hitstun MapSet.new(Enum.to_list(75..91) ++ Enum.to_list(223..232))
  @lifecycle MapSet.new([183, 184, 186, 187, 188, 189, 191, 192, 194, 195, 196, 197, 199, 200, 201])
  @fd_edge 85.5

  # ==========================================================================
  # Bucketing
  # ==========================================================================

  @doc "Coarse situation key for one frame's two players (p1 = subject)."
  def bucket(p1, p2) do
    Enum.join(
      [
        "d" <> dist_band(abs(p2.x - p1.x)),
        "z" <> zone_band(abs(p1.x)),
        "a" <> actionable_band(p1, p2),
        "p" <> pct_band(p1.percent || 0.0),
        "f" <> facing_band(p1, p2)
      ],
      "|"
    )
  end

  defp dist_band(dx) when dx < 15.0, do: "0-15"
  defp dist_band(dx) when dx < 30.0, do: "15-30"
  defp dist_band(dx) when dx < 50.0, do: "30-50"
  defp dist_band(_), do: "50+"

  defp zone_band(ax) when ax < 30.0, do: "center"
  defp zone_band(ax) when ax < 60.0, do: "mid"
  defp zone_band(ax) when ax <= @fd_edge, do: "edge"
  defp zone_band(_), do: "off"

  defp pct_band(pct) when pct < 40.0, do: "0-40"
  defp pct_band(pct) when pct < 80.0, do: "40-80"
  defp pct_band(pct) when pct < 120.0, do: "80-120"
  defp pct_band(_), do: "120+"

  defp actionable_band(p1, p2) do
    case {actionable?(p1), actionable?(p2)} do
      {true, true} -> "both"
      {true, false} -> "p1"
      {false, true} -> "p2"
      {false, false} -> "none"
    end
  end

  defp actionable?(%{action: a}) do
    not (MapSet.member?(@hitstun, a) or MapSet.member?(@lifecycle, a))
  end

  # Facing toward p2 when p1's facing sign matches the direction to p2.
  # facing is +1 (right) / -1 (left); dx == 0 or unknown facing -> "toward".
  defp facing_band(%{facing: f} = p1, p2) when f in [1, -1, 1.0, -1.0] do
    dx = p2.x - p1.x

    cond do
      dx == 0 -> "toward"
      sign(dx) == sign(f) -> "toward"
      true -> "away"
    end
  end

  defp facing_band(_, _), do: "toward"

  defp sign(x) when x < 0, do: -1
  defp sign(_), do: 1

  # ==========================================================================
  # Ledgers
  # ==========================================================================

  @doc "Occupancy counts `%{bucket_key => count}` over slim frames."
  def ledger(frames) do
    Enum.reduce(frames, %{}, fn f, acc ->
      key = bucket(f.p1, f.p2)
      Map.update(acc, key, 1, &(&1 + 1))
    end)
  end

  @doc "Sum two occupancy ledgers."
  def merge(a, b) do
    Map.merge(a, b, fn _k, av, bv -> av + bv end)
  end

  @doc """
  Diff a bot ledger against a corpus ledger. Returns one row per bucket
  present in EITHER, Laplace-smoothed, sorted by under-representation
  (`ratio = bot_frac / corpus_frac` ascending — most under-visited first).

  ## Options
    - `:alpha` — Laplace pseudocount (default 1.0)
    - `:min_corpus_frac` — drop buckets the corpus barely visits (default 0.0)
  """
  def diff(bot, corpus, opts \\ []) do
    alpha = opts[:alpha] || 1.0
    min_corpus = opts[:min_corpus_frac] || 0.0

    keys =
      MapSet.union(MapSet.new(Map.keys(bot)), MapSet.new(Map.keys(corpus)))
      |> MapSet.to_list()

    k = length(keys)
    bot_total = bot |> Map.values() |> Enum.sum()
    corpus_total = corpus |> Map.values() |> Enum.sum()

    keys
    |> Enum.map(fn key ->
      bc = Map.get(bot, key, 0)
      cc = Map.get(corpus, key, 0)
      bf = (bc + alpha) / (bot_total + alpha * k)
      cf = (cc + alpha) / (corpus_total + alpha * k)

      %{
        key: key,
        bot_count: bc,
        corpus_count: cc,
        bot_frac: bf,
        corpus_frac: cf,
        ratio: bf / cf
      }
    end)
    |> Enum.filter(fn r -> r.corpus_frac >= min_corpus end)
    |> Enum.sort_by(& &1.ratio)
  end

  @doc "Write a ledger to JSON."
  def save(ledger, path) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(ledger, pretty: true))
    ledger
  end

  @doc "Load a ledger from JSON (empty map if missing/unparseable)."
  def load(path) do
    case File.read(path) do
      {:ok, bin} ->
        case Jason.decode(bin) do
          {:ok, m} when is_map(m) -> m
          _ -> %{}
        end

      _ ->
        %{}
    end
  end
end
