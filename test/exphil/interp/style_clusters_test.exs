defmodule ExPhil.Interp.StyleClustersTest do
  use ExUnit.Case, async: true
  alias ExPhil.Interp.StyleClusters

  defp blobs(k, n, seed) do
    :rand.seed(:exsss, {seed, seed, seed})
    for c <- 1..k, _ <- 1..n do
      base = for i <- 1..5, do: c * 3.0 * rem(i, 2) + rem(c * i, 3)
      {"T#{c}", Enum.map(base, &(&1 + (:rand.uniform() - 0.5) * 0.4))}
    end
  end

  test "recovers well-separated blobs with full purity" do
    pts = blobs(6, 20, 1)
    m = StyleClusters.fit(Enum.map(pts, &elem(&1, 1)), 6, iters: 30)
    assert m.k == 6 and length(m.assignments) == 120
    pairs = Enum.zip(Enum.map(pts, &elem(&1, 0)), m.assignments)
    assert StyleClusters.purity(pairs).purity == 1.0
    assert StyleClusters.purity(pairs).clusters_used == 6
  end

  test "predict agrees with fit assignments and reports distances" do
    pts = blobs(3, 10, 2)
    vs = Enum.map(pts, &elem(&1, 1))
    m = StyleClusters.fit(vs, 3)
    pred = StyleClusters.predict(m, vs)
    assert Enum.map(pred, &elem(&1, 0)) == m.assignments
    assert Enum.all?(pred, fn {_, d} -> d >= 0 and d < 1.0 end)
  end

  test "purity of a shuffled labelling is low" do
    pairs = for i <- 1..300, do: {"T#{rem(i, 10)}", rem(div(i, 3), 10)}
    assert StyleClusters.purity(pairs).purity < 0.5
  end
end
