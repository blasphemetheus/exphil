defmodule ExPhil.Interp.StyleMetricTest do
  use ExUnit.Case, async: false
  alias ExPhil.Interp.{StyleCalibration, StyleMetric}

  # Players differ only along 3 of 20 dims; the other 17 are pure noise
  # with 3x the variance. Raw z-scored retrieval is drowned by the noise;
  # a learned metric must recover the informative dims.
  defp synth(players, games, seed) do
    :rand.seed(:exsss, {seed, seed, seed})
    for p <- 1..players, g <- 1..games do
      signal = for i <- 1..3, do: p * 1.0 + rem(p * i, 3) * 0.5 + (:rand.uniform() - 0.5) * 0.3
      noise = for _ <- 1..17, do: (:rand.uniform() - 0.5) * 6.0
      %{tag: "P#{p}", path: "p#{p}_#{g}", port: 1, corpus: "syn", features: %{}, vec: signal ++ noise}
    end
  end

  test "NCA projection improves leave-one-out retrieval on unseen players" do
    train = synth(12, 8, 1)
    test = synth(8, 8, 99) |> Enum.map(&%{&1 | tag: "Q" <> &1.tag})

    labels = train |> Enum.map(& &1.tag) |> Enum.uniq() |> Enum.with_index() |> Map.new()
    metric = StyleMetric.fit(Enum.map(train, & &1.vec), Enum.map(train, &labels[&1.tag]), dim: 4, steps: 150, batch: 96)

    assert List.last(metric.loss) < hd(metric.loss)

    before = StyleCalibration.retrieval(StyleCalibration.tagged(test))
    projected = Enum.map(test, &%{&1 | vec: StyleMetric.project(metric, &1.vec)})
    after_ = StyleCalibration.retrieval(StyleCalibration.tagged(projected))

    assert after_.top1 > before.top1 + 0.2, "before #{before.top1} after #{after_.top1}"

    imp = StyleMetric.importance(metric)
    assert Enum.sum(Enum.take(imp, 3)) / 3 > Enum.sum(Enum.drop(imp, 3)) / 17
  end

  @tag :tmp_dir
  test "save/load round-trips the projection", %{tmp_dir: dir} do
    train = synth(4, 4, 3)
    labels = train |> Enum.map(& &1.tag) |> Enum.uniq() |> Enum.with_index() |> Map.new()
    m = StyleMetric.fit(Enum.map(train, & &1.vec), Enum.map(train, &labels[&1.tag]), dim: 3, steps: 5, batch: 16)
    path = Path.join(dir, "metric.bin")
    StyleMetric.save!(m, path)
    assert StyleMetric.project(StyleMetric.load!(path), hd(train).vec) == StyleMetric.project(m, hd(train).vec)
  end
end
