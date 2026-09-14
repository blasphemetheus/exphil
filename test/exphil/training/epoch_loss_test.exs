defmodule ExPhil.Training.EpochLossTest do
  use ExUnit.Case, async: true
  alias ExPhil.Training.EpochLoss

  defp batch(weights),
    do: %{states: Nx.broadcast(0.0, {length(weights), 2}), frame_weights: Nx.tensor(weights)}

  test "a tiny final batch cannot hide loss in the preceding full batch" do
    metrics =
      EpochLoss.new()
      |> EpochLoss.add(Nx.tensor(2.0), batch(List.duplicate(1.0, 64)))
      |> EpochLoss.add(Nx.tensor(0.0), batch([1.0]))

    assert_in_delta EpochLoss.mean(metrics), 128 / 65, 1.0e-10
    assert metrics.batches == 2
    assert metrics.mass == 65
  end

  test "uses the weighted loss denominator rather than batch size or batch count" do
    metrics =
      EpochLoss.new()
      |> EpochLoss.add(8.0, batch([0.25, 0.25]))
      |> EpochLoss.add(2.0, batch([2.0]))

    assert EpochLoss.mean(metrics) == 3.2
  end

  test "regrouping a fixed set of sample losses preserves the aggregate" do
    first =
      EpochLoss.new() |> EpochLoss.add(3.0, batch([1.0, 1.0])) |> EpochLoss.add(9.0, batch([1.0]))

    second =
      EpochLoss.new() |> EpochLoss.add(1.0, batch([1.0])) |> EpochLoss.add(7.0, batch([1.0, 1.0]))

    assert EpochLoss.mean(first) == EpochLoss.mean(second)
    assert EpochLoss.mean(first) == 5.0
  end

  test "empty and numerically invalid epochs cannot become healthy from a later batch" do
    assert EpochLoss.mean(EpochLoss.new()) == :empty_epoch

    for invalid <- [:nan, :infinity, :neg_infinity, Nx.tensor(:nan)] do
      metrics =
        EpochLoss.new()
        |> EpochLoss.add(invalid, batch([1.0]))
        |> EpochLoss.add(0.0, batch([1.0]))

      assert EpochLoss.mean(metrics) == :nonfinite_batch_loss
    end

    for weights <- [[0.0], [-1.0], [:nan]] do
      assert EpochLoss.mean(EpochLoss.add(EpochLoss.new(), 0.0, batch(weights))) ==
               :invalid_batch_mass
    end

    overflow = EpochLoss.add(EpochLoss.new(), 1.0e308, batch([10.0]))
    assert EpochLoss.mean(overflow) == :nonfinite_aggregate
    assert overflow.batches == 1
  end

  test "unweighted policy families use row counts even when weights are present" do
    metrics =
      EpochLoss.new()
      |> EpochLoss.add(8.0, batch([0.25, 0.25]), :act)
      |> EpochLoss.add(2.0, batch([2.0]), :act)

    assert EpochLoss.mean(metrics) == 6.0
  end

  test "checkpoint ranking and stopping no longer follow the luckiest final batch" do
    lucky_tail =
      EpochLoss.new() |> EpochLoss.add(3.0, batch([1.0])) |> EpochLoss.add(0.0, batch([1.0]))

    consistent =
      EpochLoss.new() |> EpochLoss.add(0.2, batch([1.0])) |> EpochLoss.add(0.2, batch([1.0]))

    assert EpochLoss.mean(consistent) < EpochLoss.mean(lucky_tail)
    refute EpochLoss.mean(lucky_tail) < 0.01
  end

  test "unweighted lazy batches use their row count" do
    lazy = Nx.Batch.stack([Nx.tensor([1.0]), Nx.tensor([2.0])])
    metrics = EpochLoss.add(EpochLoss.new(), 3.0, %{states: lazy})
    assert metrics.mass == 2
    assert EpochLoss.mean(metrics) == 3.0
  end
end
