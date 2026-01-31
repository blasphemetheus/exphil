defmodule ExPhil.Evaluation.MetricsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Evaluation.Metrics

  describe "accuracy/2" do
    test "computes accuracy from logits and sparse targets" do
      logits = Nx.tensor([
        [0.1, 0.9, 0.0],  # predicts class 1
        [0.8, 0.1, 0.1],  # predicts class 0
        [0.1, 0.1, 0.8]   # predicts class 2
      ])
      targets = Nx.tensor([1, 0, 1])  # correct, correct, wrong

      acc = Metrics.accuracy(logits, targets)
      assert_in_delta acc, 2/3, 0.001
    end

    test "computes accuracy from logits and one-hot targets" do
      logits = Nx.tensor([
        [0.9, 0.1],
        [0.1, 0.9]
      ])
      targets = Nx.tensor([
        [1, 0],
        [0, 1]
      ])

      acc = Metrics.accuracy(logits, targets)
      assert_in_delta acc, 1.0, 0.001
    end

    test "returns 0 when all predictions wrong" do
      logits = Nx.tensor([[0.9, 0.1], [0.9, 0.1]])
      targets = Nx.tensor([1, 1])

      acc = Metrics.accuracy(logits, targets)
      assert_in_delta acc, 0.0, 0.001
    end
  end

  describe "top_k_accuracy/3" do
    test "computes top-3 accuracy" do
      logits = Nx.tensor([
        [0.5, 0.3, 0.1, 0.1],  # top-3: [0, 1, 2] or [0, 1, 3]
        [0.1, 0.2, 0.3, 0.4]   # top-3: [3, 2, 1]
      ])
      targets = Nx.tensor([2, 3])  # first in top-3, second in top-3

      acc = Metrics.top_k_accuracy(logits, targets, 3)
      assert_in_delta acc, 1.0, 0.001
    end

    test "top-1 equals regular accuracy" do
      logits = Nx.tensor([[0.9, 0.1], [0.1, 0.9]])
      targets = Nx.tensor([0, 1])

      top1 = Metrics.top_k_accuracy(logits, targets, 1)
      acc = Metrics.accuracy(logits, targets)
      assert_in_delta top1, acc, 0.001
    end
  end

  describe "button_accuracy/2" do
    test "computes button accuracy with sigmoid threshold" do
      logits = Nx.tensor([
        [2.0, -2.0],  # sigmoid > 0.5 for first, < 0.5 for second
        [-2.0, 2.0]
      ])
      targets = Nx.tensor([
        [1, 0],
        [0, 1]
      ])

      acc = Metrics.button_accuracy(logits, targets)
      assert_in_delta acc, 1.0, 0.001
    end

    test "returns 0 when all predictions wrong" do
      # Predicts [1, 0] for both rows, but targets are [0, 1]
      logits = Nx.tensor([[2.0, -2.0], [2.0, -2.0]])
      targets = Nx.tensor([[0, 1], [0, 1]])

      acc = Metrics.button_accuracy(logits, targets)
      assert_in_delta acc, 0.0, 0.001
    end
  end

  describe "per_button_accuracy/2" do
    test "returns accuracy for each button" do
      # 8 buttons
      logits = Nx.tensor([
        [2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        [2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
      ])
      targets = Nx.tensor([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0]  # second row: A correct, B wrong
      ])

      per_button = Metrics.per_button_accuracy(logits, targets)

      assert_in_delta per_button[:a], 1.0, 0.001
      assert_in_delta per_button[:b], 0.5, 0.001
      assert_in_delta per_button[:x], 1.0, 0.001
    end

    test "returns all button labels" do
      logits = Nx.broadcast(0.0, {1, 8})
      targets = Nx.broadcast(0, {1, 8})

      per_button = Metrics.per_button_accuracy(logits, targets)

      assert Map.keys(per_button) |> Enum.sort() == Metrics.button_labels() |> Enum.sort()
    end
  end

  describe "button_rates/2" do
    test "computes predicted and actual press rates" do
      logits = Nx.tensor([
        [2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        [2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
      ])
      targets = Nx.tensor([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0]
      ])

      {predicted, actual} = Metrics.button_rates(logits, targets)

      # A pressed in both predicted samples
      assert_in_delta predicted[:a], 1.0, 0.001
      # A pressed in 1 of 2 actual samples
      assert_in_delta actual[:a], 0.5, 0.001
    end
  end

  describe "update_stick_confusion/4" do
    test "tracks prediction errors" do
      confusion = %{}
      logits = Nx.tensor([
        [0.1, 0.9, 0.0],  # predicts 1
        [0.9, 0.1, 0.0],  # predicts 0
        [0.0, 0.1, 0.9]   # predicts 2
      ])
      targets = Nx.tensor([0, 1, 2])  # all wrong except last

      updated = Metrics.update_stick_confusion(confusion, logits, targets, 2)

      # pred=1, actual=0 should have count 1
      assert Map.get(updated, {1, 0}) == 1
      # pred=0, actual=1 should have count 1
      assert Map.get(updated, {0, 1}) == 1
      # pred=2, actual=2 is correct, not tracked
      assert Map.get(updated, {2, 2}) == nil
    end

    test "accumulates counts across calls" do
      confusion = %{{1, 0} => 5}
      logits = Nx.tensor([[0.1, 0.9, 0.0]])
      targets = Nx.tensor([0])

      updated = Metrics.update_stick_confusion(confusion, logits, targets, 2)

      assert Map.get(updated, {1, 0}) == 6
    end
  end

  describe "calibration" do
    test "empty_calibration_bins creates 10 bins" do
      bins = Metrics.empty_calibration_bins()
      assert map_size(bins) == 10
      assert Map.get(bins, 0) == {0, 0}
      assert Map.get(bins, 9) == {0, 0}
    end

    test "update_calibration bins by confidence" do
      bins = Metrics.empty_calibration_bins()

      # High confidence (95%) correct prediction
      probs = Nx.tensor([[0.02, 0.03, 0.95]])
      targets = Nx.tensor([2])

      updated = Metrics.update_calibration(bins, probs, targets)

      # Should be in bin 9 (90-100%)
      {correct, total} = Map.get(updated, 9)
      assert correct == 1
      assert total == 1
    end

    test "update_calibration tracks incorrect predictions" do
      bins = Metrics.empty_calibration_bins()

      # High confidence but wrong
      probs = Nx.tensor([[0.02, 0.03, 0.95]])
      targets = Nx.tensor([0])  # actual is 0, predicted is 2

      updated = Metrics.update_calibration(bins, probs, targets)

      {correct, total} = Map.get(updated, 9)
      assert correct == 0
      assert total == 1
    end

    test "expected_calibration_error is 0 for perfect calibration" do
      # Create bins where accuracy matches confidence midpoint
      bins = %{
        5 => {55, 100}  # 55% accuracy in 50-60% bin (midpoint 55%)
      }

      ece = Metrics.expected_calibration_error(bins)
      assert_in_delta ece, 0.0, 0.001
    end

    test "expected_calibration_error penalizes miscalibration" do
      # 90% confident but only 50% accurate
      bins = %{
        9 => {50, 100}
      }

      ece = Metrics.expected_calibration_error(bins)
      # |0.5 - 0.95| = 0.45
      assert_in_delta ece, 0.45, 0.01
    end
  end

  describe "softmax/1" do
    test "produces valid probability distribution" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]])
      probs = Metrics.softmax(logits)

      # Sum should be 1
      sum = Nx.sum(probs) |> Nx.to_number()
      assert_in_delta sum, 1.0, 0.001

      # All values should be positive
      min_val = Nx.reduce_min(probs) |> Nx.to_number()
      assert min_val > 0
    end

    test "handles large logits without overflow" do
      logits = Nx.tensor([[1000.0, 1001.0, 1002.0]])
      probs = Metrics.softmax(logits)

      sum = Nx.sum(probs) |> Nx.to_number()
      assert_in_delta sum, 1.0, 0.001
    end
  end

  describe "avg_confidence/1" do
    test "returns high confidence for peaked distribution" do
      logits = Nx.tensor([[10.0, 0.0, 0.0]])
      conf = Metrics.avg_confidence(logits)
      assert conf > 0.99
    end

    test "returns low confidence for uniform distribution" do
      logits = Nx.tensor([[1.0, 1.0, 1.0]])
      conf = Metrics.avg_confidence(logits)
      assert_in_delta conf, 1/3, 0.01
    end
  end

  describe "avg_entropy/1" do
    test "returns low entropy for confident predictions" do
      logits = Nx.tensor([[10.0, 0.0, 0.0]])
      entropy = Metrics.avg_entropy(logits)
      assert entropy < 0.1
    end

    test "returns high entropy for uncertain predictions" do
      logits = Nx.tensor([[0.0, 0.0, 0.0]])
      entropy = Metrics.avg_entropy(logits)
      # Max entropy for 3 classes = log(3) ≈ 1.1
      assert entropy > 1.0
    end
  end

  describe "random_baseline/1" do
    test "returns 1/n for n buckets" do
      assert_in_delta Metrics.random_baseline(17), 1/17, 0.001
      assert_in_delta Metrics.random_baseline(21), 1/21, 0.001
    end
  end

  describe "format_bucket/2" do
    test "formats bucket directions correctly" do
      assert Metrics.format_bucket(8, 16) == "neutral"
      assert Metrics.format_bucket(0, 16) == "far_neg"
      assert Metrics.format_bucket(6, 16) == "neg"
      assert Metrics.format_bucket(10, 16) == "pos"
      assert Metrics.format_bucket(15, 16) == "far_pos"
    end
  end

  describe "merge_metrics/2" do
    test "sums metric values" do
      m1 = %{acc: 0.5, loss: 1.0}
      m2 = %{acc: 0.6, loss: 0.8}

      merged = Metrics.merge_metrics(m1, m2)

      assert_in_delta merged.acc, 1.1, 0.001
      assert_in_delta merged.loss, 1.8, 0.001
    end

    test "handles empty first map" do
      m1 = %{}
      m2 = %{acc: 0.5}

      merged = Metrics.merge_metrics(m1, m2)
      assert merged == m2
    end
  end

  describe "average_metrics/2" do
    test "divides by batch count" do
      metrics = %{acc: 2.0, loss: 4.0}

      averaged = Metrics.average_metrics(metrics, 2)

      assert_in_delta averaged.acc, 1.0, 0.001
      assert_in_delta averaged.loss, 2.0, 0.001
    end
  end
end
