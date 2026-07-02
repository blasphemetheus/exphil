defmodule ExPhil.Training.LossFunctionTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Networks.Policy.Loss

  @doc """
  Verify loss function properties:
  - Always non-negative
  - Zero when predictions perfectly match targets
  - Increases with worse predictions
  - Reduction modes work correctly
  """

  describe "binary_cross_entropy" do
    test "loss is non-negative" do
      logits = Nx.tensor([[1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0]])
      targets = Nx.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      loss = Loss.binary_cross_entropy(logits, targets, 0.0) |> Nx.to_number()
      assert loss >= 0, "BCE loss should be non-negative, got #{loss}"
    end

    test "perfect predictions give low loss" do
      # Large positive logits for target=1, large negative for target=0
      logits = Nx.tensor([[10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])
      targets = Nx.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      loss = Loss.binary_cross_entropy(logits, targets, 0.0) |> Nx.to_number()
      assert loss < 0.01, "Perfect predictions should give near-zero loss, got #{loss}"
    end

    test "wrong predictions give high loss" do
      logits = Nx.tensor([[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]])
      targets = Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      loss = Loss.binary_cross_entropy(logits, targets, 0.0) |> Nx.to_number()
      assert loss > 1.0, "Wrong predictions should give high loss, got #{loss}"
    end

    test "reduction :none returns per-element loss" do
      logits = Nx.tensor([[1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0]])
      targets = Nx.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      per_element = Loss.binary_cross_entropy(logits, targets, 0.0, reduction: :none)
      assert Nx.shape(per_element) == {1, 8}, "Should return per-element, got #{inspect(Nx.shape(per_element))}"
    end

    test "reduction :sum returns scalar sum" do
      logits = Nx.tensor([[1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0]])
      targets = Nx.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      sum_loss = Loss.binary_cross_entropy(logits, targets, 0.0, reduction: :sum) |> Nx.to_number()
      mean_loss = Loss.binary_cross_entropy(logits, targets, 0.0, reduction: :mean) |> Nx.to_number()

      assert_in_delta sum_loss, mean_loss * 8, 0.01
    end

    test "label smoothing reduces confidence" do
      logits = Nx.tensor([[10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])
      targets = Nx.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      loss_no_smooth = Loss.binary_cross_entropy(logits, targets, 0.0) |> Nx.to_number()
      loss_smooth = Loss.binary_cross_entropy(logits, targets, 0.1) |> Nx.to_number()

      assert loss_smooth > loss_no_smooth,
        "Label smoothing should increase loss for confident predictions"
    end
  end

  describe "categorical_cross_entropy" do
    test "loss is non-negative" do
      logits = Nx.tensor([[1.0, 0.5, 0.0, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
      targets = Nx.tensor([0])

      loss = Loss.categorical_cross_entropy(logits, targets, 0.0) |> Nx.to_number()
      assert loss >= 0
    end

    test "correct prediction gives low loss" do
      logits = Nx.tensor([[10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])
      targets = Nx.tensor([0])

      loss = Loss.categorical_cross_entropy(logits, targets, 0.0) |> Nx.to_number()
      assert loss < 0.01
    end

    test "reduction :none returns per-sample loss" do
      logits = Nx.tensor([
        [10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
        [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]
      ])
      targets = Nx.tensor([0, 1])

      per_sample = Loss.categorical_cross_entropy(logits, targets, 0.0, reduction: :none)
      assert Nx.shape(per_sample) == {2}
    end
  end

  describe "focal loss" do
    test "focal loss reduces weight on easy examples" do
      # Easy example: high confidence correct prediction
      easy_logits = Nx.tensor([[10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])
      # Hard example: low confidence correct prediction
      hard_logits = Nx.tensor([[0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]])
      targets = Nx.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      easy_focal = Loss.focal_binary_cross_entropy(easy_logits, targets, 0.0, 2.0) |> Nx.to_number()
      hard_focal = Loss.focal_binary_cross_entropy(hard_logits, targets, 0.0, 2.0) |> Nx.to_number()

      easy_bce = Loss.binary_cross_entropy(easy_logits, targets, 0.0) |> Nx.to_number()
      hard_bce = Loss.binary_cross_entropy(hard_logits, targets, 0.0) |> Nx.to_number()

      # Focal loss should down-weight easy examples more than hard ones
      easy_ratio = if easy_bce > 0, do: easy_focal / easy_bce, else: 0
      hard_ratio = if hard_bce > 0, do: hard_focal / hard_bce, else: 0

      assert easy_ratio < hard_ratio,
        "Focal should down-weight easy (ratio=#{Float.round(easy_ratio, 4)}) " <>
        "more than hard (ratio=#{Float.round(hard_ratio, 4)})"
    end
  end
end
