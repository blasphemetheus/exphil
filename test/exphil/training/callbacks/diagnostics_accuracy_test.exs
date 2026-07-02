defmodule ExPhil.Training.Callbacks.DiagnosticsAccuracyTest do
  use ExUnit.Case, async: true
  @moduletag :training

  # Test the accuracy computation logic extracted from Diagnostics callback
  # We test the Nx operations directly since the callback runs them inline

  describe "button accuracy computation" do
    test "perfect predictions give 100% accuracy" do
      # Logits > 0 means predicted pressed, targets > 0.5 means actually pressed
      logits = Nx.tensor([[1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
      targets = Nx.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      pred = Nx.greater(Nx.sigmoid(logits), 0.5)
      actual = Nx.greater(targets, 0.5)
      correct = Nx.equal(pred, actual) |> Nx.as_type(:f32) |> Nx.mean() |> Nx.to_number()

      assert_in_delta correct, 1.0, 0.01
    end

    test "all wrong predictions give 0% accuracy" do
      logits = Nx.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
      targets = Nx.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

      pred = Nx.greater(Nx.sigmoid(logits), 0.5)
      actual = Nx.greater(targets, 0.5)
      correct = Nx.equal(pred, actual) |> Nx.as_type(:f32) |> Nx.mean() |> Nx.to_number()

      assert_in_delta correct, 0.0, 0.01
    end

    test "50% accuracy with half correct" do
      logits = Nx.tensor([[1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]])
      targets = Nx.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]])

      pred = Nx.greater(Nx.sigmoid(logits), 0.5)
      actual = Nx.greater(targets, 0.5)
      correct = Nx.equal(pred, actual) |> Nx.as_type(:f32) |> Nx.mean() |> Nx.to_number()

      assert_in_delta correct, 0.5, 0.01
    end
  end

  describe "stick top-1 accuracy" do
    test "correct argmax gives 100% accuracy" do
      # 17 buckets, logits peaked at bucket 8 (center)
      logits = Nx.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0]])
      target = Nx.tensor([8])

      pred = Nx.argmax(logits, axis: -1)
      correct = Nx.equal(pred, target) |> Nx.sum() |> Nx.to_number()

      assert correct == 1
    end

    test "wrong argmax gives 0% accuracy" do
      logits = Nx.tensor([[10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
      target = Nx.tensor([8])

      pred = Nx.argmax(logits, axis: -1)
      correct = Nx.equal(pred, target) |> Nx.sum() |> Nx.to_number()

      assert correct == 0
    end

    test "batch accuracy averages correctly" do
      logits = Nx.tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      ])
      targets = Nx.tensor([8, 8, 10, 4])

      pred = Nx.argmax(logits, axis: -1)
      correct = Nx.equal(pred, targets) |> Nx.sum() |> Nx.to_number()

      # 3 out of 4 correct (indices 0, 2, 3)
      assert correct == 3
    end
  end

  describe "rare action recall" do
    test "perfect recall when all pressed buttons detected" do
      # Z is at index 4
      buttons = Nx.tensor([[0, 0, 0, 0, 1.0, 0, 0, 0]])
      logits = Nx.tensor([[0, 0, 0, 0, 1.0, 0, 0, 0]])  # predicts Z pressed

      target_col = Nx.slice_along_axis(buttons, 4, 1, axis: -1) |> Nx.squeeze(axes: [-1])
      pressed = Nx.greater(target_col, 0.5)
      n_pressed = Nx.sum(pressed) |> Nx.to_number() |> trunc()

      pred_col = Nx.slice_along_axis(logits, 4, 1, axis: -1) |> Nx.squeeze(axes: [-1])
      pred_pressed = Nx.greater(pred_col, 0)
      hits = Nx.logical_and(pred_pressed, pressed) |> Nx.sum() |> Nx.to_number() |> trunc()

      assert n_pressed == 1
      assert hits == 1
    end

    test "zero recall when pressed buttons not detected" do
      buttons = Nx.tensor([[0, 0, 0, 0, 1.0, 0, 0, 0]])
      logits = Nx.tensor([[0, 0, 0, 0, -1.0, 0, 0, 0]])  # doesn't predict Z

      target_col = Nx.slice_along_axis(buttons, 4, 1, axis: -1) |> Nx.squeeze(axes: [-1])
      pressed = Nx.greater(target_col, 0.5)
      n_pressed = Nx.sum(pressed) |> Nx.to_number() |> trunc()

      pred_col = Nx.slice_along_axis(logits, 4, 1, axis: -1) |> Nx.squeeze(axes: [-1])
      pred_pressed = Nx.greater(pred_col, 0)
      hits = Nx.logical_and(pred_pressed, pressed) |> Nx.sum() |> Nx.to_number() |> trunc()

      assert n_pressed == 1
      assert hits == 0
    end
  end
end
