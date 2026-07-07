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

  describe "label smoothing × pos_weight pathology (regression)" do
    # Smoothing a pos-weighted binary target moves the BCE optimum to
    # p* = wε/(wε + 1−ε). For w=30, ε=0.1: p* = 0.77 — ABOVE the press
    # threshold, training models to hold rare buttons (live symptom:
    # constant taunts and shine-grabs from every default-config model).
    # Buttons must therefore NEVER be smoothed: for an all-negative button
    # target, confidently-off logits must always beat above-threshold ones.
    test "never-pressed high-pos-weight button: 'off' prediction beats 'pressed'" do
      batch = 4

      targets = %{
        buttons: Nx.broadcast(0.0, {batch, 8}),
        main_x: Nx.broadcast(8, {batch}),
        main_y: Nx.broadcast(8, {batch}),
        c_x: Nx.broadcast(8, {batch}),
        c_y: Nx.broadcast(8, {batch}),
        shoulder: Nx.broadcast(0, {batch})
      }

      neutral_cat = Nx.broadcast(0.0, {batch, 17})
      neutral_sh = Nx.broadcast(0.0, {batch, 5})

      base = %{
        main_x: neutral_cat,
        main_y: neutral_cat,
        c_x: neutral_cat,
        c_y: neutral_cat,
        shoulder: neutral_sh
      }

      logits_off = Map.put(base, :buttons, Nx.broadcast(-4.0, {batch, 8}))
      # sigmoid(0.5) ≈ 0.62 — the old pathological optimum region
      logits_pressed = Map.put(base, :buttons, Nx.broadcast(0.5, {batch, 8}))

      pos_weight = Nx.tensor([4.5, 4.8, 6.7, 4.0, 11.7, 3.4, 4.5, 30.0])

      for frame_weights <- [nil, Nx.broadcast(1.0, {batch})] do
        opts = [
          label_smoothing: 0.1,
          focal_loss: false,
          button_pos_weight: pos_weight,
          frame_weights: frame_weights
        ]

        loss_off = Loss.imitation_loss(logits_off, targets, opts) |> Nx.to_number()
        loss_pressed = Loss.imitation_loss(logits_pressed, targets, opts) |> Nx.to_number()

        assert loss_off < loss_pressed,
               "smoothing leaked into the pos-weighted button BCE " <>
                 "(frame_weights=#{inspect(frame_weights != nil)}): " <>
                 "'off' #{loss_off} should beat 'pressed' #{loss_pressed}"
      end
    end
  end
end
