defmodule ExPhil.Training.ConfigPassthroughTest do
  use ExUnit.Case, async: true
  @moduletag :training

  alias ExPhil.Training.Imitation

  @doc """
  Verify that ALL training config options actually reach the Imitation trainer config.

  This test exists because @default_config in imitation.ex acts as a whitelist —
  any option NOT in the default map is silently dropped by Keyword.take().
  We wasted hours debugging entropy_weight not reaching the loss function because
  of this exact issue.
  """

  @critical_loss_options [
    :focal_loss, :focal_gamma, :button_weight, :button_pos_weight,
    :stick_edge_weight, :label_smoothing, :entropy_weight
  ]

  @critical_training_options [
    :learning_rate, :batch_size, :max_grad_norm, :dropout, :precision,
    :temporal, :backbone, :window_size, :num_layers, :accumulation_steps
  ]

  describe "config passthrough" do
    test "all critical loss options are in @default_config" do
      # Build a trainer with all options set to non-default values
      opts = [
        embed_size: 32, hidden_sizes: [32], hidden_size: 32,
        focal_loss: true, focal_gamma: 1.5, button_weight: 3.0,
        button_pos_weight: nil, stick_edge_weight: 2.0,
        label_smoothing: 0.05, entropy_weight: 0.02
      ]

      trainer = Imitation.new(opts)

      for key <- @critical_loss_options do
        val = trainer.config[key]
        assert val != nil or key == :button_pos_weight,
          "Config option :#{key} was dropped — add it to @default_config in imitation.ex"
      end
    end

    test "entropy_weight reaches the config" do
      trainer = Imitation.new(
        embed_size: 32, hidden_sizes: [32], hidden_size: 32,
        entropy_weight: 0.01
      )

      assert trainer.config[:entropy_weight] == 0.01,
        "entropy_weight not in trainer config — was it added to @default_config?"
    end

    test "all critical training options are in @default_config" do
      for key <- @critical_training_options do
        trainer = Imitation.new(
          embed_size: 32, hidden_sizes: [32], hidden_size: 32
        )

        assert Map.has_key?(trainer.config, key),
          "Training option :#{key} not in @default_config — it will be silently dropped"
      end
    end
  end
end
