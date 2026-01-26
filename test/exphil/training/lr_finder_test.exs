defmodule ExPhil.Training.LRFinderTest do
  @moduledoc """
  Tests for the Learning Rate Finder module.
  """
  use ExUnit.Case, async: true

  alias ExPhil.Training.LRFinder

  describe "suggest_lr/2" do
    test "returns nil for empty history" do
      assert LRFinder.suggest_lr([]) == nil
    end

    test "returns nil for very short history" do
      history = [
        %{lr: 1.0e-6, loss: 1.0, smoothed_loss: 1.0, step: 0},
        %{lr: 1.0e-5, loss: 0.9, smoothed_loss: 0.9, step: 1}
      ]

      assert LRFinder.suggest_lr(history) == nil
    end

    test "suggests LR from descending loss curve" do
      # Create a typical loss curve: decreasing then increasing
      history = [
        %{lr: 1.0e-6, loss: 2.0, smoothed_loss: 2.0, step: 0},
        %{lr: 1.0e-5, loss: 1.8, smoothed_loss: 1.9, step: 1},
        %{lr: 1.0e-4, loss: 1.5, smoothed_loss: 1.6, step: 2},
        %{lr: 1.0e-3, loss: 1.0, smoothed_loss: 1.2, step: 3},
        %{lr: 1.0e-2, loss: 0.8, smoothed_loss: 0.9, step: 4},
        %{lr: 1.0e-1, loss: 2.0, smoothed_loss: 1.5, step: 5},
        %{lr: 1.0, loss: 10.0, smoothed_loss: 5.0, step: 6}
      ]

      suggested = LRFinder.suggest_lr(history)

      # Should suggest something in the descending region
      assert suggested != nil
      assert is_float(suggested)
      assert suggested > 1.0e-6
      assert suggested < 1.0
    end

    test "handles monotonically decreasing loss" do
      history = [
        %{lr: 1.0e-6, loss: 2.0, smoothed_loss: 2.0, step: 0},
        %{lr: 1.0e-5, loss: 1.8, smoothed_loss: 1.8, step: 1},
        %{lr: 1.0e-4, loss: 1.5, smoothed_loss: 1.5, step: 2},
        %{lr: 1.0e-3, loss: 1.2, smoothed_loss: 1.2, step: 3},
        %{lr: 1.0e-2, loss: 1.0, smoothed_loss: 1.0, step: 4},
        %{lr: 1.0e-1, loss: 0.9, smoothed_loss: 0.9, step: 5}
      ]

      suggested = LRFinder.suggest_lr(history)
      assert suggested != nil
    end
  end

  describe "format_results/1" do
    test "formats results as readable string" do
      results = %{
        history: [
          %{lr: 1.0e-6, loss: 2.0, smoothed_loss: 2.0, step: 0},
          %{lr: 1.0e-5, loss: 1.5, smoothed_loss: 1.5, step: 1},
          %{lr: 1.0e-4, loss: 1.0, smoothed_loss: 1.0, step: 2}
        ],
        suggested_lr: 1.0e-4,
        min_loss_lr: 1.0e-4,
        min_loss: 1.0,
        stopped_early: false
      }

      output = LRFinder.format_results(results)

      assert output =~ "Learning Rate Finder Results"
      assert output =~ "Suggested LR"
      assert output =~ "Min Loss LR"
      assert output =~ "Steps: 3"
    end

    test "handles nil suggested_lr" do
      results = %{
        history: [
          %{lr: 1.0e-6, loss: 2.0, smoothed_loss: 2.0, step: 0}
        ],
        suggested_lr: nil,
        min_loss_lr: 1.0e-6,
        min_loss: 2.0,
        stopped_early: false
      }

      output = LRFinder.format_results(results)
      assert output =~ "N/A"
    end
  end

  describe "find/3" do
    test "returns error for insufficient data" do
      model_params = %{w: Nx.tensor([1.0, 2.0])}
      dataset = []

      result = LRFinder.find(model_params, dataset, num_steps: 10)
      assert {:error, _} = result
    end

    # Note: Full integration test requires actual model and data
    # which is covered by the script test
  end
end
