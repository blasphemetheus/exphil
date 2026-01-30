defmodule ExPhil.Reports.BenchmarkTest do
  use ExUnit.Case, async: true

  alias ExPhil.Reports.Benchmark

  describe "parse_results/1" do
    test "handles map with results key" do
      raw = %{
        results: [
          %{name: "MLP", final_val_loss: 2.5, final_train_loss: 2.0}
        ]
      }

      results = Benchmark.parse_results(raw)
      assert length(results) == 1
      assert hd(results).name == "MLP"
    end

    test "handles list format" do
      raw = [
        %{name: "MLP", final_val_loss: 2.5, final_train_loss: 2.0},
        %{name: "LSTM", final_val_loss: 3.0, final_train_loss: 2.5}
      ]

      results = Benchmark.parse_results(raw)
      assert length(results) == 2
    end

    test "filters out non-result entries" do
      raw = [
        %{name: "MLP", final_val_loss: 2.5},
        %{machine: %{gpu: "RTX 4090"}}  # No :name key
      ]

      results = Benchmark.parse_results(raw)
      assert length(results) == 1
    end
  end

  describe "normalize_results/1" do
    test "normalizes avg_batches_per_sec to batches_per_sec" do
      raw = [%{name: "MLP", avg_batches_per_sec: 400.0}]
      [result] = Benchmark.normalize_results(raw)
      assert result.batches_per_sec == 400.0
    end

    test "normalizes inference_us_per_batch to inference_ms" do
      raw = [%{name: "MLP", inference_us_per_batch: 1000.0}]
      [result] = Benchmark.normalize_results(raw)
      assert result.inference_ms == 1.0
    end

    test "extracts loss history from epochs array" do
      raw = [%{
        name: "MLP",
        epochs: [
          %{epoch: 1, train_loss: 5.0, val_loss: 4.5},
          %{epoch: 2, train_loss: 3.0, val_loss: 3.2}
        ]
      }]

      [result] = Benchmark.normalize_results(raw)
      assert result.loss_history == [5.0, 3.0]
      assert result.val_loss_history == [4.5, 3.2]
    end
  end

  describe "build_loss_curve_data/1" do
    test "builds data points for each epoch" do
      results = [%{
        name: "MLP",
        loss_history: [5.0, 3.0, 2.5],
        val_loss_history: []
      }]

      data = Benchmark.build_loss_curve_data(results)

      assert length(data) == 3
      assert Enum.at(data, 0) == %{architecture: "MLP", epoch: 1, loss: 5.0, type: "train"}
      assert Enum.at(data, 2) == %{architecture: "MLP", epoch: 3, loss: 2.5, type: "train"}
    end
  end

  describe "build_speed_data/1" do
    test "builds speed comparison data" do
      results = [
        %{name: "MLP", batches_per_sec: 400.0},
        %{name: "LSTM", batches_per_sec: 30.0}
      ]

      data = Benchmark.build_speed_data(results)

      assert length(data) == 2
      assert hd(data) == %{architecture: "MLP", speed: 400.0}
    end
  end

  describe "build_val_loss_data/1" do
    test "sorts by val loss and filters invalid entries" do
      results = [
        %{name: "LSTM", final_val_loss: 4.0},
        %{name: "MLP", final_val_loss: 2.5},
        %{name: "Broken", final_val_loss: nil}
      ]

      data = Benchmark.build_val_loss_data(results)

      assert length(data) == 2
      assert hd(data).architecture == "MLP"  # Lower loss first
    end
  end

  describe "build_overfit_data/1" do
    test "creates train and val entries for each result" do
      results = [%{
        name: "MLP",
        final_val_loss: 2.5,
        final_train_loss: 2.0
      }]

      data = Benchmark.build_overfit_data(results)

      assert length(data) == 2
      assert Enum.find(data, & &1.type == "Train").loss == 2.0
      assert Enum.find(data, & &1.type == "Val").loss == 2.5
    end
  end

  describe "overfit_ratio/1" do
    test "calculates ratio correctly" do
      result = %{final_val_loss: 4.0, final_train_loss: 2.0}
      assert Benchmark.overfit_ratio(result) == 2.0
    end

    test "returns nil for zero train loss" do
      result = %{final_val_loss: 4.0, final_train_loss: 0}
      assert Benchmark.overfit_ratio(result) == nil
    end
  end

  describe "chart generation" do
    test "loss_curves_chart returns VegaLite struct" do
      data = [%{architecture: "MLP", epoch: 1, loss: 5.0, type: "train"}]
      chart = Benchmark.loss_curves_chart(data)
      assert %VegaLite{} = chart
    end

    test "speed_chart returns VegaLite struct" do
      data = [%{architecture: "MLP", speed: 400.0}]
      chart = Benchmark.speed_chart(data)
      assert %VegaLite{} = chart
    end

    test "inference_chart returns VegaLite struct" do
      data = [%{architecture: "MLP", inference_ms: 0.1}]
      chart = Benchmark.inference_chart(data)
      assert %VegaLite{} = chart
    end

    test "val_loss_chart returns VegaLite struct" do
      data = [%{architecture: "MLP", loss: 2.5}]
      chart = Benchmark.val_loss_chart(data)
      assert %VegaLite{} = chart
    end

    test "overfit_chart returns VegaLite struct" do
      data = [%{architecture: "MLP", type: "Train", loss: 2.0}]
      chart = Benchmark.overfit_chart(data)
      assert %VegaLite{} = chart
    end
  end

  describe "to_json_spec/1" do
    test "converts chart to JSON string" do
      data = [%{architecture: "MLP", speed: 400.0}]
      chart = Benchmark.speed_chart(data)
      json = Benchmark.to_json_spec(chart)

      assert is_binary(json)
      assert String.contains?(json, "MLP")
    end
  end

  describe "generate_html/2" do
    test "generates complete HTML with all sections" do
      results = [%{
        name: "MLP",
        final_val_loss: 2.5,
        final_train_loss: 2.0,
        batches_per_sec: 400.0,
        inference_ms: 0.1,
        loss_history: [5.0, 3.0, 2.5],
        val_loss_history: [4.5, 3.2, 2.8]
      }]

      html = Benchmark.generate_html(results)

      assert String.contains?(html, "<!DOCTYPE html>")
      assert String.contains?(html, "Winner: MLP")
      assert String.contains?(html, "vegaEmbed")
      assert String.contains?(html, "2.5")  # val loss
    end

    test "handles empty results" do
      html = Benchmark.generate_html([])
      assert String.contains?(html, "<!DOCTYPE html>")
    end
  end
end
