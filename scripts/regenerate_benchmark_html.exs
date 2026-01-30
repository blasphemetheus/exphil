#!/usr/bin/env elixir
# Regenerate HTML visualization from benchmark JSON results
#
# Usage:
#   mix run scripts/regenerate_benchmark_html.exs --json checkpoints/benchmark_results_20260130_002013.json
#

Mix.install([
  {:jason, "~> 1.4"},
  {:vega_lite, "~> 0.1.11"}
])

alias VegaLite

# Parse args
args = System.argv()

json_path = case Enum.find_index(args, &(&1 == "--json")) do
  nil ->
    # Try to find most recent benchmark JSON
    case Path.wildcard("checkpoints/benchmark_results_*.json") |> Enum.sort() |> List.last() do
      nil ->
        IO.puts("Usage: mix run scripts/regenerate_benchmark_html.exs --json <path>")
        System.halt(1)
      path -> path
    end
  idx -> Enum.at(args, idx + 1)
end

IO.puts("Loading results from: #{json_path}")

# Load JSON
results = json_path |> File.read!() |> Jason.decode!(keys: :atoms)

# Sort by validation loss
sorted_results = results
  |> Enum.filter(fn r -> r.final_val_loss && !is_nil(r.final_val_loss) && r.final_val_loss != :nan end)
  |> Enum.sort_by(& &1.final_val_loss)

IO.puts("Loaded #{length(results)} results, #{length(sorted_results)} with valid val_loss")

# Build loss curve data for line chart
loss_curve_data = results
  |> Enum.flat_map(fn r ->
    r.loss_history
    |> Enum.with_index(1)
    |> Enum.map(fn {loss, epoch} ->
      %{architecture: r.name, epoch: epoch, loss: loss, type: "train"}
    end)
  end)

val_loss_data = results
  |> Enum.flat_map(fn r ->
    (r[:val_loss_history] || [])
    |> Enum.with_index(1)
    |> Enum.map(fn {loss, epoch} ->
      %{architecture: r.name, epoch: epoch, loss: loss, type: "val"}
    end)
  end)

all_loss_data = loss_curve_data ++ val_loss_data

# Training loss curves
comparison_plot =
  VegaLite.new(width: 700, height: 400, title: "Training Loss by Architecture")
  |> VegaLite.data_from_values(loss_curve_data)
  |> VegaLite.mark(:line, point: true)
  |> VegaLite.encode_field(:x, "epoch", type: :quantitative, title: "Epoch")
  |> VegaLite.encode_field(:y, "loss", type: :quantitative, title: "Training Loss")
  |> VegaLite.encode_field(:color, "architecture", type: :nominal, title: "Architecture")
  |> VegaLite.encode_field(:stroke_dash, "architecture", type: :nominal)

# Speed comparison
speed_data = results |> Enum.map(fn r -> %{architecture: r.name, speed: r.batches_per_sec} end)

speed_plot =
  VegaLite.new(width: 700, height: 300, title: "Training Speed (batches/sec, higher is better)")
  |> VegaLite.data_from_values(speed_data)
  |> VegaLite.mark(:bar)
  |> VegaLite.encode_field(:x, "architecture", type: :nominal, title: "Architecture", sort: "-y")
  |> VegaLite.encode_field(:y, "speed", type: :quantitative, title: "Batches/sec")
  |> VegaLite.encode_field(:color, "architecture", type: :nominal, legend: nil)

# Inference time comparison
inference_data = results |> Enum.map(fn r ->
  %{architecture: r.name, inference_ms: r.inference_time_ms || 0}
end)

inference_plot =
  VegaLite.new(width: 700, height: 300, title: "Inference Time (ms, lower is better)")
  |> VegaLite.data_from_values(inference_data)
  |> VegaLite.mark(:bar)
  |> VegaLite.encode_field(:x, "architecture", type: :nominal, title: "Architecture", sort: "y")
  |> VegaLite.encode_field(:y, "inference_ms", type: :quantitative, title: "Inference (ms)")
  |> VegaLite.encode_field(:color, "architecture", type: :nominal, legend: nil)

# Final validation loss bar chart
loss_bar_data = sorted_results
  |> Enum.map(fn r -> %{architecture: r.name, loss: r.final_val_loss} end)

loss_bar_plot =
  VegaLite.new(width: 700, height: 300, title: "Final Validation Loss (lower is better)")
  |> VegaLite.data_from_values(loss_bar_data)
  |> VegaLite.mark(:bar)
  |> VegaLite.encode_field(:x, "architecture", type: :nominal, title: "Architecture", sort: "y")
  |> VegaLite.encode_field(:y, "loss",
    type: :quantitative,
    title: "Validation Loss",
    scale: [zero: false]
  )
  |> VegaLite.encode_field(:color, "architecture", type: :nominal, legend: nil)

# Train vs Val loss comparison (overfitting indicator)
overfit_data = sorted_results
  |> Enum.flat_map(fn r ->
    [
      %{architecture: r.name, type: "Train", loss: r.final_train_loss},
      %{architecture: r.name, type: "Val", loss: r.final_val_loss}
    ]
  end)

overfit_plot =
  VegaLite.new(width: 700, height: 350, title: "Train vs Validation Loss (overfitting indicator)")
  |> VegaLite.data_from_values(overfit_data)
  |> VegaLite.mark(:bar, tooltip: true)
  |> VegaLite.encode_field(:x, "architecture",
    type: :nominal,
    title: "Architecture",
    axis: [label_angle: -45]
  )
  |> VegaLite.encode_field(:y, "loss",
    type: :quantitative,
    title: "Loss",
    scale: [zero: false]
  )
  |> VegaLite.encode_field(:color, "type", type: :nominal, title: "Phase")
  |> VegaLite.encode_field(:x_offset, "type", type: :nominal)

# Convert plots to JSON specs
spec = comparison_plot |> VegaLite.to_spec() |> Jason.encode!()
speed_spec = speed_plot |> VegaLite.to_spec() |> Jason.encode!()
inference_spec = inference_plot |> VegaLite.to_spec() |> Jason.encode!()
loss_bar_spec = loss_bar_plot |> VegaLite.to_spec() |> Jason.encode!()
overfit_spec = overfit_plot |> VegaLite.to_spec() |> Jason.encode!()

# Build results table
best = List.first(sorted_results)
results_table = sorted_results
  |> Enum.with_index(1)
  |> Enum.map(fn {r, rank} ->
    overfit_ratio = if r.final_train_loss > 0, do: Float.round(r.final_val_loss / r.final_train_loss, 2), else: "N/A"
    """
    <tr>
      <td>#{rank}</td>
      <td>#{r.name}</td>
      <td>#{Float.round(r.final_val_loss, 4)}</td>
      <td>#{Float.round(r.final_train_loss, 4)}</td>
      <td>#{overfit_ratio}x</td>
      <td>#{Float.round(r.batches_per_sec, 1)}</td>
      <td>#{Float.round(r.inference_time_ms || 0, 1)}ms</td>
    </tr>
    """
  end)
  |> Enum.join("\n")

html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Architecture Benchmark Report</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 40px; max-width: 1000px; margin: 0 auto; }
    h1 { color: #333; }
    h2 { color: #555; margin-top: 40px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
    .summary { background: #d4edda; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #28a745; }
    .warning { background: #fff3cd; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffc107; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #333; color: white; }
    tr:nth-child(1) td { background: #d4edda; font-weight: bold; }
    tr:hover { background: #f5f5f5; }
    .plot { margin: 30px 0; }
    .winner { color: #28a745; font-weight: bold; font-size: 1.2em; }
  </style>
</head>
<body>
  <h1>🏆 Architecture Benchmark Report</h1>

  <div class="summary">
    <span class="winner">Winner: #{best.name}</span><br>
    <strong>Validation Loss:</strong> #{Float.round(best.final_val_loss, 4)} |
    <strong>Speed:</strong> #{Float.round(best.batches_per_sec, 1)} batches/sec |
    <strong>Inference:</strong> #{Float.round(best.inference_time_ms || 0, 1)}ms
  </div>

  <h2>📊 Results Table</h2>
  <table>
    <tr>
      <th>Rank</th>
      <th>Architecture</th>
      <th>Val Loss</th>
      <th>Train Loss</th>
      <th>Overfit Ratio</th>
      <th>Speed (b/s)</th>
      <th>Inference</th>
    </tr>
    #{results_table}
  </table>

  <h2>📉 Final Validation Loss</h2>
  <div id="loss_bar" class="plot"></div>

  <h2>⚠️ Overfitting Analysis (Train vs Val)</h2>
  <div class="warning">
    <strong>High overfit ratio = model memorizing training data.</strong>
    Ratio close to 1.0x is ideal. Mamba shows severe overfitting (val >> train).
  </div>
  <div id="overfit" class="plot"></div>

  <h2>📈 Training Loss Curves</h2>
  <div id="loss_curves" class="plot"></div>

  <h2>⚡ Training Speed</h2>
  <div id="speed" class="plot"></div>

  <h2>🎮 Inference Time</h2>
  <div id="inference" class="plot"></div>

  <script>
    vegaEmbed('#loss_curves', #{spec}, {actions: false});
    vegaEmbed('#speed', #{speed_spec}, {actions: false});
    vegaEmbed('#inference', #{inference_spec}, {actions: false});
    vegaEmbed('#loss_bar', #{loss_bar_spec}, {actions: false});
    vegaEmbed('#overfit', #{overfit_spec}, {actions: false});
  </script>
</body>
</html>
"""

# Save HTML
output_path = String.replace(json_path, ".json", ".html")
File.write!(output_path, html)
IO.puts("✓ HTML report saved to: #{output_path}")
