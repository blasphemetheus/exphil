defmodule ExPhil.Reports.Benchmark do
  @moduledoc """
  Generates benchmark report visualizations using VegaLite.

  Extracts chart generation from scripts into a testable module.
  """

  alias VegaLite

  @doc """
  Generate a training loss curves chart.

  ## Example

      iex> data = [%{architecture: "MLP", epoch: 1, loss: 5.0}, ...]
      iex> chart = loss_curves_chart(data)
  """
  def loss_curves_chart(loss_data, opts \\ []) do
    title = Keyword.get(opts, :title, "Training Loss by Architecture")
    width = Keyword.get(opts, :width, 700)
    height = Keyword.get(opts, :height, 400)

    VegaLite.new(width: width, height: height, title: title)
    |> VegaLite.data_from_values(loss_data)
    |> VegaLite.mark(:line, point: true)
    |> VegaLite.encode_field(:x, "epoch", type: :quantitative, title: "Epoch")
    |> VegaLite.encode_field(:y, "loss", type: :quantitative, title: "Training Loss")
    |> VegaLite.encode_field(:color, "architecture", type: :nominal, title: "Architecture")
    |> VegaLite.encode_field(:stroke_dash, "architecture", type: :nominal)
  end

  @doc """
  Generate a training speed bar chart.
  """
  def speed_chart(speed_data, opts \\ []) do
    title = Keyword.get(opts, :title, "Training Speed (batches/sec, higher is better)")
    width = Keyword.get(opts, :width, 700)
    height = Keyword.get(opts, :height, 300)

    VegaLite.new(width: width, height: height, title: title)
    |> VegaLite.data_from_values(speed_data)
    |> VegaLite.mark(:bar)
    |> VegaLite.encode_field(:x, "architecture", type: :nominal, title: "Architecture", sort: "-y")
    |> VegaLite.encode_field(:y, "speed", type: :quantitative, title: "Batches/sec")
    |> VegaLite.encode_field(:color, "architecture", type: :nominal, legend: nil)
  end

  @doc """
  Generate an inference time bar chart.
  """
  def inference_chart(inference_data, opts \\ []) do
    title = Keyword.get(opts, :title, "Inference Time (ms, lower is better)")
    width = Keyword.get(opts, :width, 700)
    height = Keyword.get(opts, :height, 300)

    VegaLite.new(width: width, height: height, title: title)
    |> VegaLite.data_from_values(inference_data)
    |> VegaLite.mark(:bar)
    |> VegaLite.encode_field(:x, "architecture", type: :nominal, title: "Architecture", sort: "y")
    |> VegaLite.encode_field(:y, "inference_ms", type: :quantitative, title: "Inference (ms)")
    |> VegaLite.encode_field(:color, "architecture", type: :nominal, legend: nil)
  end

  @doc """
  Generate a final validation loss bar chart.
  """
  def val_loss_chart(loss_data, opts \\ []) do
    title = Keyword.get(opts, :title, "Final Validation Loss (lower is better)")
    width = Keyword.get(opts, :width, 700)
    height = Keyword.get(opts, :height, 300)

    VegaLite.new(width: width, height: height, title: title)
    |> VegaLite.data_from_values(loss_data)
    |> VegaLite.mark(:bar)
    |> VegaLite.encode_field(:x, "architecture", type: :nominal, title: "Architecture", sort: "y")
    |> VegaLite.encode_field(:y, "loss",
      type: :quantitative,
      title: "Validation Loss",
      scale: [zero: false]
    )
    |> VegaLite.encode_field(:color, "architecture", type: :nominal, legend: nil)
  end

  @doc """
  Generate a train vs validation loss comparison chart (overfitting indicator).
  """
  def overfit_chart(overfit_data, opts \\ []) do
    title = Keyword.get(opts, :title, "Train vs Validation Loss (overfitting indicator)")
    width = Keyword.get(opts, :width, 700)
    height = Keyword.get(opts, :height, 350)

    VegaLite.new(width: width, height: height, title: title)
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
  end

  @doc """
  Generate theoretical complexity chart.
  """
  def complexity_chart(complexity_data, opts \\ []) do
    title = Keyword.get(opts, :title, "Theoretical Complexity")
    width = Keyword.get(opts, :width, 600)
    height = Keyword.get(opts, :height, 350)

    VegaLite.new(width: width, height: height, title: title)
    |> VegaLite.data_from_values(complexity_data)
    |> VegaLite.mark(:bar, tooltip: true)
    |> VegaLite.encode_field(:x, "architecture",
      type: :nominal,
      title: "Architecture",
      axis: [label_angle: -45]
    )
    |> VegaLite.encode_field(:y, "ops",
      type: :quantitative,
      title: "Relative Operations (log scale)",
      scale: [type: :log, domain: [5, 5000]]
    )
    |> VegaLite.encode_field(:color, "type", type: :nominal, title: "Phase")
    |> VegaLite.encode_field(:x_offset, "type", type: :nominal)
  end

  @doc """
  Parse benchmark results from JSON file or decoded map.

  Returns a normalized list of results with consistent field names.
  """
  def parse_results(json_path) when is_binary(json_path) do
    json_path
    |> File.read!()
    |> Jason.decode!(keys: :atoms)
    |> parse_results()
  end

  def parse_results(raw_data) when is_map(raw_data) do
    # Handle map format with "results" key
    if Map.has_key?(raw_data, :results) do
      raw_data.results
    else
      [raw_data]
    end
    |> normalize_results()
  end

  def parse_results(raw_data) when is_list(raw_data) do
    # Filter to only actual results (have :name key)
    raw_data
    |> Enum.filter(fn r -> is_map(r) && Map.has_key?(r, :name) end)
    |> normalize_results()
  end

  @doc """
  Normalize results to have consistent field names.
  """
  def normalize_results(results) do
    Enum.map(results, fn r ->
      %{
        name: r[:name],
        final_val_loss: r[:final_val_loss],
        final_train_loss: r[:final_train_loss],
        # Handle both field name variants
        batches_per_sec: r[:avg_batches_per_sec] || r[:batches_per_sec] || 0,
        inference_ms: get_inference_ms(r),
        theoretical_complexity: r[:theoretical_complexity],
        epochs: r[:epochs] || [],
        loss_history: get_loss_history(r, :train),
        val_loss_history: get_loss_history(r, :val)
      }
    end)
  end

  defp get_inference_ms(r) do
    cond do
      r[:inference_time_ms] -> r[:inference_time_ms]
      r[:inference_us_per_batch] -> r[:inference_us_per_batch] / 1000.0
      true -> 0
    end
  end

  defp get_loss_history(r, :train) do
    cond do
      r[:loss_history] -> r[:loss_history]
      r[:epochs] -> Enum.map(r.epochs, & &1.train_loss)
      true -> []
    end
  end

  defp get_loss_history(r, :val) do
    cond do
      r[:val_loss_history] -> r[:val_loss_history]
      r[:epochs] -> Enum.map(r.epochs, & &1.val_loss)
      true -> []
    end
  end

  @doc """
  Build loss curve data from normalized results.
  """
  def build_loss_curve_data(results) do
    results
    |> Enum.flat_map(fn r ->
      r.loss_history
      |> Enum.with_index(1)
      |> Enum.map(fn {loss, epoch} ->
        %{architecture: r.name, epoch: epoch, loss: loss, type: "train"}
      end)
    end)
  end

  @doc """
  Build validation loss curve data from normalized results.
  """
  def build_val_loss_curve_data(results) do
    results
    |> Enum.flat_map(fn r ->
      r.val_loss_history
      |> Enum.with_index(1)
      |> Enum.map(fn {loss, epoch} ->
        %{architecture: r.name, epoch: epoch, loss: loss, type: "val"}
      end)
    end)
  end

  @doc """
  Build speed comparison data from normalized results.
  """
  def build_speed_data(results) do
    Enum.map(results, fn r ->
      %{architecture: r.name, speed: r.batches_per_sec}
    end)
  end

  @doc """
  Build inference comparison data from normalized results.
  """
  def build_inference_data(results) do
    Enum.map(results, fn r ->
      %{architecture: r.name, inference_ms: r.inference_ms}
    end)
  end

  @doc """
  Build final validation loss data from normalized results.
  """
  def build_val_loss_data(results) do
    results
    |> Enum.filter(fn r -> r.final_val_loss && is_number(r.final_val_loss) end)
    |> Enum.sort_by(& &1.final_val_loss)
    |> Enum.map(fn r ->
      %{architecture: r.name, loss: r.final_val_loss}
    end)
  end

  @doc """
  Build overfitting comparison data from normalized results.
  """
  def build_overfit_data(results) do
    results
    |> Enum.filter(fn r ->
      r.final_val_loss && is_number(r.final_val_loss) &&
      r.final_train_loss && is_number(r.final_train_loss)
    end)
    |> Enum.sort_by(& &1.final_val_loss)
    |> Enum.flat_map(fn r ->
      [
        %{architecture: r.name, type: "Train", loss: r.final_train_loss},
        %{architecture: r.name, type: "Val", loss: r.final_val_loss}
      ]
    end)
  end

  @doc """
  Calculate overfit ratio (val_loss / train_loss).
  """
  def overfit_ratio(result) do
    if result.final_train_loss && result.final_train_loss > 0 do
      Float.round(result.final_val_loss / result.final_train_loss, 2)
    else
      nil
    end
  end

  @doc """
  Convert a VegaLite chart to JSON spec string.
  """
  def to_json_spec(chart) do
    chart |> VegaLite.to_spec() |> Jason.encode!()
  end

  @doc """
  Generate complete HTML report from results.
  """
  def generate_html(results, opts \\ []) do
    title = Keyword.get(opts, :title, "Architecture Benchmark Report")

    # Sort by val loss for display
    sorted = results
      |> Enum.filter(fn r -> r.final_val_loss && is_number(r.final_val_loss) end)
      |> Enum.sort_by(& &1.final_val_loss)

    best = List.first(sorted)

    # Build chart data
    loss_data = build_loss_curve_data(results)
    speed_data = build_speed_data(results)
    inference_data = build_inference_data(results)
    val_loss_data = build_val_loss_data(results)
    overfit_data = build_overfit_data(results)

    # Generate charts
    loss_spec = loss_curves_chart(loss_data) |> to_json_spec()
    speed_spec = speed_chart(speed_data) |> to_json_spec()
    inference_spec = inference_chart(inference_data) |> to_json_spec()
    val_loss_spec = val_loss_chart(val_loss_data) |> to_json_spec()
    overfit_spec = overfit_chart(overfit_data) |> to_json_spec()

    # Build results table
    results_table = build_results_table(sorted)

    # Generate HTML
    """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>#{title}</title>
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
      <h1>#{title}</h1>

      #{if best do
        """
        <div class="summary">
          <span class="winner">Winner: #{best.name}</span><br>
          <strong>Validation Loss:</strong> #{Float.round(best.final_val_loss, 4)} |
          <strong>Speed:</strong> #{Float.round(best.batches_per_sec, 1)} batches/sec |
          <strong>Inference:</strong> #{Float.round(best.inference_ms, 2)}ms
        </div>
        """
      else
        ""
      end}

      <h2>Results Table</h2>
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

      <h2>Final Validation Loss</h2>
      <div id="val_loss" class="plot"></div>

      <h2>Overfitting Analysis (Train vs Val)</h2>
      <div class="warning">
        <strong>High overfit ratio = model memorizing training data.</strong>
        Ratio close to 1.0x is ideal.
      </div>
      <div id="overfit" class="plot"></div>

      <h2>Training Loss Curves</h2>
      <div id="loss_curves" class="plot"></div>

      <h2>Training Speed</h2>
      <div id="speed" class="plot"></div>

      <h2>Inference Time</h2>
      <div id="inference" class="plot"></div>

      <script>
        vegaEmbed('#loss_curves', #{loss_spec}, {actions: false});
        vegaEmbed('#speed', #{speed_spec}, {actions: false});
        vegaEmbed('#inference', #{inference_spec}, {actions: false});
        vegaEmbed('#val_loss', #{val_loss_spec}, {actions: false});
        vegaEmbed('#overfit', #{overfit_spec}, {actions: false});
      </script>
    </body>
    </html>
    """
  end

  defp build_results_table(sorted_results) do
    sorted_results
    |> Enum.with_index(1)
    |> Enum.map(fn {r, rank} ->
      ratio = overfit_ratio(r)
      ratio_str = if ratio, do: "#{ratio}x", else: "N/A"

      """
      <tr>
        <td>#{rank}</td>
        <td>#{r.name}</td>
        <td>#{Float.round(r.final_val_loss, 4)}</td>
        <td>#{Float.round(r.final_train_loss || 0, 4)}</td>
        <td>#{ratio_str}</td>
        <td>#{Float.round(r.batches_per_sec, 1)}</td>
        <td>#{Float.round(r.inference_ms, 2)}ms</td>
      </tr>
      """
    end)
    |> Enum.join("\n")
  end
end
