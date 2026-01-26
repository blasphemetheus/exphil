defmodule ExPhil.Training.Plots do
  @moduledoc """
  Training visualization utilities using VegaLite.

  Generates loss curves and other training plots for quick visual analysis.
  """

  alias VegaLite, as: Vl

  @doc """
  Create a loss curve plot from training history.

  ## Parameters

  - `history` - List of maps with at least `:epoch` and `:train_loss` keys.
                Optional: `:val_loss` for validation curve.

  ## Options

  - `:title` - Plot title (default: "Training Loss")
  - `:width` - Plot width in pixels (default: 600)
  - `:height` - Plot height in pixels (default: 400)

  ## Examples

      iex> history = [
      ...>   %{epoch: 1, train_loss: 0.5, val_loss: 0.6},
      ...>   %{epoch: 2, train_loss: 0.3, val_loss: 0.4}
      ...> ]
      iex> Plots.loss_curve(history)
      %VegaLite{...}
  """
  @spec loss_curve([map()], keyword()) :: VegaLite.t()
  def loss_curve(history, opts \\ []) do
    title = Keyword.get(opts, :title, "Training Loss")
    width = Keyword.get(opts, :width, 600)
    height = Keyword.get(opts, :height, 400)

    # Check if we have validation loss
    has_val = Enum.any?(history, &Map.has_key?(&1, :val_loss))

    # Transform data for VegaLite (long format for multi-line plot)
    data =
      if has_val do
        history
        |> Enum.flat_map(fn entry ->
          [
            %{epoch: entry.epoch, loss: entry.train_loss, type: "train"},
            %{epoch: entry.epoch, loss: entry[:val_loss], type: "val"}
          ]
        end)
        |> Enum.reject(fn %{loss: l} -> is_nil(l) end)
      else
        Enum.map(history, fn entry ->
          %{epoch: entry.epoch, loss: entry.train_loss, type: "train"}
        end)
      end

    Vl.new(width: width, height: height, title: title)
    |> Vl.data_from_values(data)
    |> Vl.mark(:line, point: true)
    |> Vl.encode_field(:x, "epoch", type: :quantitative, title: "Epoch")
    |> Vl.encode_field(:y, "loss", type: :quantitative, title: "Loss", scale: [zero: false])
    |> Vl.encode_field(:color, "type",
      type: :nominal,
      title: "Type",
      scale: [domain: ["train", "val"], range: ["#1f77b4", "#ff7f0e"]]
    )
  end

  @doc """
  Create a learning rate schedule plot.

  ## Parameters

  - `schedule_fn` - Function that takes step number and returns LR
  - `steps` - Total number of steps to plot

  ## Options

  - `:title` - Plot title (default: "Learning Rate Schedule")
  - `:width` - Plot width in pixels (default: 600)
  - `:height` - Plot height in pixels (default: 300)
  """
  @spec lr_schedule(function(), non_neg_integer(), keyword()) :: VegaLite.t()
  def lr_schedule(schedule_fn, steps, opts \\ []) do
    title = Keyword.get(opts, :title, "Learning Rate Schedule")
    width = Keyword.get(opts, :width, 600)
    height = Keyword.get(opts, :height, 300)

    # Sample points (don't need every step)
    sample_rate = max(1, div(steps, 200))

    data =
      0..steps
      |> Enum.take_every(sample_rate)
      |> Enum.map(fn step ->
        lr = schedule_fn.(Nx.tensor(step)) |> Nx.to_number()
        %{step: step, learning_rate: lr}
      end)

    Vl.new(width: width, height: height, title: title)
    |> Vl.data_from_values(data)
    |> Vl.mark(:line)
    |> Vl.encode_field(:x, "step", type: :quantitative, title: "Step")
    |> Vl.encode_field(:y, "learning_rate",
      type: :quantitative,
      title: "Learning Rate",
      axis: [format: ".1e"]
    )
  end

  @doc """
  Save a VegaLite plot to a PNG file.

  Requires the `vix` library for PNG export. If not available,
  saves as HTML instead.

  ## Options

  - `:format` - Output format: :png, :svg, or :html (default: :html)

  ## Examples

      iex> plot = Plots.loss_curve(history)
      iex> Plots.save!(plot, "checkpoints/loss_curve.png")
      :ok
  """
  @spec save!(VegaLite.t(), String.t(), keyword()) :: :ok
  def save!(plot, path, opts \\ []) do
    format = Keyword.get(opts, :format, detect_format(path))

    case format do
      :html ->
        html = to_html(plot)
        File.write!(path, html)

      :svg ->
        # SVG export requires vega_lite_convert package, fall back to HTML
        html_path = String.replace(path, ".svg", ".html")
        html = to_html(plot)
        File.write!(html_path, html)
        require Logger

        Logger.warning(
          "SVG export requires vega_lite_convert library, saved as HTML: #{html_path}"
        )

      :png ->
        # PNG requires additional dependency, fall back to HTML
        html_path = String.replace(path, ".png", ".html")
        html = to_html(plot)
        File.write!(html_path, html)
        require Logger
        Logger.warning("PNG export requires vix library, saved as HTML: #{html_path}")
    end

    :ok
  end

  defp detect_format(path) do
    case Path.extname(path) do
      ".png" -> :png
      ".svg" -> :svg
      ".html" -> :html
      _ -> :html
    end
  end

  @doc """
  Convert a VegaLite plot to standalone HTML.
  """
  @spec to_html(VegaLite.t()) :: String.t()
  def to_html(plot) do
    spec = Vl.to_spec(plot) |> Jason.encode!()

    """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Training Plot</title>
      <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
      <style>
        body { font-family: sans-serif; padding: 20px; }
        #vis { margin: 0 auto; }
      </style>
    </head>
    <body>
      <div id="vis"></div>
      <script>
        var spec = #{spec};
        vegaEmbed('#vis', spec);
      </script>
    </body>
    </html>
    """
  end

  @doc """
  Create a summary report with multiple plots.

  Generates an HTML page with loss curve and optional additional metrics.

  ## Parameters

  - `history` - Training history
  - `path` - Output path for HTML file

  ## Options

  - `:title` - Report title
  - `:metadata` - Additional metadata to display (keyword list or map)
  """
  @spec save_report!(list(), String.t(), keyword()) :: :ok
  def save_report!(history, path, opts \\ []) do
    title = Keyword.get(opts, :title, "Training Report")
    metadata = Keyword.get(opts, :metadata, [])

    loss_plot = loss_curve(history)
    loss_spec = Vl.to_spec(loss_plot) |> Jason.encode!()

    # Format metadata
    metadata_html =
      if Enum.empty?(metadata) do
        ""
      else
        items = Enum.map(metadata, fn {k, v} -> "<li><strong>#{k}:</strong> #{v}</li>" end)
        "<ul>#{Enum.join(items)}</ul>"
      end

    # Get final metrics
    final = List.last(history) || %{}
    final_train = Map.get(final, :train_loss, "N/A")
    final_val = Map.get(final, :val_loss, "N/A")

    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>#{title}</title>
      <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
      <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }
        h1 { color: #333; }
        .metrics { background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .metrics h3 { margin-top: 0; }
        #loss-plot { margin: 20px 0; }
        .footer { color: #666; font-size: 0.9em; margin-top: 30px; }
      </style>
    </head>
    <body>
      <h1>#{title}</h1>

      <div class="metrics">
        <h3>Final Metrics</h3>
        <p>Train Loss: <strong>#{format_loss(final_train)}</strong></p>
        <p>Val Loss: <strong>#{format_loss(final_val)}</strong></p>
        <p>Epochs: <strong>#{length(history)}</strong></p>
        #{metadata_html}
      </div>

      <div id="loss-plot"></div>

      <div class="footer">
        Generated by ExPhil Training at #{DateTime.utc_now() |> Calendar.strftime("%Y-%m-%d %H:%M:%S UTC")}
      </div>

      <script>
        vegaEmbed('#loss-plot', #{loss_spec});
      </script>
    </body>
    </html>
    """

    File.write!(path, html)
    :ok
  end

  defp format_loss(loss) when is_float(loss), do: Float.round(loss, 4)
  defp format_loss(loss), do: loss
end
