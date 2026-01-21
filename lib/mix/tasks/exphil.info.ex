defmodule Mix.Tasks.Exphil.Info do
  @moduledoc """
  Show detailed information about an ExPhil checkpoint.

  ## Usage

      mix exphil.info PATH

  ## Examples

      mix exphil.info checkpoints/model.axon
      mix exphil.info checkpoints/model_policy.bin

  """
  use Mix.Task

  alias ExPhil.Training.Output

  @shortdoc "Show information about an ExPhil checkpoint"

  @impl Mix.Task
  def run(args) do
    case args do
      [] ->
        Mix.shell().error("Usage: mix exphil.info PATH")
        System.halt(1)

      [path | _] ->
        show_info(path)
    end
  end

  defp show_info(path) do
    unless File.exists?(path) do
      Mix.shell().error("File not found: #{path}")
      System.halt(1)
    end

    # Load and parse checkpoint
    case load_checkpoint(path) do
      {:ok, data, type} ->
        display_info(path, data, type)

      {:error, reason} ->
        Mix.shell().error("Failed to load checkpoint: #{reason}")
        System.halt(1)
    end
  end

  defp load_checkpoint(path) do
    try do
      binary = File.read!(path)
      data = :erlang.binary_to_term(binary)

      type = cond do
        Map.has_key?(data, :policy_params) -> :checkpoint
        Map.has_key?(data, :params) -> :policy
        true -> :unknown
      end

      {:ok, data, type}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp display_info(path, data, type) do
    stat = File.stat!(path)

    Output.section("Checkpoint Information")

    # Basic info
    Output.kv("Path", path)
    Output.kv("Type", type_name(type))
    Output.kv("Size", Output.format_bytes(stat.size))
    Output.kv("Modified", format_datetime(stat.mtime))

    # Config info
    config = data[:config] || %{}

    Output.puts_raw("")
    Output.puts_raw("  " <> Output.colorize("Architecture:", :bold))

    temporal = config[:temporal] || false
    if temporal do
      Output.kv("Mode", "Temporal (sequence)")
      Output.kv("Backbone", config[:backbone] || :sliding_window)
      Output.kv("Window size", config[:window_size] || 60)
      Output.kv("Num layers", config[:num_layers] || 2)

      if config[:backbone] == :mamba do
        Output.kv("State size", config[:state_size] || 16)
        Output.kv("Expand factor", config[:expand_factor] || 2)
      end
    else
      Output.kv("Mode", "Single-frame (MLP)")
    end

    hidden_sizes = config[:hidden_sizes] || [512, 512]
    Output.kv("Hidden sizes", inspect(hidden_sizes))
    Output.kv("Embed size", config[:embed_size] || "unknown")

    # Discretization
    Output.puts_raw("")
    Output.puts_raw("  " <> Output.colorize("Discretization:", :bold))
    Output.kv("Axis buckets", config[:axis_buckets] || 16)
    Output.kv("Shoulder buckets", config[:shoulder_buckets] || 4)

    # Training info (if checkpoint)
    if type == :checkpoint do
      Output.puts_raw("")
      Output.puts_raw("  " <> Output.colorize("Training:", :bold))
      Output.kv("Step", data[:step] || "unknown")
      Output.kv("Learning rate", config[:learning_rate] || "unknown")
      Output.kv("Batch size", config[:batch_size] || "unknown")

      if config[:label_smoothing] && config[:label_smoothing] > 0 do
        Output.kv("Label smoothing", config[:label_smoothing])
      end

      if config[:focal_loss] do
        Output.kv("Focal loss", "enabled (gamma=#{config[:focal_gamma] || 2.0})")
      end
    end

    # Parameter counts
    params = data[:params] || data[:policy_params]
    if params do
      Output.puts_raw("")
      Output.puts_raw("  " <> Output.colorize("Parameters:", :bold))

      param_count = count_params(params)
      Output.kv("Total params", format_number(param_count))

      # Layer breakdown
      layers = list_layers(params)
      if length(layers) <= 15 do
        Enum.each(layers, fn {name, count} ->
          Output.puts_raw("    #{String.pad_trailing(name, 25)} #{format_number(count)}")
        end)
      else
        Output.puts_raw("    (#{length(layers)} layers)")
      end
    end

    Output.divider()
  end

  defp type_name(:checkpoint), do: "Training checkpoint (.axon)"
  defp type_name(:policy), do: "Exported policy (.bin)"
  defp type_name(_), do: "Unknown"

  defp format_datetime({{year, month, day}, {hour, min, _sec}}) do
    "#{year}-#{pad(month)}-#{pad(day)} #{pad(hour)}:#{pad(min)}"
  end

  defp pad(n), do: String.pad_leading(to_string(n), 2, "0")

  defp count_params(params) when is_map(params) do
    params
    |> Enum.reduce(0, fn {_key, value}, acc ->
      acc + count_params(value)
    end)
  end

  defp count_params(%Nx.Tensor{} = tensor), do: Nx.size(tensor)
  defp count_params(%Axon.ModelState{data: data}), do: count_params(data)
  defp count_params(_), do: 0

  defp list_layers(params) when is_map(params) do
    params
    |> Enum.flat_map(fn {key, value} ->
      case value do
        %Axon.ModelState{data: data} ->
          list_layers(data)
        %{} = nested when not is_struct(nested, Nx.Tensor) ->
          Enum.map(list_layers(nested), fn {k, v} -> {"#{key}.#{k}", v} end)
        %Nx.Tensor{} = tensor ->
          [{to_string(key), Nx.size(tensor)}]
        _ ->
          []
      end
    end)
    |> Enum.filter(fn {_k, v} -> v > 0 end)
  end

  defp list_layers(_), do: []

  defp format_number(n) when n >= 1_000_000 do
    "#{Float.round(n / 1_000_000, 2)}M"
  end

  defp format_number(n) when n >= 1_000 do
    "#{Float.round(n / 1_000, 1)}K"
  end

  defp format_number(n), do: to_string(n)
end
