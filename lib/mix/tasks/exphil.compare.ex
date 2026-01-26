defmodule Mix.Tasks.Exphil.Compare do
  @moduledoc """
  Compare two ExPhil checkpoints side-by-side.

  Shows differences in configuration, architecture, and training metrics
  between two models.

  ## Usage

      mix exphil.compare PATH_A PATH_B

  ## Options

    * `--all` - Show all config fields, not just differences
    * `--metrics` - Focus on training metrics comparison

  ## Examples

      mix exphil.compare checkpoints/model_v1.axon checkpoints/model_v2.axon
      mix exphil.compare checkpoints/old.axon checkpoints/new.axon --all

  """
  use Mix.Task

  alias ExPhil.Training.Output

  @shortdoc "Compare two ExPhil checkpoints"

  # Config fields grouped by category for organized display
  @architecture_fields [
    :temporal,
    :backbone,
    :window_size,
    :num_layers,
    :hidden_sizes,
    :embed_size,
    :state_size,
    :expand_factor,
    :conv_size,
    :head_dim,
    :num_heads,
    :hidden_size,
    :dropout,
    :layer_norm,
    :residual
  ]

  @training_fields [
    :learning_rate,
    :batch_size,
    :epochs,
    :accumulation_steps,
    :lr_schedule,
    :warmup_steps,
    :decay_steps,
    :max_grad_norm,
    :optimizer,
    :precision
  ]

  @regularization_fields [
    :label_smoothing,
    :focal_loss,
    :focal_gamma,
    :augment,
    :mirror_prob,
    :noise_prob,
    :noise_scale,
    :ema,
    :ema_decay
  ]

  @discretization_fields [:axis_buckets, :shoulder_buckets]

  @impl Mix.Task
  def run(args) do
    {opts, paths, _} =
      OptionParser.parse(args,
        strict: [all: :boolean, metrics: :boolean]
      )

    case paths do
      [path_a, path_b] ->
        compare(path_a, path_b, opts)

      _ ->
        Mix.shell().error("Usage: mix exphil.compare PATH_A PATH_B [--all] [--metrics]")
        System.halt(1)
    end
  end

  defp compare(path_a, path_b, opts) do
    # Load both checkpoints
    with {:ok, data_a, type_a} <- load_checkpoint(path_a),
         {:ok, data_b, type_b} <- load_checkpoint(path_b) do
      name_a = Path.basename(path_a)
      name_b = Path.basename(path_b)

      Output.puts_raw("""

      ╔════════════════════════════════════════════════════════════════╗
      ║                    Model Comparison                            ║
      ╚════════════════════════════════════════════════════════════════╝
      """)

      # Basic info
      Output.puts_raw("  " <> Output.colorize("Files:", :bold))
      Output.puts_raw("    A: #{path_a}")
      Output.puts_raw("    B: #{path_b}")
      Output.puts_raw("")

      stat_a = File.stat!(path_a)
      stat_b = File.stat!(path_b)

      show_comparison_row("Type", type_name(type_a), type_name(type_b))

      show_comparison_row(
        "Size",
        Output.format_bytes(stat_a.size),
        Output.format_bytes(stat_b.size)
      )

      config_a = data_a[:config] || %{}
      config_b = data_b[:config] || %{}

      show_all = Keyword.get(opts, :all, false)
      show_metrics = Keyword.get(opts, :metrics, false)

      # Architecture comparison
      Output.puts_raw("")
      Output.puts_raw("  " <> Output.colorize("Architecture:", :bold))
      compare_fields(config_a, config_b, @architecture_fields, name_a, name_b, show_all)

      # Training comparison
      Output.puts_raw("")
      Output.puts_raw("  " <> Output.colorize("Training:", :bold))
      compare_fields(config_a, config_b, @training_fields, name_a, name_b, show_all)

      # Regularization
      Output.puts_raw("")
      Output.puts_raw("  " <> Output.colorize("Regularization:", :bold))
      compare_fields(config_a, config_b, @regularization_fields, name_a, name_b, show_all)

      # Discretization
      Output.puts_raw("")
      Output.puts_raw("  " <> Output.colorize("Discretization:", :bold))
      compare_fields(config_a, config_b, @discretization_fields, name_a, name_b, show_all)

      # Parameter counts
      params_a = data_a[:params] || data_a[:policy_params]
      params_b = data_b[:params] || data_b[:policy_params]

      if params_a && params_b do
        Output.puts_raw("")
        Output.puts_raw("  " <> Output.colorize("Parameters:", :bold))
        count_a = count_params(params_a)
        count_b = count_params(params_b)
        show_comparison_row("Total params", format_number(count_a), format_number(count_b))

        diff = count_b - count_a
        diff_str = if diff >= 0, do: "+#{format_number(diff)}", else: format_number(diff)
        diff_pct = if count_a > 0, do: Float.round(diff / count_a * 100, 1), else: 0.0
        Output.puts_raw("    #{Output.colorize("Difference:", :dim)} #{diff_str} (#{diff_pct}%)")
      end

      # Training metrics comparison (if both are checkpoints)
      if show_metrics || (type_a == :checkpoint && type_b == :checkpoint) do
        Output.puts_raw("")
        Output.puts_raw("  " <> Output.colorize("Training Progress:", :bold))

        step_a = data_a[:step]
        step_b = data_b[:step]

        if step_a || step_b do
          show_comparison_row("Step", step_a || "?", step_b || "?")
        end

        # Metrics from checkpoint
        metrics_a = data_a[:metrics] || %{}
        metrics_b = data_b[:metrics] || %{}

        if metrics_a[:train_loss] || metrics_b[:train_loss] do
          loss_a = format_loss(metrics_a[:train_loss])
          loss_b = format_loss(metrics_b[:train_loss])
          show_comparison_row("Train loss", loss_a, loss_b)
        end

        if metrics_a[:val_loss] || metrics_b[:val_loss] do
          loss_a = format_loss(metrics_a[:val_loss])
          loss_b = format_loss(metrics_b[:val_loss])
          show_comparison_row("Val loss", loss_a, loss_b)
        end

        if metrics_a[:best_loss] || metrics_b[:best_loss] do
          loss_a = format_loss(metrics_a[:best_loss])
          loss_b = format_loss(metrics_b[:best_loss])
          show_comparison_row("Best loss", loss_a, loss_b)
        end
      end

      # Summary
      Output.puts_raw("")
      Output.divider()

      # Show summary of differences
      diff_count =
        count_differences(
          config_a,
          config_b,
          @architecture_fields ++
            @training_fields ++ @regularization_fields ++ @discretization_fields
        )

      if diff_count == 0 do
        Output.puts_raw("  " <> Output.colorize("Configurations are identical", :green))
      else
        Output.puts_raw(
          "  " <> Output.colorize("#{diff_count} configuration difference(s)", :yellow)
        )
      end

      Output.puts_raw("")
    else
      {:error, path, reason} ->
        Mix.shell().error("Failed to load #{path}: #{reason}")
        System.halt(1)
    end
  end

  defp load_checkpoint(path) do
    unless File.exists?(path) do
      {:error, path, "File not found"}
    else
      try do
        binary = File.read!(path)
        data = :erlang.binary_to_term(binary)

        type =
          cond do
            Map.has_key?(data, :policy_params) -> :checkpoint
            Map.has_key?(data, :params) -> :policy
            true -> :unknown
          end

        {:ok, data, type}
      rescue
        e -> {:error, path, Exception.message(e)}
      end
    end
  end

  defp compare_fields(config_a, config_b, fields, _name_a, _name_b, show_all) do
    fields
    |> Enum.each(fn field ->
      val_a = get_config_value(config_a, field)
      val_b = get_config_value(config_b, field)

      # Show if different, or if --all and at least one has a value
      should_show = val_a != val_b || (show_all && (val_a != nil || val_b != nil))

      if should_show do
        show_comparison_row(format_field_name(field), format_value(val_a), format_value(val_b))
      end
    end)
  end

  defp get_config_value(config, field) when is_map(config) do
    # Try both atom and string keys
    Map.get(config, field) || Map.get(config, to_string(field))
  end

  defp get_config_value(config, field) when is_list(config) do
    Keyword.get(config, field)
  end

  defp get_config_value(_, _), do: nil

  defp show_comparison_row(label, val_a, val_b) do
    label_str = String.pad_trailing(to_string(label), 18)
    val_a_str = String.pad_trailing(format_display(val_a), 20)
    val_b_str = format_display(val_b)

    # Highlight if different
    if val_a != val_b do
      Output.puts_raw(
        "    #{label_str} #{Output.colorize(val_a_str, :yellow)} #{Output.colorize(val_b_str, :cyan)}"
      )
    else
      Output.puts_raw("    #{label_str} #{val_a_str} #{val_b_str}")
    end
  end

  defp format_field_name(field) do
    field
    |> to_string()
    |> String.replace("_", " ")
    |> String.capitalize()
  end

  defp format_value(nil), do: "-"
  defp format_value(true), do: "yes"
  defp format_value(false), do: "no"
  defp format_value(list) when is_list(list), do: inspect(list, charlists: :as_lists)
  defp format_value(val), do: to_string(val)

  defp format_display(val) when is_binary(val), do: val
  defp format_display(val), do: to_string(val)

  defp format_loss(nil), do: "-"
  defp format_loss(loss) when is_float(loss), do: :erlang.float_to_binary(loss, decimals: 6)
  defp format_loss(loss), do: to_string(loss)

  defp count_differences(config_a, config_b, fields) do
    fields
    |> Enum.count(fn field ->
      val_a = get_config_value(config_a, field)
      val_b = get_config_value(config_b, field)
      val_a != val_b && (val_a != nil || val_b != nil)
    end)
  end

  defp type_name(:checkpoint), do: "checkpoint"
  defp type_name(:policy), do: "policy"
  defp type_name(_), do: "unknown"

  defp count_params(params) when is_map(params) do
    params
    |> Enum.reduce(0, fn {_key, value}, acc ->
      acc + count_params(value)
    end)
  end

  defp count_params(%Nx.Tensor{} = tensor), do: Nx.size(tensor)
  defp count_params(%Axon.ModelState{data: data}), do: count_params(data)
  defp count_params(_), do: 0

  defp format_number(n) when n >= 1_000_000, do: "#{Float.round(n / 1_000_000, 2)}M"
  defp format_number(n) when n >= 1_000, do: "#{Float.round(n / 1_000, 1)}K"
  defp format_number(n), do: to_string(n)
end
