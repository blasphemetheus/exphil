#!/usr/bin/env elixir
# Model Registry Management
#
# Usage:
#   mix run scripts/registry.exs [command] [options]
#
# Commands:
#   list                    - List all registered models
#   show ID                 - Show details for a model by ID or name
#   tag ID TAG [TAG...]     - Add tags to a model
#   untag ID TAG [TAG...]   - Remove tags from a model
#   best                    - Show the best model by loss
#   delete ID               - Remove a model from registry (keeps files)
#   delete ID --files       - Remove a model and delete its files
#
# Options:
#   --tags TAG,TAG          - Filter by tags (comma-separated)
#   --backbone TYPE         - Filter by backbone (mlp, mamba, lstm, etc.)
#   --limit N               - Limit number of results
#   --json                  - Output as JSON

alias ExPhil.Training.Registry
alias ExPhil.Training.Output

args = System.argv()

defmodule RegistryUI do
  @moduledoc "UI helpers specific to registry display"

  def format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  def format_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"
  def format_bytes(bytes), do: "#{Float.round(bytes / 1024 / 1024, 1)} MB"

  def format_model(model, opts \\ []) do
    verbose = Keyword.get(opts, :verbose, false)

    status = if File.exists?(model.checkpoint_path), do: "✓", else: "✗"
    tags = if model.tags != [], do: " [#{Enum.join(model.tags, ", ")}]", else: ""
    loss = get_in(model, [:metrics, :final_loss])
    loss_str = if loss, do: " loss=#{loss}", else: ""

    line = "#{status} #{model.name} (#{model.id})#{tags}#{loss_str}"

    if verbose do
      """
      #{line}
        Path: #{model.checkpoint_path}
        Created: #{model.created_at}
        Backbone: #{get_in(model, [:training_config, :backbone]) || "mlp"}
        Epochs: #{get_in(model, [:metrics, :epochs_completed]) || "?"}
      """
    else
      line
    end
  end
end

defmodule Commands do
  alias ExPhil.Training.Output

  def list(args) do
    opts = parse_filter_opts(args)

    case Registry.list(opts) do
      {:ok, []} ->
        Output.puts("No models found.")

      {:ok, models} ->
        json? = "--json" in args

        if json? do
          IO.puts(Jason.encode!(models, pretty: true))
        else
          Output.section("Registered models (#{length(models)})")

          Enum.each(models, fn model ->
            Output.puts(RegistryUI.format_model(model))
          end)

          Output.puts("")
          Output.puts("Use 'mix run scripts/registry.exs show NAME' for details.")
        end

      {:error, reason} ->
        Output.error("Error: #{inspect(reason)}")
    end
  end

  def show([id | _]) do
    case Registry.get(id) do
      {:ok, model} ->
        Output.banner("Model: #{model.name}")

        Output.puts("""
        ID:           #{model.id}
        Created:      #{model.created_at}
        Parent:       #{model.parent_id || "none"}

        Files:
          Checkpoint:   #{model.checkpoint_path} #{if File.exists?(model.checkpoint_path), do: "✓", else: "✗ (missing)"}
          Policy:       #{model.policy_path || "none"} #{if model.policy_path && File.exists?(model.policy_path), do: "✓", else: ""}
          Config:       #{model.config_path || "none"} #{if model.config_path && File.exists?(model.config_path), do: "✓", else: ""}

        Tags:         #{if model.tags != [], do: Enum.join(model.tags, ", "), else: "none"}

        Training Config:
          Backbone:     #{get_in(model, [:training_config, :backbone]) || "mlp"}
          Temporal:     #{get_in(model, [:training_config, :temporal]) || false}
          Epochs:       #{get_in(model, [:training_config, :epochs]) || "?"}
          Batch Size:   #{get_in(model, [:training_config, :batch_size]) || "?"}
          Window Size:  #{get_in(model, [:training_config, :window_size]) || "N/A"}

        Metrics:
          Final Loss:   #{get_in(model, [:metrics, :final_loss]) || "?"}
          Epochs Done:  #{get_in(model, [:metrics, :epochs_completed]) || "?"}
          Train Frames: #{get_in(model, [:metrics, :training_frames]) || "?"}
          Val Frames:   #{get_in(model, [:metrics, :validation_frames]) || "?"}
          Train Time:   #{format_time(get_in(model, [:metrics, :total_time_seconds]))}
        """)

      {:error, :not_found} ->
        Output.error("Model '#{id}' not found.")
    end
  end

  def show([]) do
    Output.puts("Usage: mix run scripts/registry.exs show ID")
  end

  def best(args) do
    opts = parse_filter_opts(args)

    case Registry.best(opts) do
      {:ok, model} ->
        Output.success(
          "Best model: #{model.name} (loss=#{get_in(model, [:metrics, :final_loss])})"
        )

        Output.puts("  Path: #{model.checkpoint_path}")

      {:error, :no_models_with_metric} ->
        Output.warning("No models with loss metric found.")

      {:error, reason} ->
        Output.error("Error: #{inspect(reason)}")
    end
  end

  def tag([id | tags]) when tags != [] do
    case Registry.tag(id, tags) do
      :ok ->
        Output.success("Added tags #{inspect(tags)} to '#{id}'")

      {:error, :not_found} ->
        Output.error("Model '#{id}' not found.")

      {:error, reason} ->
        Output.error("Error: #{inspect(reason)}")
    end
  end

  def tag(_) do
    Output.puts("Usage: mix run scripts/registry.exs tag ID TAG [TAG...]")
  end

  def untag([id | tags]) when tags != [] do
    case Registry.untag(id, tags) do
      :ok ->
        Output.success("Removed tags #{inspect(tags)} from '#{id}'")

      {:error, :not_found} ->
        Output.error("Model '#{id}' not found.")

      {:error, reason} ->
        Output.error("Error: #{inspect(reason)}")
    end
  end

  def untag(_) do
    Output.puts("Usage: mix run scripts/registry.exs untag ID TAG [TAG...]")
  end

  def delete(args) do
    delete_files = "--files" in args
    args = args -- ["--files"]

    case args do
      [id | _] ->
        case Registry.delete(id, delete_files: delete_files) do
          :ok ->
            extra = if delete_files, do: " (and deleted files)", else: ""
            Output.success("Removed '#{id}' from registry#{extra}")

          {:error, :not_found} ->
            Output.error("Model '#{id}' not found.")

          {:error, reason} ->
            Output.error("Error: #{inspect(reason)}")
        end

      [] ->
        Output.puts("Usage: mix run scripts/registry.exs delete ID [--files]")
    end
  end

  def lineage([id | _]) do
    case Registry.lineage(id) do
      {:ok, models} ->
        Output.section("Lineage for '#{id}'")

        Enum.with_index(models)
        |> Enum.each(fn {model, idx} ->
          prefix = String.duplicate("  ", idx)
          Output.puts("#{prefix}└─ #{model.name} (#{model.id})")
        end)

      {:error, :not_found} ->
        Output.error("Model '#{id}' not found.")

      {:error, reason} ->
        Output.error("Error: #{inspect(reason)}")
    end
  end

  def lineage([]) do
    Output.puts("Usage: mix run scripts/registry.exs lineage ID")
  end

  # Helpers

  defp parse_filter_opts(args) do
    opts = []

    # Parse --tags
    opts =
      case find_arg_value(args, "--tags") do
        nil -> opts
        tags_str -> Keyword.put(opts, :tags, String.split(tags_str, ","))
      end

    # Parse --backbone
    opts =
      case find_arg_value(args, "--backbone") do
        nil -> opts
        backbone -> Keyword.put(opts, :backbone, String.to_atom(backbone))
      end

    # Parse --limit
    opts =
      case find_arg_value(args, "--limit") do
        nil -> opts
        limit_str -> Keyword.put(opts, :limit, String.to_integer(limit_str))
      end

    opts
  end

  defp find_arg_value(args, flag) do
    case Enum.find_index(args, &(&1 == flag)) do
      nil -> nil
      idx -> Enum.at(args, idx + 1)
    end
  end

  defp format_time(nil), do: "?"
  defp format_time(seconds) when seconds < 60, do: "#{seconds}s"

  defp format_time(seconds) do
    min = div(seconds, 60)
    sec = rem(seconds, 60)
    "#{min}m #{sec}s"
  end
end

# Main dispatch
case args do
  ["list" | rest] ->
    Commands.list(rest)

  ["show" | rest] ->
    Commands.show(rest)

  ["best" | rest] ->
    Commands.best(rest)

  ["tag" | rest] ->
    Commands.tag(rest)

  ["untag" | rest] ->
    Commands.untag(rest)

  ["delete" | rest] ->
    Commands.delete(rest)

  ["lineage" | rest] ->
    Commands.lineage(rest)

  [] ->
    Commands.list([])

  [cmd | _] ->
    Output.error("Unknown command: #{cmd}")

    Output.puts("""

    Usage: mix run scripts/registry.exs [command] [options]

    Commands:
      list                    - List all registered models
      show ID                 - Show details for a model
      best                    - Show the best model by loss
      tag ID TAG [TAG...]     - Add tags to a model
      untag ID TAG [TAG...]   - Remove tags from a model
      delete ID [--files]     - Remove a model from registry
      lineage ID              - Show model lineage tree

    Options:
      --tags TAG,TAG          - Filter by tags
      --backbone TYPE         - Filter by backbone
      --limit N               - Limit results
      --json                  - Output as JSON
    """)
end
