defmodule ExPhil.Training.Config.Diff do
  @moduledoc """
  Configuration diff display for training.

  Provides utilities for comparing training options against defaults
  and formatting the differences for display. Useful for verifying
  configuration at training start.

  ## Usage

      opts = [epochs: 20, batch_size: 128]

      # Get structured diff
      diff = Diff.from_defaults(opts, defaults_fn)
      # => [{:epochs, 20, 10}, {:batch_size, 128, 64}]

      # Format for display
      Diff.format(opts, defaults_fn)
      # =>   epochs: 20 (default: 10)
      #      batch_size: 128 (default: 64)

  ## See Also

  - `ExPhil.Training.Config` - Main configuration module
  - `ExPhil.Training.Output` - Training output formatting
  """

  @doc """
  Get a list of options that differ from defaults.

  Useful for displaying at training start to verify configuration.
  Returns a list of `{key, current_value, default_value}` tuples.

  ## Parameters

  - `opts` - Current configuration options
  - `defaults_fn` - Function that returns default options (0-arity)
  - `diff_opts` - Options for diffing

  ## Options

  - `:skip` - List of keys to skip (default: [:replays, :checkpoint, :name, :wandb_name, :wandb_project])
  - `:include_nil` - Include keys where current is nil but default is not (default: false)

  ## Examples

      iex> defaults_fn = fn -> [epochs: 10, batch_size: 64] end
      iex> Diff.from_defaults([epochs: 20, batch_size: 64], defaults_fn)
      [{:epochs, 20, 10}]

  """
  @spec from_defaults(keyword(), (-> keyword()), keyword()) :: [{atom(), any(), any()}]
  def from_defaults(opts, defaults_fn, diff_opts \\ []) do
    defaults = defaults_fn.()

    skip_keys =
      Keyword.get(diff_opts, :skip, [:replays, :checkpoint, :name, :wandb_name, :wandb_project])

    include_nil = Keyword.get(diff_opts, :include_nil, false)

    opts
    |> Enum.filter(fn {key, value} ->
      key not in skip_keys and
        Keyword.has_key?(defaults, key) and
        value != Keyword.get(defaults, key) and
        (include_nil or value != nil)
    end)
    |> Enum.map(fn {key, value} ->
      {key, value, Keyword.get(defaults, key)}
    end)
    |> Enum.sort_by(fn {key, _, _} -> key end)
  end

  @doc """
  Format config diff as a human-readable string.

  Returns a string showing changed settings, or nil if no changes.
  Output is sorted alphabetically by key.

  ## Parameters

  - `opts` - Current configuration options
  - `defaults_fn` - Function that returns default options (0-arity)

  ## Examples

      iex> defaults_fn = fn -> [epochs: 10, batch_size: 64] end
      iex> result = Diff.format([epochs: 20, batch_size: 128], defaults_fn)
      iex> result =~ "epochs: 20 (default: 10)"
      true

  """
  @spec format(keyword(), (-> keyword())) :: String.t() | nil
  def format(opts, defaults_fn) do
    diff = from_defaults(opts, defaults_fn)

    if Enum.empty?(diff) do
      nil
    else
      diff
      |> Enum.map(fn {key, current, default} ->
        "  #{key}: #{format_value(current)} (default: #{format_value(default)})"
      end)
      |> Enum.join("\n")
    end
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp format_value(value) when is_list(value), do: inspect(value, charlists: :as_lists)
  defp format_value(value) when is_atom(value), do: ":#{value}"

  defp format_value(value) when is_float(value),
    do: :erlang.float_to_binary(value, [:compact, decimals: 6])

  defp format_value(nil), do: "nil"
  defp format_value(value), do: "#{value}"
end
