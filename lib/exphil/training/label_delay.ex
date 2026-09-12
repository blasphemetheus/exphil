defmodule ExPhil.Training.LabelDelay do
  @moduledoc """
  Resolves training delay aliases into one nonnegative reaction delay.

  `label_delay`, `frame_delay`, and `action_delay` are aliases in training
  configuration. Explicit values in one source must agree, including zero.
  Higher-precedence sources override all aliases of lower-precedence sources.
  Live Dolphin delay flags and legacy checkpoint numbering are unchanged.
  """

  @keys [:label_delay, :frame_delay, :action_delay]

  def keys, do: @keys

  def resolve!(opts) do
    delay = value!(opts) || 0
    Enum.reduce(@keys, opts, &Keyword.put(&2, &1, delay))
  end

  def merge_layers!(layers) do
    delay = Enum.reduce(layers, 0, fn layer, previous -> value!(layer) || previous end)
    resolve!(label_delay: delay)
  end

  def explicit?(opts), do: Enum.any?(@keys, &(Keyword.get(opts, &1) != nil))

  defp value!(opts) do
    values = Keyword.take(opts, @keys) |> Enum.reject(fn {_key, value} -> is_nil(value) end)

    Enum.each(values, fn {key, value} ->
      unless is_integer(value) and value >= 0,
        do:
          raise(
            ArgumentError,
            "#{key} must be a nonnegative integer reaction delay, got: #{inspect(value)}"
          )
    end)

    case values |> Keyword.values() |> Enum.uniq() do
      [] ->
        nil

      [delay] ->
        delay

      _ ->
        raise ArgumentError,
              "Conflicting training delay aliases: #{inspect(values)}; use one --label-delay"
    end
  end
end
