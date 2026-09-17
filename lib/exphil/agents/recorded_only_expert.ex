defmodule ExPhil.Agents.RecordedOnlyExpert do
  @moduledoc "Training marker for validated causal recordings; never synthesizes labels."
  def from_frames([], _opts), do: :recorded_only

  def from_frames(_, _),
    do: raise(ArgumentError, "Recorded-only training does not accept fixtures")

  def label(_, _, _, _),
    do: raise(ArgumentError, "Recorded-only training cannot relabel rollouts")
end
