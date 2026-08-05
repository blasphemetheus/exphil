defmodule ExPhil.Bridge.ActionQueue do
  @moduledoc """
  Frame-keyed action scheduling (LATENCY_ARCHITECTURE direction #3).

  Explicit local delay: an action tagged for frame t+D is HELD here and
  only applied when the console reports frame t+D — so the state->action
  latency is exactly D frames regardless of how fast inference ran.

  Pure data structure (no controller access), a direct port of
  melee_bridge.py's `ActionQueue`, so the exact-D semantics stay pinned
  by tests without Dolphin.
  """

  @type t :: %__MODULE__{
          queue: %{integer() => [term()]},
          last_frame: integer() | nil
        }

  defstruct queue: %{}, last_frame: nil

  @doc "An empty queue."
  @spec new() :: t()
  def new, do: %__MODULE__{}

  @doc "Schedule `item` to fire when the frame clock reaches `frame`."
  @spec schedule(t(), integer(), term()) :: t()
  def schedule(%__MODULE__{} = q, frame, item) when is_integer(frame) do
    %{q | queue: Map.update(q.queue, frame, [item], &(&1 ++ [item]))}
  end

  @doc """
  Pop all items scheduled for frames <= `frame`, oldest frame first.

  A frame REGRESSION (new game: Melee restarts at -123) drops the whole
  queue — actions scheduled against the previous game's timeline must
  never fire into the new one.
  """
  @spec pop_due(t(), integer()) :: {[term()], t()}
  def pop_due(%__MODULE__{} = q, frame) when is_integer(frame) do
    q =
      if q.last_frame != nil and frame < q.last_frame do
        %{q | queue: %{}}
      else
        q
      end

    q = %{q | last_frame: frame}

    {due, remaining} = Map.split_with(q.queue, fn {f, _} -> f <= frame end)

    items =
      due
      |> Enum.sort_by(fn {f, _} -> f end)
      |> Enum.flat_map(fn {_f, list} -> list end)

    {items, %{q | queue: remaining}}
  end

  @doc "Total queued item count."
  @spec size(t()) :: non_neg_integer()
  def size(%__MODULE__{queue: queue}) do
    queue |> Map.values() |> Enum.map(&length/1) |> Enum.sum()
  end
end
