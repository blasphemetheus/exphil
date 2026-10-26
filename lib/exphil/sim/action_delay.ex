defmodule ExPhil.Sim.ActionDelay do
  @moduledoc """
  Explicit per-row controller FIFO for synchronous simulator experiments.

  A decision from observation t with delay d reaches Env.step at t+d and
  therefore controls state t+d+1. Reset rows discard only their pending inputs.
  This models fixed actuation latency, not network jitter or observation lag.
  """

  def new(delay, rows, neutral)
      when is_integer(delay) and delay >= 0 and is_integer(rows) and rows > 0 do
    %{pending: List.duplicate(List.duplicate(neutral, rows), delay), rows: rows, neutral: neutral}
  end

  def push(%{rows: rows} = queue, controllers) when length(controllers) == rows do
    case queue.pending do
      [] -> {controllers, queue}
      [ready | rest] -> {ready, %{queue | pending: rest ++ [controllers]}}
    end
  end

  def reset_rows(queue, rows) do
    pending =
      Enum.map(queue.pending, fn controllers ->
        controllers
        |> Enum.with_index()
        |> Enum.map(fn {controller, row} ->
          if row in rows, do: queue.neutral, else: controller
        end)
      end)

    %{queue | pending: pending}
  end
end
