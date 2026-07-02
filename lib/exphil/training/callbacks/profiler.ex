defmodule ExPhil.Training.Callbacks.Profiler do
  @moduledoc "Track timing breakdown across training phases."

  use ExPhil.Training.Callback

  alias ExPhil.Training.Output

  @impl true
  def init(_opts) do
    %{
      epoch_timings: [],
      train_start: nil
    }
  end

  @impl true
  def on_train_begin(state, cb) do
    {:cont, state, %{cb | train_start: System.monotonic_time(:millisecond)}}
  end

  @impl true
  def on_train_end(state, cb) do
    total_ms = System.monotonic_time(:millisecond) - cb.train_start

    if cb.epoch_timings != [] do
      Output.puts("\n  Timing breakdown:")
      Output.puts("    Total: #{div(total_ms, 1000)}s")
      Output.puts("    Per-epoch average:")

      avg = fn key ->
        vals = Enum.map(cb.epoch_timings, &Map.get(&1, key, 0))
        if vals != [], do: div(Enum.sum(vals), length(vals)), else: 0
      end

      Output.puts("      Training: #{avg.(:train_ms)}ms")
      Output.puts("      Validation: #{avg.(:val_ms)}ms")
      Output.puts("      Diagnostics: #{avg.(:diag_ms)}ms")
    end

    {:cont, state, cb}
  end
end
