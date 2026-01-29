defmodule ExPhil.Training.Profiler do
  @moduledoc """
  Training profiler for identifying performance bottlenecks.

  Collects timing statistics for different phases of training:
  - Batch preparation (data loading, shuffling)
  - Forward pass
  - Backward pass (gradient computation)
  - Optimizer update
  - Validation
  - Checkpoint saving

  ## Usage

      # Start profiling
      Profiler.start()

      # Time operations
      Profiler.time(:batch_prep, fn -> prepare_batch() end)
      Profiler.time(:forward, fn -> forward_pass() end)

      # Get report
      Profiler.report()

      # Stop profiling
      Profiler.stop()

  ## Integration with training

  Enable with `--profile` flag:

      mix run scripts/train_from_replays.exs --profile --epochs 1

  """

  use Agent

  @type timing :: %{
          count: non_neg_integer(),
          total_ms: float(),
          min_ms: float(),
          max_ms: float(),
          samples: [float()]
        }

  @type state :: %{
          enabled: boolean(),
          timings: %{atom() => timing()},
          start_time: integer() | nil
        }

  @doc """
  Start the profiler agent.
  """
  def start do
    case Agent.start_link(fn -> initial_state() end, name: __MODULE__) do
      {:ok, pid} -> {:ok, pid}
      {:error, {:already_started, pid}} -> {:ok, pid}
    end
  end

  @doc """
  Stop the profiler agent.
  """
  def stop do
    if Process.whereis(__MODULE__) do
      Agent.stop(__MODULE__)
    end

    :ok
  end

  @doc """
  Enable or disable profiling.
  """
  def set_enabled(enabled) do
    ensure_started()
    Agent.update(__MODULE__, &%{&1 | enabled: enabled})
  end

  @doc """
  Check if profiling is enabled.
  """
  def enabled? do
    if Process.whereis(__MODULE__) do
      Agent.get(__MODULE__, & &1.enabled)
    else
      false
    end
  end

  @doc """
  Time a function and record the duration under the given key.

  If profiling is disabled, just executes the function without timing.
  """
  def time(key, fun) when is_atom(key) and is_function(fun, 0) do
    if enabled?() do
      start = System.monotonic_time(:microsecond)
      result = fun.()
      elapsed_us = System.monotonic_time(:microsecond) - start
      elapsed_ms = elapsed_us / 1000

      record(key, elapsed_ms)
      result
    else
      fun.()
    end
  end

  @doc """
  Record a timing measurement directly (in milliseconds).
  """
  def record(key, elapsed_ms) when is_atom(key) and is_number(elapsed_ms) do
    if enabled?() do
      Agent.update(__MODULE__, fn state ->
        timing = Map.get(state.timings, key, empty_timing())

        updated_timing = %{
          timing
          | count: timing.count + 1,
            total_ms: timing.total_ms + elapsed_ms,
            min_ms: min(timing.min_ms, elapsed_ms),
            max_ms: max(timing.max_ms, elapsed_ms),
            # Keep last 1000 samples for percentile calculation
            samples: Enum.take([elapsed_ms | timing.samples], 1000)
        }

        %{state | timings: Map.put(state.timings, key, updated_timing)}
      end)
    end

    :ok
  end

  @doc """
  Get current timing statistics.
  """
  def get_stats do
    ensure_started()
    Agent.get(__MODULE__, & &1.timings)
  end

  @doc """
  Reset all timing statistics.
  """
  def reset do
    ensure_started()
    Agent.update(__MODULE__, fn state -> %{state | timings: %{}} end)
  end

  @doc """
  Generate a formatted profiling report.
  """
  def report do
    stats = get_stats()

    if map_size(stats) == 0 do
      "No profiling data collected."
    else
      format_report(stats)
    end
  end

  @doc """
  Print the profiling report to stderr.
  """
  def print_report do
    IO.puts(:stderr, "\n" <> report())
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp initial_state do
    %{
      enabled: false,
      timings: %{},
      start_time: System.monotonic_time(:millisecond)
    }
  end

  defp empty_timing do
    %{
      count: 0,
      total_ms: 0.0,
      min_ms: :infinity,
      max_ms: 0.0,
      samples: []
    }
  end

  defp ensure_started do
    unless Process.whereis(__MODULE__) do
      start()
    end
  end

  defp format_report(stats) do
    # Sort by total time descending
    sorted =
      stats
      |> Enum.sort_by(fn {_key, timing} -> -timing.total_ms end)

    total_tracked = Enum.sum(Enum.map(sorted, fn {_, t} -> t.total_ms end))

    header = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                           TRAINING PROFILE REPORT                            ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║ Phase              │ Count   │ Total (s) │ Avg (ms) │ Min (ms) │ Max (ms) │ % ║
    ╟────────────────────┼─────────┼───────────┼──────────┼──────────┼──────────┼───╢
    """

    rows =
      sorted
      |> Enum.map(fn {key, timing} ->
        avg_ms = if timing.count > 0, do: timing.total_ms / timing.count, else: 0.0
        pct = if total_tracked > 0, do: timing.total_ms / total_tracked * 100, else: 0.0
        min_ms = if timing.min_ms == :infinity, do: 0.0, else: timing.min_ms

        # Format key name (max 18 chars)
        key_str = key |> to_string() |> String.slice(0, 18) |> String.pad_trailing(18)

        "║ #{key_str} │ #{pad_num(timing.count, 7)} │ #{pad_float(timing.total_ms / 1000, 9, 2)} │ #{pad_float(avg_ms, 8, 2)} │ #{pad_float(min_ms, 8, 2)} │ #{pad_float(timing.max_ms, 8, 2)} │#{pad_float(pct, 3, 0)}║"
      end)
      |> Enum.join("\n")

    footer = """
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║ Total tracked time: #{pad_float(total_tracked / 1000, 10, 2)} seconds#{String.duplicate(" ", 38)}║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """

    header <> rows <> "\n" <> footer
  end

  defp pad_num(num, width) do
    num |> Integer.to_string() |> String.pad_leading(width)
  end

  defp pad_float(num, width, decimals) do
    # Use Elixir formatting to avoid Erlang io_lib edge cases with 0 decimals
    rounded = Float.round(num * 1.0, decimals)
    formatted = :erlang.float_to_binary(rounded, decimals: decimals)
    String.pad_leading(formatted, width)
  end

  @doc """
  Get a summary suitable for inline display during training.
  """
  def inline_summary do
    stats = get_stats()

    if map_size(stats) == 0 do
      ""
    else
      # Show top 3 by total time
      top3 =
        stats
        |> Enum.sort_by(fn {_key, t} -> -t.total_ms end)
        |> Enum.take(3)
        |> Enum.map(fn {key, t} ->
          avg = if t.count > 0, do: t.total_ms / t.count, else: 0.0
          "#{key}=#{Float.round(avg, 1)}ms"
        end)
        |> Enum.join(" ")

      "[Profile: #{top3}]"
    end
  end
end
