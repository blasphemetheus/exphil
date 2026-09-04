defmodule ExPhil.Training.BpttProf do
  @moduledoc """
  Wall-time stage profiler for the BPTT training loop, enabled with
  `EXPHIL_BPTT_PROFILE=1`. Near-zero cost when disabled.

  Accumulates per-stage totals in the calling process's dictionary —
  `TrajectoryCursors.batch_stream` batches are produced lazily inside
  the training process's reduce, so producer and consumer stages land in
  the same accumulator and one report covers the whole step.

  Built for the 2026-09-04 BPTT throughput hunt: the jitted
  fwd+bwd graph measured ~23ms while training ran at ~3s/it — the gap
  is host-side batch assembly, and this instrument names the stage.
  """

  alias ExPhil.Training.Output

  @key :__exphil_bptt_prof__

  def enabled?, do: System.get_env("EXPHIL_BPTT_PROFILE") == "1"

  @doc "Time `fun` under `key`, accumulating microseconds and call count."
  def time(key, fun) do
    t0 = System.monotonic_time(:microsecond)
    result = fun.()
    dt = System.monotonic_time(:microsecond) - t0

    acc = Process.get(@key, %{})
    {total, count} = Map.get(acc, key, {0, 0})
    Process.put(@key, Map.put(acc, key, {total + dt, count + 1}))

    result
  end

  @doc """
  Print accumulated stage totals (sorted by total time) and reset.
  Call every N steps from the training loop; no-op when empty.
  """
  def report(label) do
    acc = Process.get(@key, %{})

    if acc != %{} do
      total_us = acc |> Map.values() |> Enum.map(&elem(&1, 0)) |> Enum.sum()

      lines =
        acc
        |> Enum.sort_by(fn {_k, {us, _}} -> -us end)
        |> Enum.map(fn {k, {us, count}} ->
          ms = Float.round(us / 1000, 1)
          per = Float.round(us / count / 1000, 1)
          pct = Float.round(100 * us / max(total_us, 1), 1)
          "      #{k}: #{ms}ms total, #{per}ms/call x#{count} (#{pct}%)"
        end)

      Output.puts("  [bptt-prof #{label}] stage totals:")
      Enum.each(lines, &Output.puts/1)
      Process.put(@key, %{})
    end

    :ok
  end
end
