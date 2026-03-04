#!/usr/bin/env elixir
# Benchmark: Futhark parallel prefix scan vs CUDA C sequential scan
#
# Tests at multiple sequence lengths to find the crossover point where
# Futhark's O(log T) parallel scan beats CUDA C's O(T) sequential scan.
#
# Usage:
#   mix run scripts/benchmark_futhark_scan.exs
#   mix run scripts/benchmark_futhark_scan.exs --batch 32 --hidden 512
#   mix run scripts/benchmark_futhark_scan.exs --seq-lengths 30,60,120,240,480,960

alias ExPhil.Training.Output

require Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      iterations: :integer,
      warmup: :integer,
      batch: :integer,
      hidden: :integer,
      seq_lengths: :string,
      help: :boolean
    ]
  )

if opts[:help] do
  IO.puts("""
  Benchmark Futhark Parallel Prefix Scan

  Options:
    --iterations N       Timed iterations (default: 30)
    --warmup N           Warmup iterations (default: 5)
    --batch N            Batch size (default: 4)
    --hidden N           Hidden dimension (default: 64)
    --seq-lengths L      Comma-separated sequence lengths (default: 30,60,120,240,480,960)
    --help               Show this help
  """)

  System.halt(0)
end

iterations = opts[:iterations] || 30
warmup = opts[:warmup] || 5
batch = opts[:batch] || 4
hidden = opts[:hidden] || 64

seq_lengths =
  case opts[:seq_lengths] do
    nil -> [30, 60, 120, 240, 480, 960]
    str -> str |> String.split(",") |> Enum.map(&String.to_integer/1)
  end

Output.banner("Futhark Parallel Prefix Scan Benchmark")

Output.config([
  {"Batch", batch},
  {"Hidden dim", hidden},
  {"Sequence lengths", Enum.join(seq_lengths, ", ")},
  {"Warmup", warmup},
  {"Iterations", iterations}
])

IO.puts("")

# ============================================================================
# Check availability
# ============================================================================

futhark_available = ExPhil.Native.FutharkScan.available?()

if futhark_available do
  Output.success("Futhark NIF loaded")
else
  Output.warning("Futhark NIF not available. Build with: cd native/futhark_scan && make && make install")
  Output.puts("  Will only run CUDA C reference and Nx fallback")
end

xla_available = Code.ensure_loaded?(Edifice.CUDA.FusedScan)

IO.puts("")

# ============================================================================
# Benchmark harness
# ============================================================================

defmodule FutharkBench do
  def bench(fun, warmup_iters, timed_iters) do
    for _ <- 1..warmup_iters do
      fun.() |> Nx.backend_transfer(Nx.BinaryBackend)
    end

    times =
      for _ <- 1..timed_iters do
        t0 = System.monotonic_time(:microsecond)
        fun.() |> Nx.backend_transfer(Nx.BinaryBackend)
        System.monotonic_time(:microsecond) - t0
      end

    sorted = Enum.sort(times)
    median = Enum.at(sorted, div(length(sorted), 2))
    {median, Enum.min(times), Enum.max(times)}
  end

  def try_bench(fun, warmup_iters, timed_iters) do
    try do
      {:ok, bench(fun, warmup_iters, timed_iters)}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end
end

Nx.default_backend(EXLA.Backend)

# ============================================================================
# Run benchmarks at each sequence length
# ============================================================================

Output.puts("seq_len | Futhark (μs) | CUDA C (μs) | Nx (μs)  | Futhark/CUDA C | Winner")
Output.puts(String.duplicate("-", 80))

results =
  Enum.map(seq_lengths, fn seq_len ->
    key = Nx.Random.key(42)
    {a_vals, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {batch, seq_len, hidden}, type: :f32)
    {b_vals, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
    {h0, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {batch, hidden}, type: :f32)

    a_gpu = Nx.backend_transfer(a_vals, EXLA.Backend)
    b_gpu = Nx.backend_transfer(b_vals, EXLA.Backend)

    # Correctness check (Futhark vs Nx reference)
    if futhark_available do
      futhark_result = ExPhil.Native.FutharkScan.linear_scan(a_vals, b_vals, h0)

      # Nx reference
      {_, nx_states} =
        Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_state, acc} ->
          a_t = a_vals[[.., t, ..]]
          b_t = b_vals[[.., t, ..]]
          h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
          {h_new, [h_new | acc]}
        end)

      nx_ref = nx_states |> Enum.reverse() |> Nx.stack(axis: 1)

      diff = Nx.subtract(nx_ref, futhark_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      if diff > 1.0e-3 do
        Output.warning("seq_len=#{seq_len}: Futhark max diff #{Float.round(diff, 5)} exceeds 1e-3")
      end
    end

    # Benchmark Futhark
    futhark_med =
      if futhark_available do
        case FutharkBench.try_bench(fn -> ExPhil.Native.FutharkScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      else
        nil
      end

    # Benchmark CUDA C
    cuda_c_med =
      if xla_available do
        case FutharkBench.try_bench(fn -> Edifice.CUDA.FusedScan.linear_scan(a_gpu, b_gpu) end, warmup, iterations) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      else
        nil
      end

    # Benchmark Nx fallback
    nx_fn = fn ->
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_state, acc} ->
        a_t = a_vals[[.., t, ..]]
        b_t = b_vals[[.., t, ..]]
        h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
        {h_new, [h_new | acc]}
      end)
      |> then(fn {_, states} -> states |> Enum.reverse() |> Nx.stack(axis: 1) end)
    end

    {nx_med, _, _} = FutharkBench.bench(nx_fn, warmup, iterations)

    # Format output
    seq_str = String.pad_leading("#{seq_len}", 7)
    fut_str = if futhark_med, do: String.pad_leading("#{round(futhark_med)}", 12), else: "         N/A"
    cuda_str = if cuda_c_med, do: String.pad_leading("#{round(cuda_c_med)}", 11), else: "        N/A"
    nx_str = String.pad_leading("#{round(nx_med)}", 8)

    ratio_str =
      if futhark_med && cuda_c_med do
        ratio = futhark_med / max(cuda_c_med, 1)
        String.pad_leading(:erlang.float_to_binary(ratio, decimals: 2) <> "x", 14)
      else
        "           N/A"
      end

    winner =
      cond do
        futhark_med && cuda_c_med && futhark_med < cuda_c_med -> "Futhark"
        futhark_med && cuda_c_med -> "CUDA C"
        true -> "N/A"
      end

    IO.puts("#{seq_str} | #{fut_str} | #{cuda_str} | #{nx_str} | #{ratio_str} | #{winner}")

    %{seq_len: seq_len, futhark: futhark_med, cuda_c: cuda_c_med, nx: nx_med}
  end)

# ============================================================================
# Summary
# ============================================================================

IO.puts("")
IO.puts(String.duplicate("=", 60))
IO.puts("Summary")
IO.puts(String.duplicate("=", 60))

# Find crossover point
crossover =
  Enum.find(results, fn r ->
    r.futhark != nil && r.cuda_c != nil && r.futhark < r.cuda_c
  end)

if crossover do
  Output.success("Futhark parallel scan beats CUDA C at seq_len >= #{crossover.seq_len}")
else
  if futhark_available do
    Output.puts("Futhark did not beat CUDA C at any tested sequence length")
  end
end

IO.puts("")
Output.puts("Key insight: Futhark's parallel prefix scan uses O(log T) parallel steps")
Output.puts("but 2x the total work. It wins when T is large enough for parallelism")
Output.puts("to overcome the extra work — typically T > 256 for small hidden dims.")

IO.puts("")
Output.success("Benchmark complete")
