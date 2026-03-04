#!/usr/bin/env elixir
# Benchmark: Rust-CUDA (Rustler NIF) linear scan vs CUDA C vs Nx
#
# Tests the Rust-CUDA integration at multiple sequence lengths.
# The kernel implements h[t] = a[t] * h[t-1] + b[t].
#
# Usage:
#   mix run scripts/benchmark_rust_scan.exs
#   mix run scripts/benchmark_rust_scan.exs --batch 32 --hidden 512
#   mix run scripts/benchmark_rust_scan.exs --seq-lengths 30,60,120,240,480,960

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
  Benchmark Rust-CUDA Linear Scan (Rustler NIF)

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

Output.banner("Rust-CUDA Linear Scan Benchmark (Rustler NIF)")

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

rust_available = ExPhil.Native.RustLinearScan.available?()

if rust_available do
  Output.success("Rust-CUDA NIF loaded")

  if ExPhil.Native.RustLinearScan.cuda_available?() do
    Output.success("CUDA device available")
  else
    Output.warning("No CUDA device — will use CPU fallback inside NIF")
  end
else
  Output.warning("Rust-CUDA NIF not available. Build with:")
  Output.puts("  cd native/rust_linear_scan_nif")
  Output.puts("  cargo build --release --features cuda")
  Output.puts("  cp target/release/librust_linear_scan_nif.so ../../priv/native/")
  Output.puts("")
  Output.puts("  Will only run CUDA C reference and Nx fallback")
end

xla_available = Code.ensure_loaded?(Edifice.CUDA.FusedScan)

IO.puts("")

# ============================================================================
# Benchmark harness
# ============================================================================

defmodule RustBench do
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
# Correctness check
# ============================================================================

if rust_available do
  Output.puts("Correctness check...")

  key = Nx.Random.key(42)
  {a_test, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {2, 10, 8}, type: :f32)
  {b_test, key} = Nx.Random.normal(key, shape: {2, 10, 8}, type: :f32)
  {h0_test, _} = Nx.Random.normal(key, 0.0, 0.1, shape: {2, 8}, type: :f32)

  rust_result = ExPhil.Native.RustLinearScan.linear_scan(a_test, b_test, h0_test)

  # Nx reference
  {_, nx_states} =
    Enum.reduce(0..9, {h0_test, []}, fn t, {h_state, acc} ->
      a_t = a_test[[.., t, ..]]
      b_t = b_test[[.., t, ..]]
      h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
      {h_new, [h_new | acc]}
    end)

  nx_ref = nx_states |> Enum.reverse() |> Nx.stack(axis: 1)

  diff = Nx.subtract(nx_ref, rust_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

  if diff < 1.0e-4 do
    Output.success("Correctness verified (max diff: #{Float.round(diff, 7)})")
  else
    Output.warning("Max diff #{Float.round(diff, 5)} exceeds 1e-4 — check implementation")
  end

  IO.puts("")
end

# ============================================================================
# Run benchmarks at each sequence length
# ============================================================================

Output.puts("seq_len | Rust NIF (μs) | CUDA C (μs) | Nx (μs)  | Rust/CUDA C | Winner")
Output.puts(String.duplicate("-", 80))

results =
  Enum.map(seq_lengths, fn seq_len ->
    key = Nx.Random.key(42)
    {a_vals, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {batch, seq_len, hidden}, type: :f32)
    {b_vals, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
    {h0, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {batch, hidden}, type: :f32)

    a_gpu = Nx.backend_transfer(a_vals, EXLA.Backend)
    b_gpu = Nx.backend_transfer(b_vals, EXLA.Backend)

    # Benchmark Rust NIF
    rust_med =
      if rust_available do
        case RustBench.try_bench(fn -> ExPhil.Native.RustLinearScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      else
        nil
      end

    # Benchmark CUDA C
    cuda_c_med =
      if xla_available do
        case RustBench.try_bench(fn -> Edifice.CUDA.FusedScan.linear_scan(a_gpu, b_gpu) end, warmup, iterations) do
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

    {nx_med, _, _} = RustBench.bench(nx_fn, warmup, iterations)

    # Format output
    seq_str = String.pad_leading("#{seq_len}", 7)
    rust_str = if rust_med, do: String.pad_leading("#{round(rust_med)}", 13), else: "          N/A"
    cuda_str = if cuda_c_med, do: String.pad_leading("#{round(cuda_c_med)}", 11), else: "        N/A"
    nx_str = String.pad_leading("#{round(nx_med)}", 8)

    ratio_str =
      if rust_med && cuda_c_med do
        ratio = rust_med / max(cuda_c_med, 1)
        String.pad_leading(:erlang.float_to_binary(ratio, decimals: 2) <> "x", 11)
      else
        "        N/A"
      end

    winner =
      cond do
        rust_med && cuda_c_med && rust_med < cuda_c_med -> "Rust NIF"
        rust_med && cuda_c_med -> "CUDA C"
        true -> "N/A"
      end

    IO.puts("#{seq_str} | #{rust_str} | #{cuda_str} | #{nx_str} | #{ratio_str} | #{winner}")

    %{seq_len: seq_len, rust: rust_med, cuda_c: cuda_c_med, nx: nx_med}
  end)

# ============================================================================
# Summary
# ============================================================================

IO.puts("")
IO.puts(String.duplicate("=", 60))
IO.puts("Summary")
IO.puts(String.duplicate("=", 60))

# Analysis
rust_results = Enum.filter(results, & &1.rust)

if length(rust_results) > 0 do
  avg_ratio =
    rust_results
    |> Enum.filter(& &1.cuda_c)
    |> Enum.map(fn r -> r.rust / max(r.cuda_c, 1) end)
    |> then(fn ratios ->
      if length(ratios) > 0 do
        Enum.sum(ratios) / length(ratios)
      else
        nil
      end
    end)

  if avg_ratio do
    IO.puts("")
    Output.puts("Average Rust/CUDA C ratio: #{:erlang.float_to_binary(avg_ratio, decimals: 2)}x")

    cond do
      avg_ratio < 1.1 ->
        Output.success("Rust NIF is competitive with CUDA C (< 10% overhead)")
      avg_ratio < 2.0 ->
        Output.puts("Rust NIF has moderate overhead vs CUDA C (#{round((avg_ratio - 1) * 100)}%)")
      true ->
        Output.puts("Rust NIF has significant overhead vs CUDA C (#{:erlang.float_to_binary(avg_ratio, decimals: 1)}x)")
    end
  end
end

IO.puts("")
Output.puts("Key insight: Rust-CUDA via Rustler provides the most native Elixir")
Output.puts("integration (NIF = zero serialization), with Rust type safety from")
Output.puts("Elixir to GPU. Overhead comes from data copy (Nx → binary → GPU → binary → Nx).")

IO.puts("")
Output.success("Benchmark complete")
