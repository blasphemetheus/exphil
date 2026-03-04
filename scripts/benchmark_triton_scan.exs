#!/usr/bin/env elixir
# Benchmark: Triton AOT linear scan vs CUDA C vs Rust NIF vs Nx
#
# Tests the Triton AOT-compiled kernel (zero Python at runtime) at
# multiple sequence lengths against all other backends.
#
# Usage:
#   mix run scripts/benchmark_triton_scan.exs
#   mix run scripts/benchmark_triton_scan.exs --batch 32 --hidden 512
#   mix run scripts/benchmark_triton_scan.exs --seq-lengths 30,60,120,240,480,960

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
  Benchmark Triton AOT Linear Scan (cubin via C NIF)

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

Output.banner("Triton AOT Linear Scan Benchmark (cubin → C NIF)")

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

triton_available = ExPhil.Native.TritonScan.available?()

if triton_available do
  Output.success("Triton NIF loaded (AOT cubin)")
else
  Output.warning("Triton NIF not available. Build with: cd native/triton_scan && make && make install")
  Output.puts("  Will only run CUDA C reference, Rust NIF, and Nx fallback")
end

rust_available = ExPhil.Native.RustLinearScan.available?()
if rust_available, do: Output.success("Rust-CUDA NIF loaded"), else: Output.puts("  Rust NIF: not available")

xla_available = Code.ensure_loaded?(Edifice.CUDA.FusedScan)
if xla_available, do: Output.success("CUDA C (FusedScan) loaded"), else: Output.puts("  CUDA C: not available")

futhark_available = ExPhil.Native.FutharkScan.available?()
if futhark_available, do: Output.success("Futhark NIF loaded"), else: Output.puts("  Futhark: not available")

IO.puts("")

# ============================================================================
# Benchmark harness
# ============================================================================

defmodule TritonBench do
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
    catch
      kind, reason -> {:error, "#{kind}: #{inspect(reason)}"}
    end
  end
end

Nx.default_backend(EXLA.Backend)

# ============================================================================
# Correctness check
# ============================================================================

if triton_available do
  Output.puts("Correctness check...")

  key = Nx.Random.key(42)
  {a_test, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {2, 10, 8}, type: :f32)
  {b_test, key} = Nx.Random.normal(key, shape: {2, 10, 8}, type: :f32)
  {h0_test, _} = Nx.Random.normal(key, 0.0, 0.1, shape: {2, 8}, type: :f32)

  triton_result = ExPhil.Native.TritonScan.linear_scan(a_test, b_test, h0_test)

  {_, nx_states} =
    Enum.reduce(0..9, {h0_test, []}, fn t, {h_state, acc} ->
      a_t = a_test[[.., t, ..]]
      b_t = b_test[[.., t, ..]]
      h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
      {h_new, [h_new | acc]}
    end)

  nx_ref = nx_states |> Enum.reverse() |> Nx.stack(axis: 1)

  diff = Nx.subtract(nx_ref, triton_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

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

Output.puts("seq_len | Triton (μs) | Rust NIF (μs) | CUDA C (μs) | Futhark (μs) | Nx (μs)  | Triton/CUDA C")
Output.puts(String.duplicate("-", 100))

results =
  Enum.map(seq_lengths, fn seq_len ->
    key = Nx.Random.key(42)
    {a_vals, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {batch, seq_len, hidden}, type: :f32)
    {b_vals, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
    {h0, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {batch, hidden}, type: :f32)

    a_gpu = Nx.backend_transfer(a_vals, EXLA.Backend)
    b_gpu = Nx.backend_transfer(b_vals, EXLA.Backend)

    # Benchmark Triton
    triton_med =
      if triton_available do
        case TritonBench.try_bench(fn -> ExPhil.Native.TritonScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      else
        nil
      end

    # Benchmark Rust NIF
    rust_med =
      if rust_available do
        case TritonBench.try_bench(fn -> ExPhil.Native.RustLinearScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      else
        nil
      end

    # Benchmark CUDA C
    cuda_c_med =
      if xla_available do
        case TritonBench.try_bench(fn -> Edifice.CUDA.FusedScan.linear_scan(a_gpu, b_gpu) end, warmup, iterations) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      else
        nil
      end

    # Benchmark Futhark
    futhark_med =
      if futhark_available do
        case TritonBench.try_bench(fn -> ExPhil.Native.FutharkScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
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

    {nx_med, _, _} = TritonBench.bench(nx_fn, warmup, iterations)

    # Format output
    seq_str = String.pad_leading("#{seq_len}", 7)
    triton_str = if triton_med, do: String.pad_leading("#{round(triton_med)}", 11), else: "        N/A"
    rust_str = if rust_med, do: String.pad_leading("#{round(rust_med)}", 13), else: "          N/A"
    cuda_str = if cuda_c_med, do: String.pad_leading("#{round(cuda_c_med)}", 11), else: "        N/A"
    futhark_str = if futhark_med, do: String.pad_leading("#{round(futhark_med)}", 12), else: "         N/A"
    nx_str = String.pad_leading("#{round(nx_med)}", 8)

    ratio_str =
      if triton_med && cuda_c_med do
        ratio = triton_med / max(cuda_c_med, 1)
        String.pad_leading(:erlang.float_to_binary(ratio, decimals: 2) <> "x", 13)
      else
        "          N/A"
      end

    IO.puts("#{seq_str} | #{triton_str} | #{rust_str} | #{cuda_str} | #{futhark_str} | #{nx_str} | #{ratio_str}")

    %{seq_len: seq_len, triton: triton_med, rust: rust_med, cuda_c: cuda_c_med, futhark: futhark_med, nx: nx_med}
  end)

# ============================================================================
# Summary
# ============================================================================

IO.puts("")
IO.puts(String.duplicate("=", 60))
IO.puts("Summary")
IO.puts(String.duplicate("=", 60))

triton_results = Enum.filter(results, & &1.triton)

if length(triton_results) > 0 do
  # vs CUDA C
  cuda_ratios =
    triton_results
    |> Enum.filter(& &1.cuda_c)
    |> Enum.map(fn r -> r.triton / max(r.cuda_c, 1) end)

  if length(cuda_ratios) > 0 do
    avg = Enum.sum(cuda_ratios) / length(cuda_ratios)
    IO.puts("")
    Output.puts("Average Triton/CUDA C ratio: #{:erlang.float_to_binary(avg, decimals: 2)}x")

    cond do
      avg < 1.1 ->
        Output.success("Triton AOT matches CUDA C performance (< 10% overhead)")
      avg < 2.0 ->
        Output.puts("Triton AOT has moderate overhead vs CUDA C (#{round((avg - 1) * 100)}%)")
      true ->
        Output.puts("Triton AOT has significant overhead vs CUDA C (#{:erlang.float_to_binary(avg, decimals: 1)}x)")
    end
  end

  # vs Rust
  rust_ratios =
    triton_results
    |> Enum.filter(& &1.rust)
    |> Enum.map(fn r -> r.triton / max(r.rust, 1) end)

  if length(rust_ratios) > 0 do
    avg = Enum.sum(rust_ratios) / length(rust_ratios)
    Output.puts("Average Triton/Rust ratio: #{:erlang.float_to_binary(avg, decimals: 2)}x")
  end
end

IO.puts("")
Output.puts("Key insight: Triton kernel is written in Python, AOT-compiled to cubin,")
Output.puts("then loaded from C NIF via CUDA driver API. Zero Python at runtime.")
Output.puts("Overhead vs CUDA C comes from NIF data transfer (Nx → binary → GPU → binary → Nx)")
Output.puts("vs XLA's in-graph execution where tensors stay on GPU.")

IO.puts("")
Output.success("Benchmark complete")
