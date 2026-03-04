#!/usr/bin/env elixir
# Benchmark: Mojo linear scan vs CUDA C reference
#
# Tests Mojo's kernel (if available) and NumPy fallback against CUDA C.
# Mojo installation on NixOS/WSL2 is experimental — NumPy fallback
# provides a useful baseline even if Mojo fails to install.
#
# Usage:
#   mix run scripts/benchmark_mojo_scan.exs
#   mix run scripts/benchmark_mojo_scan.exs --batch 32 --seq 120 --hidden 512

alias ExPhil.Training.Output

require Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      iterations: :integer,
      warmup: :integer,
      batch: :integer,
      seq: :integer,
      hidden: :integer,
      help: :boolean
    ]
  )

if opts[:help] do
  IO.puts("""
  Benchmark Mojo Linear Scan

  Options:
    --iterations N   Timed iterations (default: 30)
    --warmup N       Warmup iterations (default: 5)
    --batch N        Batch size (default: 4)
    --seq N          Sequence length (default: 60)
    --hidden N       Hidden dimension (default: 64)
    --help           Show this help
  """)

  System.halt(0)
end

iterations = opts[:iterations] || 30
warmup = opts[:warmup] || 5
batch = opts[:batch] || 4
seq_len = opts[:seq] || 60
hidden = opts[:hidden] || 64

Output.banner("Mojo Linear Scan Benchmark")

Output.config([
  {"Batch", batch},
  {"Sequence length", seq_len},
  {"Hidden dim", hidden},
  {"Warmup", warmup},
  {"Iterations", iterations}
])

IO.puts("")

# ============================================================================
# Start Mojo/NumPy server
# ============================================================================

Output.step(1, 4, "Starting Mojo/NumPy server")

mojo_available =
  case ExPhil.Bridge.MojoPort.start_link() do
    {:ok, _pid} ->
      case ExPhil.Bridge.MojoPort.ping() do
        {:ok, info} ->
          device = info["device"] || "unknown"
          mojo_loaded = info["mojo_available"] || false

          if mojo_loaded do
            Output.success("Mojo kernel loaded (device: #{device})")
          else
            Output.warning("Mojo not available, using NumPy fallback (device: #{device})")
          end

          true

        {:error, reason} ->
          Output.error("Server ping failed: #{inspect(reason)}")
          false
      end

    {:error, reason} ->
      Output.error("Failed to start server: #{inspect(reason)}")
      false
  end

if not mojo_available do
  Output.error("Server not available. Install: pip install msgpack numpy")
  System.halt(1)
end

# ============================================================================
# Generate test data
# ============================================================================

Output.step(2, 4, "Generating test data")

key = Nx.Random.key(42)
{a_vals, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {batch, seq_len, hidden}, type: :f32)
{b_vals, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
{h0, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {batch, hidden}, type: :f32)

IO.puts("")

# ============================================================================
# Correctness check
# ============================================================================

Output.step(3, 4, "Checking correctness")

# Nx reference
{_, nx_states} =
  Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_state, acc} ->
    a_t = a_vals[[.., t, ..]]
    b_t = b_vals[[.., t, ..]]
    h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
    {h_new, [h_new | acc]}
  end)

reference = nx_states |> Enum.reverse() |> Nx.stack(axis: 1)

case ExPhil.Bridge.MojoPort.linear_scan(a_vals, b_vals, h0) do
  {:ok, mojo_result} ->
    diff = Nx.subtract(reference, mojo_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    if diff < 1.0e-4 do
      Output.success("Correct (max diff: #{Float.round(diff, 7)})")
    else
      Output.warning("Max diff #{Float.round(diff, 4)} — check precision")
    end

  {:error, reason} ->
    Output.error("Scan failed: #{inspect(reason)}")
end

IO.puts("")

# ============================================================================
# Benchmarks
# ============================================================================

Output.step(4, 4, "Running benchmarks")

IO.puts("")
results = []

# -- Server-side benchmark (no serialization overhead) --
Output.puts("Server-side benchmark (no serialization):")

case ExPhil.Bridge.MojoPort.benchmark(batch, seq_len, hidden, warmup: warmup, iterations: iterations) do
  {:ok, stats} ->
    numpy_med = stats["numpy_median_us"]
    IO.puts("  NumPy vectorized:  #{round(numpy_med)} us (median)")
    results = [{:numpy_internal, numpy_med} | results]

    if mojo_med = stats["mojo_median_us"] do
      IO.puts("  Mojo SIMD:         #{round(mojo_med)} us (median)")
      results = [{:mojo_internal, mojo_med} | results]
    end

  {:error, reason} ->
    Output.error("Server benchmark failed: #{inspect(reason)}")
end

IO.puts("")

# -- End-to-end (including serialization) --
Output.puts("End-to-end (Elixir -> Python -> Elixir):")

# Warmup
for _ <- 1..warmup do
  ExPhil.Bridge.MojoPort.linear_scan!(a_vals, b_vals, h0)
end

e2e_times =
  for _ <- 1..iterations do
    t0 = System.monotonic_time(:microsecond)
    ExPhil.Bridge.MojoPort.linear_scan!(a_vals, b_vals, h0)
    System.monotonic_time(:microsecond) - t0
  end

sorted = Enum.sort(e2e_times)
e2e_median = Enum.at(sorted, div(length(sorted), 2))
IO.puts("  Mojo/NumPy e2e: #{e2e_median} us (median), #{Enum.min(e2e_times)} us (min)")
results = [{:mojo_e2e, e2e_median} | results]

IO.puts("")

# -- CUDA C reference --
Output.puts("CUDA C reference (FusedScan):")

a_gpu = Nx.backend_transfer(a_vals, EXLA.Backend)
b_gpu = Nx.backend_transfer(b_vals, EXLA.Backend)

for _ <- 1..warmup do
  Edifice.CUDA.FusedScan.linear_scan(a_gpu, b_gpu) |> Nx.backend_transfer(Nx.BinaryBackend)
end

cuda_times =
  for _ <- 1..iterations do
    t0 = System.monotonic_time(:microsecond)
    Edifice.CUDA.FusedScan.linear_scan(a_gpu, b_gpu) |> Nx.backend_transfer(Nx.BinaryBackend)
    System.monotonic_time(:microsecond) - t0
  end

cuda_sorted = Enum.sort(cuda_times)
cuda_median = Enum.at(cuda_sorted, div(length(cuda_sorted), 2))
IO.puts("  CUDA C:      #{cuda_median} us (median), #{Enum.min(cuda_times)} us (min)")
results = [{:cuda_c, cuda_median} | results]

# ============================================================================
# Summary
# ============================================================================

IO.puts("")
IO.puts(String.duplicate("=", 60))
IO.puts("Summary")
IO.puts(String.duplicate("=", 60))

results_map = Map.new(results)

for {label, key} <- [
      {"Mojo SIMD (kernel)", :mojo_internal},
      {"NumPy vectorized (kernel)", :numpy_internal},
      {"Mojo/NumPy (e2e)", :mojo_e2e},
      {"CUDA C (FusedScan)", :cuda_c}
    ] do
  case Map.get(results_map, key) do
    nil -> :skip
    us ->
      label_pad = String.pad_trailing(label, 28)
      IO.puts("  #{label_pad} #{round(us)} us")
  end
end

IO.puts("")
Output.success("Benchmark complete")
