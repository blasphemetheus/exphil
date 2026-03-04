#!/usr/bin/env elixir
# Benchmark: Julia CUDA.jl linear scan vs CUDA C reference
#
# Compares Julia's CUDA.jl and KernelAbstractions.jl implementations
# against the CUDA C fused_linear_scan kernel (via FusedScan dispatch).
#
# Usage:
#   mix run scripts/benchmark_julia_scan.exs
#   mix run scripts/benchmark_julia_scan.exs --iterations 50
#   mix run scripts/benchmark_julia_scan.exs --batch 32 --seq 120 --hidden 512

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
      help: :boolean,
      skip_reference: :boolean
    ]
  )

if opts[:help] do
  IO.puts("""
  Benchmark Julia CUDA.jl Linear Scan

  Options:
    --iterations N     Timed iterations (default: 30)
    --warmup N         Warmup iterations (default: 5)
    --batch N          Batch size (default: 4)
    --seq N            Sequence length (default: 60)
    --hidden N         Hidden dimension (default: 64)
    --skip-reference   Skip CUDA C reference benchmark
    --help             Show this help
  """)

  System.halt(0)
end

iterations = opts[:iterations] || 30
warmup = opts[:warmup] || 5
batch = opts[:batch] || 4
seq_len = opts[:seq] || 60
hidden = opts[:hidden] || 64
skip_reference = opts[:skip_reference] || false

Output.banner("Julia CUDA.jl Linear Scan Benchmark")

Output.config([
  {"Batch", batch},
  {"Sequence length", seq_len},
  {"Hidden dim", hidden},
  {"Tensor size", "#{batch * seq_len * hidden * 4 / 1024} KB (f32)"},
  {"Warmup", warmup},
  {"Iterations", iterations}
])

IO.puts("")

# ============================================================================
# Start Julia server
# ============================================================================

Output.step(1, 4, "Starting Julia server (JIT warmup ~30-60s)")

julia_available =
  case ExPhil.Bridge.JuliaPort.start_link() do
    {:ok, _pid} ->
      Output.puts("  Waiting for Julia JIT warmup...")

      case ExPhil.Bridge.JuliaPort.ping() do
        {:ok, info} ->
          Output.success("Julia server ready (device: #{info["device"]})")
          true

        {:error, reason} ->
          Output.error("Julia ping failed: #{inspect(reason)}")
          false
      end

    {:error, reason} ->
      Output.error("Failed to start Julia: #{inspect(reason)}")
      false
  end

if not julia_available do
  Output.error("Julia not available. Run: cd native/julia_scan && bash setup.sh")
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

# Reference: pure Nx sequential scan
reference_fn = fn ->
  Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_state, acc} ->
    a_t = a_vals[[.., t, ..]]
    b_t = b_vals[[.., t, ..]]
    h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
    {h_new, [h_new | acc]}
  end)
  |> then(fn {_final, states} ->
    states
    |> Enum.reverse()
    |> Nx.stack(axis: 1)
  end)
end

reference = reference_fn.()

# Julia CUDA.jl
case ExPhil.Bridge.JuliaPort.linear_scan(a_vals, b_vals, h0, mode: "cuda") do
  {:ok, julia_cuda} ->
    diff = Nx.subtract(reference, julia_cuda) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    if diff < 1.0e-4 do
      Output.success("CUDA.jl: correct (max diff: #{Float.round(diff, 7)})")
    else
      Output.warning("CUDA.jl: max diff #{Float.round(diff, 4)} exceeds 1e-4 threshold")
    end

  {:error, reason} ->
    Output.error("CUDA.jl scan failed: #{inspect(reason)}")
end

# Julia KernelAbstractions
case ExPhil.Bridge.JuliaPort.linear_scan(a_vals, b_vals, h0, mode: "ka") do
  {:ok, julia_ka} ->
    diff = Nx.subtract(reference, julia_ka) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    if diff < 1.0e-4 do
      Output.success("KernelAbstractions: correct (max diff: #{Float.round(diff, 7)})")
    else
      Output.warning("KernelAbstractions: max diff #{Float.round(diff, 4)} exceeds 1e-4 threshold")
    end

  {:error, reason} ->
    Output.error("KA scan failed: #{inspect(reason)}")
end

IO.puts("")

# ============================================================================
# Benchmarks
# ============================================================================

Output.step(4, 4, "Running benchmarks")

IO.puts("")

results =
  case ExPhil.Bridge.JuliaPort.benchmark(batch, seq_len, hidden, warmup: warmup, iterations: iterations) do
    {:ok, stats} ->
      cuda_med = stats["cuda_median_us"]
      ka_med = stats["ka_median_us"]

      Output.puts("Julia internal benchmark (GPU-only timing, no serialization):")
      IO.puts("  CUDA.jl kernel:    #{round(cuda_med)} μs (median)")
      IO.puts("  KA kernel:         #{round(ka_med)} μs (median)")
      [{:julia_cuda_internal, cuda_med}, {:julia_ka_internal, ka_med}]

    {:error, reason} ->
      Output.error("Julia benchmark failed: #{inspect(reason)}")
      []
  end

IO.puts("")

# -- Julia end-to-end (including serialization) --
Output.puts("Julia end-to-end (Elixir → Julia → Elixir, including serialization):")

bench_e2e = fn label, mode ->
  # Warmup
  for _ <- 1..warmup do
    ExPhil.Bridge.JuliaPort.linear_scan!(a_vals, b_vals, h0, mode: mode)
  end

  # Timed
  times =
    for _ <- 1..iterations do
      t0 = System.monotonic_time(:microsecond)
      ExPhil.Bridge.JuliaPort.linear_scan!(a_vals, b_vals, h0, mode: mode)
      System.monotonic_time(:microsecond) - t0
    end

  sorted = Enum.sort(times)
  median = Enum.at(sorted, div(length(sorted), 2))
  IO.puts("  #{label}: #{median} μs (median), #{Enum.min(times)} μs (min), #{Enum.max(times)} μs (max)")
  median
end

cuda_e2e = bench_e2e.("CUDA.jl e2e", "cuda")
ka_e2e = bench_e2e.("KA e2e     ", "ka")
results = [{:julia_cuda_e2e, cuda_e2e}, {:julia_ka_e2e, ka_e2e} | results]

IO.puts("")

# -- CUDA C reference (via FusedScan dispatch) --
results =
  if not skip_reference do
    Output.puts("CUDA C reference (FusedScan XLA custom call / NIF):")

    a_gpu = Nx.backend_transfer(a_vals, EXLA.Backend)
    b_gpu = Nx.backend_transfer(b_vals, EXLA.Backend)

    # Warmup
    for _ <- 1..warmup do
      Edifice.CUDA.FusedScan.linear_scan(a_gpu, b_gpu) |> Nx.backend_transfer(Nx.BinaryBackend)
    end

    cuda_c_times =
      for _ <- 1..iterations do
        t0 = System.monotonic_time(:microsecond)
        Edifice.CUDA.FusedScan.linear_scan(a_gpu, b_gpu) |> Nx.backend_transfer(Nx.BinaryBackend)
        System.monotonic_time(:microsecond) - t0
      end

    sorted = Enum.sort(cuda_c_times)
    cuda_c_median = Enum.at(sorted, div(length(sorted), 2))

    IO.puts("  CUDA C:    #{cuda_c_median} μs (median), #{Enum.min(cuda_c_times)} μs (min), #{Enum.max(cuda_c_times)} μs (max)")
    [{:cuda_c, cuda_c_median} | results]
  else
    results
  end

# -- Pure Nx fallback --
Output.puts("Pure Nx sequential fallback:")

for _ <- 1..warmup do
  reference_fn.()
end

nx_times =
  for _ <- 1..iterations do
    t0 = System.monotonic_time(:microsecond)
    reference_fn.()
    System.monotonic_time(:microsecond) - t0
  end

nx_sorted = Enum.sort(nx_times)
nx_median = Enum.at(nx_sorted, div(length(nx_sorted), 2))
IO.puts("  Nx fallback: #{nx_median} μs (median)")
results = [{:nx_fallback, nx_median} | results]

# ============================================================================
# Summary
# ============================================================================

IO.puts("")
IO.puts(String.duplicate("=", 60))
IO.puts("Summary")
IO.puts(String.duplicate("=", 60))

results_map = Map.new(results)

IO.puts("")
IO.puts("Implementation            | Median (μs) | vs Nx")
IO.puts(String.duplicate("-", 60))

for {label, key} <- [
      {"Julia CUDA.jl (kernel)", :julia_cuda_internal},
      {"Julia KA (kernel)", :julia_ka_internal},
      {"Julia CUDA.jl (e2e)", :julia_cuda_e2e},
      {"Julia KA (e2e)", :julia_ka_e2e},
      {"CUDA C (FusedScan)", :cuda_c},
      {"Pure Nx fallback", :nx_fallback}
    ] do
  case Map.get(results_map, key) do
    nil ->
      :skip

    us ->
      ratio = if nx_median > 0, do: nx_median / max(us, 1), else: 0
      label_pad = String.pad_trailing(label, 25)
      us_str = String.pad_leading("#{round(us)}", 10)
      ratio_str = :erlang.float_to_binary(ratio, decimals: 1) <> "x"
      IO.puts("#{label_pad} | #{us_str}  | #{ratio_str}")
  end
end

IO.puts("")
Output.success("Benchmark complete")
