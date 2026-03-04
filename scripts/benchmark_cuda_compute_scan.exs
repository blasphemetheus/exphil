#!/usr/bin/env elixir
# Benchmark: CuPy/CCCL linear scan vs CUDA C vs Rust NIF vs Nx
#
# Tests sequential GPU scan and parallel prefix scan (Blelloch algorithm)
# via CuPy RawKernel, accessed through an Elixir Port.
#
# Usage:
#   mix run scripts/benchmark_cuda_compute_scan.exs
#   mix run scripts/benchmark_cuda_compute_scan.exs --batch 32 --hidden 512
#   mix run scripts/benchmark_cuda_compute_scan.exs --seq-lengths 30,60,120,240,480,960

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
  Benchmark CuPy/CCCL Linear Scan (Port-based)

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

Output.banner("CuPy/CCCL Linear Scan Benchmark (Port-based)")

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

Output.step(1, 4, "Checking availability")

cupy_available =
  case ExPhil.Bridge.CudaComputePort.start_link() do
    {:ok, _} ->
      case ExPhil.Bridge.CudaComputePort.ping() do
        {:ok, resp} ->
          device = resp["device"] || "unknown"
          Output.success("CuPy/CCCL server started (device: #{device})")
          true

        _ ->
          Output.warning("CuPy/CCCL server started but ping failed")
          false
      end

    _ ->
      Output.warning("CuPy/CCCL server failed to start")
      false
  end

if cupy_available do
  case ExPhil.Bridge.CudaComputePort.info() do
    {:ok, info} ->
      if info["cupy_available"] do
        Output.success("CuPy #{info["cupy_version"] || "?"} on #{info["gpu_name"] || "GPU"}")
        Output.success("Modes available: #{inspect(info["modes"])}")
      else
        Output.puts("  CuPy not available — using NumPy CPU fallback")
      end

    _ ->
      :ok
  end
end

xla_available = Code.ensure_loaded?(Edifice.CUDA.FusedScan)
if xla_available, do: Output.success("CUDA C (FusedScan): available")

rust_available = ExPhil.Native.RustLinearScan.available?()
if rust_available, do: Output.success("Rust-CUDA NIF: available")

IO.puts("")

# ============================================================================
# Benchmark harness
# ============================================================================

defmodule CuPyBench do
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

Output.step(2, 4, "Correctness check")

if cupy_available do
  key = Nx.Random.key(42)
  {a_test, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {2, 10, 8}, type: :f32)
  {b_test, key} = Nx.Random.normal(key, shape: {2, 10, 8}, type: :f32)
  {h0_test, _} = Nx.Random.normal(key, 0.0, 0.1, shape: {2, 8}, type: :f32)

  # Nx reference
  {_, nx_states} =
    Enum.reduce(0..9, {h0_test, []}, fn t, {h_state, acc} ->
      a_t = a_test[[.., t, ..]]
      b_t = b_test[[.., t, ..]]
      h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
      {h_new, [h_new | acc]}
    end)

  nx_ref = nx_states |> Enum.reverse() |> Nx.stack(axis: 1)

  # Check sequential mode
  case ExPhil.Bridge.CudaComputePort.linear_scan(a_test, b_test, h0_test, mode: "sequential") do
    {:ok, seq_result} ->
      diff = Nx.subtract(nx_ref, seq_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      if diff < 1.0e-4 do
        Output.success("Sequential mode correct (max diff: #{Float.round(diff, 7)})")
      else
        Output.warning("Sequential mode max diff #{Float.round(diff, 5)} exceeds 1e-4")
      end

    {:error, reason} ->
      Output.warning("Sequential mode failed: #{inspect(reason)}")
  end

  # Check parallel mode
  case ExPhil.Bridge.CudaComputePort.linear_scan(a_test, b_test, h0_test, mode: "parallel") do
    {:ok, par_result} ->
      diff = Nx.subtract(nx_ref, par_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      if diff < 1.0e-3 do
        Output.success("Parallel mode correct (max diff: #{Float.round(diff, 7)})")
      else
        Output.warning("Parallel mode max diff #{Float.round(diff, 5)} exceeds 1e-3")
      end

    {:error, reason} ->
      Output.warning("Parallel mode failed: #{inspect(reason)}")
  end
end

IO.puts("")

# ============================================================================
# Run benchmarks
# ============================================================================

Output.step(3, 4, "Running benchmarks")

IO.puts("")
IO.puts("seq_len | CuPy Seq (μs) | CuPy Par (μs) | CUDA C (μs) | Nx (μs)    | CuPy/CUDA C")
IO.puts(String.duplicate("-", 90))

results =
  Enum.map(seq_lengths, fn seq_len ->
    key = Nx.Random.key(42)
    {a_vals, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {batch, seq_len, hidden}, type: :f32)
    {b_vals, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
    {h0, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {batch, hidden}, type: :f32)

    a_gpu = Nx.backend_transfer(a_vals, EXLA.Backend)
    b_gpu = Nx.backend_transfer(b_vals, EXLA.Backend)

    # CuPy sequential
    cupy_seq_med =
      if cupy_available do
        case CuPyBench.try_bench(
               fn -> ExPhil.Bridge.CudaComputePort.linear_scan!(a_vals, b_vals, h0, mode: "sequential") end,
               warmup,
               iterations
             ) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      end

    # CuPy parallel prefix scan
    cupy_par_med =
      if cupy_available do
        case CuPyBench.try_bench(
               fn -> ExPhil.Bridge.CudaComputePort.linear_scan!(a_vals, b_vals, h0, mode: "parallel") end,
               warmup,
               iterations
             ) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      end

    # CUDA C reference
    cuda_c_med =
      if xla_available do
        case CuPyBench.try_bench(
               fn -> Edifice.CUDA.FusedScan.linear_scan(a_gpu, b_gpu) end,
               warmup,
               iterations
             ) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      end

    # Nx fallback
    nx_fn = fn ->
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_state, acc} ->
        a_t = a_vals[[.., t, ..]]
        b_t = b_vals[[.., t, ..]]
        h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
        {h_new, [h_new | acc]}
      end)
      |> then(fn {_, states} -> states |> Enum.reverse() |> Nx.stack(axis: 1) end)
    end

    {nx_med, _, _} = CuPyBench.bench(nx_fn, warmup, iterations)

    # Format row
    seq_str = String.pad_leading("#{seq_len}", 7)
    cupy_seq_str = if cupy_seq_med, do: String.pad_leading("#{round(cupy_seq_med)}", 13), else: "          N/A"
    cupy_par_str = if cupy_par_med, do: String.pad_leading("#{round(cupy_par_med)}", 14), else: "           N/A"
    cuda_str = if cuda_c_med, do: String.pad_leading("#{round(cuda_c_med)}", 11), else: "        N/A"
    nx_str = String.pad_leading("#{round(nx_med)}", 10)

    ratio_str =
      if cupy_seq_med && cuda_c_med do
        ratio = cupy_seq_med / max(cuda_c_med, 1)
        String.pad_leading(:erlang.float_to_binary(ratio, decimals: 2) <> "x", 11)
      else
        "        N/A"
      end

    IO.puts("#{seq_str} | #{cupy_seq_str} | #{cupy_par_str} | #{cuda_str} | #{nx_str} | #{ratio_str}")

    %{seq_len: seq_len, cupy_seq: cupy_seq_med, cupy_par: cupy_par_med, cuda_c: cuda_c_med, nx: nx_med}
  end)

# ============================================================================
# Summary
# ============================================================================

Output.step(4, 4, "Summary")

IO.puts("")
IO.puts(String.duplicate("=", 70))
IO.puts("Analysis")
IO.puts(String.duplicate("=", 70))
IO.puts("")

cupy_results = Enum.filter(results, & &1.cupy_seq)

if length(cupy_results) > 0 do
  avg_ratio =
    cupy_results
    |> Enum.filter(& &1.cuda_c)
    |> Enum.map(fn r -> r.cupy_seq / max(r.cuda_c, 1) end)
    |> then(fn ratios ->
      if length(ratios) > 0, do: Enum.sum(ratios) / length(ratios), else: nil
    end)

  if avg_ratio do
    Output.puts("Average CuPy Sequential / CUDA C ratio: #{:erlang.float_to_binary(avg_ratio, decimals: 2)}x")
  end

  # Compare sequential vs parallel
  par_results = Enum.filter(results, fn r -> r.cupy_seq && r.cupy_par end)

  if length(par_results) > 0 do
    IO.puts("")
    Output.puts("Sequential vs Parallel prefix scan:")

    for r <- par_results do
      ratio = r.cupy_par / max(r.cupy_seq, 1)
      par_label = if ratio < 1, do: "parallel faster", else: "sequential faster"
      Output.puts("  seq_len=#{r.seq_len}: par/seq = #{:erlang.float_to_binary(ratio, decimals: 2)}x (#{par_label})")
    end
  end
end

IO.puts("")
Output.puts("Key insight: CuPy via Port adds serialization overhead (msgpack + stdio)")
Output.puts("on top of the CuPy→GPU data copy. The parallel prefix scan (Blelloch) uses")
Output.puts("O(log T) depth but does 2x the work, trading compute for parallelism.")

IO.puts("")
Output.success("Benchmark complete")
