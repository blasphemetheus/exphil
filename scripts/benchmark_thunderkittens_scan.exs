#!/usr/bin/env elixir
# Benchmark: ThunderKittens CUDA NIF linear scan vs CUDA C vs Rust NIF vs Nx
#
# ThunderKittens (HazyResearch) is a tile-level C++ DSL for AI kernels.
# For fused_linear_scan, TK doesn't add advantage — it's designed for
# matmul/attention with tensor cores. This benchmark tests the NIF
# integration overhead, same as Rust-CUDA and Triton.
#
# Usage:
#   mix run scripts/benchmark_thunderkittens_scan.exs
#   mix run scripts/benchmark_thunderkittens_scan.exs --batch 32 --hidden 512

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
  Benchmark ThunderKittens Linear Scan (CUDA NIF)

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

Output.banner("ThunderKittens Linear Scan Benchmark (CUDA NIF)")

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

tk_available = ExPhil.Native.ThunderKittensScan.available?()

if tk_available do
  Output.success("ThunderKittens NIF loaded")

  case ExPhil.Native.ThunderKittensScan.device_info() do
    {:ok, name, {sm_major, sm_minor}, _tk_capable} ->
      Output.success("GPU: #{name} (sm_#{sm_major}#{sm_minor})")

      if sm_major >= 8 do
        Output.success("ThunderKittens native mode available (sm_80+)")
      else
        Output.puts("  Note: sm_#{sm_major}#{sm_minor} < sm_80 — using CUDA fallback kernel")
        Output.puts("  ThunderKittens tile primitives require Ampere (A100/RTX 3000+) or newer")
      end

    _ ->
      :ok
  end
else
  Output.warning("ThunderKittens NIF not available. Build with:")
  Output.puts("  cd native/thunderkittens_scan")
  Output.puts("  make && make install")
  Output.puts("")
  Output.puts("  Will only run CUDA C reference and Nx fallback")
end

xla_available = Code.ensure_loaded?(Edifice.CUDA.FusedScan)
if xla_available, do: Output.success("CUDA C (FusedScan): available")

rust_available = ExPhil.Native.RustLinearScan.available?()
if rust_available, do: Output.success("Rust-CUDA NIF: available")

IO.puts("")

# ============================================================================
# Benchmark harness
# ============================================================================

defmodule TKBench do
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

if tk_available do
  key = Nx.Random.key(42)
  {a_test, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {2, 10, 8}, type: :f32)
  {b_test, key} = Nx.Random.normal(key, shape: {2, 10, 8}, type: :f32)
  {h0_test, _} = Nx.Random.normal(key, 0.0, 0.1, shape: {2, 8}, type: :f32)

  tk_result = ExPhil.Native.ThunderKittensScan.linear_scan(a_test, b_test, h0_test)

  # Nx reference
  {_, nx_states} =
    Enum.reduce(0..9, {h0_test, []}, fn t, {h_state, acc} ->
      a_t = a_test[[.., t, ..]]
      b_t = b_test[[.., t, ..]]
      h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
      {h_new, [h_new | acc]}
    end)

  nx_ref = nx_states |> Enum.reverse() |> Nx.stack(axis: 1)

  diff = Nx.subtract(nx_ref, tk_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

  if diff < 1.0e-4 do
    Output.success("Correctness verified (max diff: #{Float.round(diff, 7)})")
  else
    Output.warning("Max diff #{Float.round(diff, 5)} exceeds 1e-4 — check implementation")
  end

  IO.puts("")
end

# ============================================================================
# Run benchmarks
# ============================================================================

Output.step(3, 4, "Running benchmarks")

IO.puts("")
IO.puts("seq_len | TK NIF (μs) | Rust NIF (μs) | CUDA C (μs) | Nx (μs)   | TK/CUDA C")
IO.puts(String.duplicate("-", 85))

results =
  Enum.map(seq_lengths, fn seq_len ->
    key = Nx.Random.key(42)
    {a_vals, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {batch, seq_len, hidden}, type: :f32)
    {b_vals, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
    {h0, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {batch, hidden}, type: :f32)

    a_gpu = Nx.backend_transfer(a_vals, EXLA.Backend)
    b_gpu = Nx.backend_transfer(b_vals, EXLA.Backend)

    # ThunderKittens NIF
    tk_med =
      if tk_available do
        case TKBench.try_bench(fn -> ExPhil.Native.ThunderKittensScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      end

    # Rust-CUDA NIF
    rust_med =
      if rust_available do
        case TKBench.try_bench(fn -> ExPhil.Native.RustLinearScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, {med, _, _}} -> med
          {:error, _} -> nil
        end
      end

    # CUDA C reference
    cuda_c_med =
      if xla_available do
        case TKBench.try_bench(fn -> Edifice.CUDA.FusedScan.linear_scan(a_gpu, b_gpu) end, warmup, iterations) do
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

    {nx_med, _, _} = TKBench.bench(nx_fn, warmup, iterations)

    # Format row
    seq_str = String.pad_leading("#{seq_len}", 7)
    tk_str = if tk_med, do: String.pad_leading("#{round(tk_med)}", 11), else: "        N/A"
    rust_str = if rust_med, do: String.pad_leading("#{round(rust_med)}", 13), else: "          N/A"
    cuda_str = if cuda_c_med, do: String.pad_leading("#{round(cuda_c_med)}", 11), else: "        N/A"
    nx_str = String.pad_leading("#{round(nx_med)}", 9)

    ratio_str =
      if tk_med && cuda_c_med do
        ratio = tk_med / max(cuda_c_med, 1)
        String.pad_leading(:erlang.float_to_binary(ratio, decimals: 2) <> "x", 9)
      else
        "      N/A"
      end

    IO.puts("#{seq_str} | #{tk_str} | #{rust_str} | #{cuda_str} | #{nx_str} | #{ratio_str}")

    %{seq_len: seq_len, tk: tk_med, rust: rust_med, cuda_c: cuda_c_med, nx: nx_med}
  end)

# ============================================================================
# Summary
# ============================================================================

Output.step(4, 4, "Summary")

IO.puts("")
IO.puts(String.duplicate("=", 70))
IO.puts("Analysis")
IO.puts(String.duplicate("=", 70))

tk_results = Enum.filter(results, & &1.tk)

if length(tk_results) > 0 do
  avg_ratio =
    tk_results
    |> Enum.filter(& &1.cuda_c)
    |> Enum.map(fn r -> r.tk / max(r.cuda_c, 1) end)
    |> then(fn ratios ->
      if length(ratios) > 0, do: Enum.sum(ratios) / length(ratios), else: nil
    end)

  if avg_ratio do
    IO.puts("")
    Output.puts("Average TK NIF / CUDA C ratio: #{:erlang.float_to_binary(avg_ratio, decimals: 2)}x")
  end
end

IO.puts("")
Output.puts("Note: ThunderKittens' value is for tile-level attention/matmul kernels,")
Output.puts("not simple sequential scans. For fused_linear_scan (h = a*h + b),")
Output.puts("the kernel is identical to standard CUDA C. The NIF overhead (~2-3x)")
Output.puts("is the same as all NIF-based approaches.")
Output.puts("")
Output.puts("TK shines when:")
Output.puts("  - Using tensor cores (matmul, attention) on sm_80+ GPUs")
Output.puts("  - Managing shared memory tiles (16x16+ blocks)")
Output.puts("  - Fusing multiple operations (e.g., attention + softmax + matmul)")

IO.puts("")
Output.success("Benchmark complete")
