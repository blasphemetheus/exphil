#!/usr/bin/env elixir
# Benchmark: Bend (HVM2) linear scan — learning exercise
#
# Bend auto-parallelizes based on data dependencies. For linear scan,
# the sequential recurrence can't be parallelized over timesteps,
# but batch and hidden dimensions can be.
#
# This is NOT a performance benchmark against CUDA C — Bend is expected
# to be 10-100x slower. The goal is to understand Bend's execution model.
#
# Usage:
#   mix run scripts/benchmark_bend_scan.exs

alias ExPhil.Training.Output

require Output

Output.banner("Bend (HVM2) Linear Scan — Learning Exercise")

IO.puts("")

# ============================================================================
# Check availability
# ============================================================================

Output.step(1, 3, "Checking Bend availability")

if ExPhil.Bridge.BendPort.available?() do
  Output.success("Bend found in PATH")
else
  Output.warning("Bend not found. Install with: cargo install bend-lang")
  Output.puts("  Requires Rust toolchain (already in shell.nix)")
  Output.puts("")
  Output.puts("  Continuing with Nx-only comparison...")
end

IO.puts("")

# ============================================================================
# Run correctness test
# ============================================================================

Output.step(2, 3, "Running correctness test")

if ExPhil.Bridge.BendPort.available?() do
  for backend <- ["rust", "c"] do
    case ExPhil.Bridge.BendPort.run_test(backend: backend) do
      {:ok, output} ->
        Output.success("#{backend} backend:")
        # Indent output
        output
        |> String.split("\n")
        |> Enum.each(fn line -> IO.puts("    #{line}") end)

      {:error, reason} ->
        Output.error("#{backend} backend failed: #{reason}")
    end
  end

  # Try CUDA backend separately (likely to fail)
  case ExPhil.Bridge.BendPort.run_test(backend: "cuda") do
    {:ok, output} ->
      Output.success("CUDA backend:")
      output |> String.split("\n") |> Enum.each(fn line -> IO.puts("    #{line}") end)

    {:error, _reason} ->
      Output.puts("  CUDA backend: not available (HVM2 GPU backend is experimental)")
  end
else
  Output.puts("  Skipping (Bend not installed)")
end

IO.puts("")

# ============================================================================
# Benchmark
# ============================================================================

Output.step(3, 3, "Benchmarking (tiny size: batch=1, seq_len=4, hidden=4)")

# Use tiny sizes that Bend can handle
results = ExPhil.Bridge.BendPort.benchmark(1, 4, 4, iterations: 5)

IO.puts("")
IO.puts(String.duplicate("=", 60))
IO.puts("Results (batch=1, seq_len=4, hidden=4)")
IO.puts(String.duplicate("=", 60))

if results.bend_available do
  IO.puts("  Bend (Rust backend): #{results.bend_median_us} us (median)")
  IO.puts("    Note: includes compilation + process spawn overhead")
end

IO.puts("  Pure Nx (sequential): #{results.nx_median_us} us (median)")

if results.bend_available && results.bend_median_us do
  ratio = results.bend_median_us / max(results.nx_median_us, 1)
  IO.puts("")
  IO.puts("  Bend/Nx ratio: #{:erlang.float_to_binary(ratio, decimals: 1)}x")
  IO.puts("  (Expected: Bend is much slower due to compilation + process overhead)")
end

IO.puts("")
IO.puts(String.duplicate("=", 60))
IO.puts("Key Insights")
IO.puts(String.duplicate("=", 60))
IO.puts("")
IO.puts("  1. Bend's execution model (interaction nets) is fundamentally")
IO.puts("     different from GPU thread blocks. It auto-parallelizes based")
IO.puts("     on data dependencies, not explicit thread layouts.")
IO.puts("")
IO.puts("  2. Linear scan is inherently sequential over timesteps.")
IO.puts("     Bend can only parallelize across batch/hidden dims,")
IO.puts("     same as the CUDA C kernel.")
IO.puts("")
IO.puts("  3. Bend uses f24 (24-bit floats), not IEEE 754 f32.")
IO.puts("     This limits numerical precision for ML workloads.")
IO.puts("")
IO.puts("  4. No binary tensor I/O or C FFI makes integration with")
IO.puts("     existing ML pipelines impractical.")
IO.puts("")
IO.puts("  5. The HVM2 GPU backend uses interaction net evaluation,")
IO.puts("     not CUDA kernels. It's optimized for tree-structured")
IO.puts("     parallelism, not dense tensor operations.")
IO.puts("")

Output.success("Exploration complete")
