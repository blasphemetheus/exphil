#!/usr/bin/env elixir
# Unified Kernel Language Comparison Benchmark
#
# Benchmarks all four GPU kernel languages (Julia, Futhark, Mojo, Bend)
# against the CUDA C reference for the fused_linear_scan kernel (h = a*h + b).
#
# Usage:
#   mix run scripts/benchmark_kernel_languages.exs
#   mix run scripts/benchmark_kernel_languages.exs --size training
#   mix run scripts/benchmark_kernel_languages.exs --batch 32 --seq 120 --hidden 512

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
      size: :string,
      help: :boolean
    ]
  )

if opts[:help] do
  IO.puts("""
  Unified Kernel Language Comparison Benchmark

  Options:
    --size SIZE        Preset size: "inference" (4,60,64), "training" (32,120,512), or "both" (default: both)
    --iterations N     Timed iterations (default: 30)
    --warmup N         Warmup iterations (default: 5)
    --batch N          Override batch size
    --seq N            Override sequence length
    --hidden N         Override hidden dimension
    --help             Show this help
  """)

  System.halt(0)
end

iterations = opts[:iterations] || 30
warmup = opts[:warmup] || 5
size_preset = opts[:size] || "both"

sizes =
  case size_preset do
    "inference" -> [%{label: "Inference", batch: 4, seq_len: 60, hidden: 64}]
    "training" -> [%{label: "Training", batch: 32, seq_len: 120, hidden: 512}]
    _ ->
      [
        %{label: "Inference (4×60×64)", batch: 4, seq_len: 60, hidden: 64},
        %{label: "Training (32×120×512)", batch: 32, seq_len: 120, hidden: 512}
      ]
  end

# Override from CLI
sizes =
  Enum.map(sizes, fn s ->
    s
    |> Map.update!(:batch, fn v -> opts[:batch] || v end)
    |> Map.update!(:seq_len, fn v -> opts[:seq] || v end)
    |> Map.update!(:hidden, fn v -> opts[:hidden] || v end)
  end)

Output.banner("GPU Kernel Language Comparison")

Output.config([
  {"Sizes", Enum.map_join(sizes, ", ", & &1.label)},
  {"Warmup", warmup},
  {"Iterations", iterations}
])

IO.puts("")

# ============================================================================
# Check availability of each language
# ============================================================================

Output.step(1, 3, "Checking language availability")

# CUDA C reference
xla_available = Code.ensure_loaded?(Edifice.CUDA.FusedScan)
if xla_available, do: Output.success("CUDA C (FusedScan): available"), else: Output.puts("  CUDA C: not available")

# Rust-CUDA (Rustler NIF)
rust_available = ExPhil.Native.RustLinearScan.available?()
if rust_available, do: Output.success("Rust-CUDA NIF: available"), else: Output.puts("  Rust-CUDA: not available")

# Triton AOT (C NIF)
triton_available = ExPhil.Native.TritonScan.available?()
if triton_available, do: Output.success("Triton AOT NIF: available"), else: Output.puts("  Triton AOT: not available")

# Julia
julia_available =
  case ExPhil.Bridge.JuliaPort.start_link() do
    {:ok, _} ->
      case ExPhil.Bridge.JuliaPort.ping() do
        {:ok, _} -> true
        _ -> false
      end
    _ -> false
  end

if julia_available, do: Output.success("Julia CUDA.jl: available"), else: Output.puts("  Julia: not available")

# Futhark
futhark_available = ExPhil.Native.FutharkScan.available?()
if futhark_available, do: Output.success("Futhark NIF: available"), else: Output.puts("  Futhark: not available")

# Mojo/NumPy
mojo_available =
  case ExPhil.Bridge.MojoPort.start_link() do
    {:ok, _} ->
      case ExPhil.Bridge.MojoPort.ping() do
        {:ok, _} -> true
        _ -> false
      end
    _ -> false
  end

if mojo_available, do: Output.success("Mojo/NumPy: available"), else: Output.puts("  Mojo/NumPy: not available")

# CuPy/CCCL (Port)
cupy_available =
  case ExPhil.Bridge.CudaComputePort.start_link() do
    {:ok, _} ->
      case ExPhil.Bridge.CudaComputePort.ping() do
        {:ok, _} -> true
        _ -> false
      end
    _ -> false
  end

if cupy_available, do: Output.success("CuPy/CCCL: available"), else: Output.puts("  CuPy/CCCL: not available")

# ThunderKittens (CUDA NIF)
tk_available = ExPhil.Native.ThunderKittensScan.available?()
if tk_available, do: Output.success("ThunderKittens NIF: available"), else: Output.puts("  ThunderKittens: not available")

# Bend
bend_available = ExPhil.Bridge.BendPort.available?()
if bend_available, do: Output.success("Bend: available"), else: Output.puts("  Bend: not available (learning exercise only)")

IO.puts("")

# ============================================================================
# Benchmark harness
# ============================================================================

Nx.default_backend(EXLA.Backend)

defmodule UnifiedBench do
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
    Enum.at(sorted, div(length(sorted), 2))
  end

  def try_bench(fun, warmup_iters, timed_iters) do
    try do
      {:ok, bench(fun, warmup_iters, timed_iters)}
    rescue
      _ -> {:error, :failed}
    end
  end
end

# ============================================================================
# Run benchmarks
# ============================================================================

Output.step(2, 3, "Running benchmarks")

all_results =
  Enum.map(sizes, fn %{label: label, batch: batch, seq_len: seq_len, hidden: hidden} ->
    IO.puts("")
    Output.puts("--- #{label}: batch=#{batch}, seq_len=#{seq_len}, hidden=#{hidden} ---")

    key = Nx.Random.key(42)
    {a_vals, key} = Nx.Random.uniform(key, 0.5, 0.99, shape: {batch, seq_len, hidden}, type: :f32)
    {b_vals, key} = Nx.Random.normal(key, shape: {batch, seq_len, hidden}, type: :f32)
    {h0, _key} = Nx.Random.normal(key, 0.0, 0.1, shape: {batch, hidden}, type: :f32)

    a_gpu = Nx.backend_transfer(a_vals, EXLA.Backend)
    b_gpu = Nx.backend_transfer(b_vals, EXLA.Backend)

    results = %{label: label, batch: batch, seq_len: seq_len, hidden: hidden}

    # CUDA C reference
    results =
      if xla_available do
        case UnifiedBench.try_bench(fn -> Edifice.CUDA.FusedScan.linear_scan(a_gpu, b_gpu) end, warmup, iterations) do
          {:ok, med} ->
            IO.puts("  CUDA C:      #{med} us")
            Map.put(results, :cuda_c, med)
          {:error, _} ->
            Map.put(results, :cuda_c, nil)
        end
      else
        Map.put(results, :cuda_c, nil)
      end

    # Rust-CUDA (Rustler NIF)
    results =
      if rust_available do
        case UnifiedBench.try_bench(fn -> ExPhil.Native.RustLinearScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, med} ->
            IO.puts("  Rust NIF:    #{med} us")
            Map.put(results, :rust_nif, med)
          {:error, _} ->
            Map.put(results, :rust_nif, nil)
        end
      else
        Map.put(results, :rust_nif, nil)
      end

    # Triton AOT (C NIF)
    results =
      if triton_available do
        case UnifiedBench.try_bench(fn -> ExPhil.Native.TritonScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, med} ->
            IO.puts("  Triton AOT:  #{med} us")
            Map.put(results, :triton_aot, med)
          {:error, _} ->
            Map.put(results, :triton_aot, nil)
        end
      else
        Map.put(results, :triton_aot, nil)
      end

    # Julia (end-to-end)
    results =
      if julia_available do
        case UnifiedBench.try_bench(fn -> ExPhil.Bridge.JuliaPort.linear_scan!(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, med} ->
            IO.puts("  Julia e2e:   #{med} us")
            Map.put(results, :julia_e2e, med)
          {:error, _} ->
            Map.put(results, :julia_e2e, nil)
        end
      else
        Map.put(results, :julia_e2e, nil)
      end

    # Futhark (NIF)
    results =
      if futhark_available do
        case UnifiedBench.try_bench(fn -> ExPhil.Native.FutharkScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, med} ->
            IO.puts("  Futhark NIF: #{med} us")
            Map.put(results, :futhark, med)
          {:error, _} ->
            Map.put(results, :futhark, nil)
        end
      else
        Map.put(results, :futhark, nil)
      end

    # Mojo/NumPy (end-to-end)
    results =
      if mojo_available do
        case UnifiedBench.try_bench(fn -> ExPhil.Bridge.MojoPort.linear_scan!(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, med} ->
            IO.puts("  Mojo/NumPy:  #{med} us")
            Map.put(results, :mojo_e2e, med)
          {:error, _} ->
            Map.put(results, :mojo_e2e, nil)
        end
      else
        Map.put(results, :mojo_e2e, nil)
      end

    # CuPy/CCCL (end-to-end via Port)
    results =
      if cupy_available do
        case UnifiedBench.try_bench(fn -> ExPhil.Bridge.CudaComputePort.linear_scan!(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, med} ->
            IO.puts("  CuPy/CCCL:  #{med} us")
            Map.put(results, :cupy_e2e, med)
          {:error, _} ->
            Map.put(results, :cupy_e2e, nil)
        end
      else
        Map.put(results, :cupy_e2e, nil)
      end

    # ThunderKittens (NIF)
    results =
      if tk_available do
        case UnifiedBench.try_bench(fn -> ExPhil.Native.ThunderKittensScan.linear_scan(a_vals, b_vals, h0) end, warmup, iterations) do
          {:ok, med} ->
            IO.puts("  TK NIF:     #{med} us")
            Map.put(results, :tk_nif, med)
          {:error, _} ->
            Map.put(results, :tk_nif, nil)
        end
      else
        Map.put(results, :tk_nif, nil)
      end

    # Pure Nx fallback
    nx_fn = fn ->
      Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_state, acc} ->
        a_t = a_vals[[.., t, ..]]
        b_t = b_vals[[.., t, ..]]
        h_new = Nx.add(Nx.multiply(a_t, h_state), b_t)
        {h_new, [h_new | acc]}
      end)
      |> then(fn {_, states} -> states |> Enum.reverse() |> Nx.stack(axis: 1) end)
    end

    nx_med = UnifiedBench.bench(nx_fn, warmup, iterations)
    IO.puts("  Nx fallback: #{nx_med} us")
    results = Map.put(results, :nx, nx_med)

    results
  end)

# ============================================================================
# Summary table
# ============================================================================

Output.step(3, 3, "Summary")

IO.puts("")

for result <- all_results do
  IO.puts(String.duplicate("=", 70))
  IO.puts("#{result.label}")
  IO.puts(String.duplicate("=", 70))
  IO.puts("")
  IO.puts("Implementation         | Median (μs) | vs CUDA C | vs Nx")
  IO.puts(String.duplicate("-", 70))

  cuda_c = result[:cuda_c]

  for {label, key} <- [
        {"CUDA C (FusedScan)", :cuda_c},
        {"Rust-CUDA (NIF)", :rust_nif},
        {"Triton AOT (NIF)", :triton_aot},
        {"ThunderKittens (NIF)", :tk_nif},
        {"CuPy/CCCL (e2e)", :cupy_e2e},
        {"Julia CUDA.jl (e2e)", :julia_e2e},
        {"Futhark (NIF)", :futhark},
        {"Mojo/NumPy (e2e)", :mojo_e2e},
        {"Pure Nx fallback", :nx}
      ] do
    case Map.get(result, key) do
      nil ->
        IO.puts("#{String.pad_trailing(label, 22)} |         N/A |       N/A |   N/A")

      us ->
        us_str = String.pad_leading("#{round(us)}", 9)

        cuda_ratio =
          if cuda_c && cuda_c > 0 do
            r = us / cuda_c
            String.pad_leading(:erlang.float_to_binary(r, decimals: 1) <> "x", 9)
          else
            "      N/A"
          end

        nx_ratio =
          if result[:nx] && result[:nx] > 0 do
            r = result[:nx] / max(us, 1)
            String.pad_leading(:erlang.float_to_binary(r, decimals: 1) <> "x", 5)
          else
            "  N/A"
          end

        IO.puts("#{String.pad_trailing(label, 22)} | #{us_str}  | #{cuda_ratio} | #{nx_ratio}")
    end
  end

  IO.puts("")
end

# Code size comparison
IO.puts(String.duplicate("=", 70))
IO.puts("Code Size Comparison")
IO.puts(String.duplicate("=", 70))
IO.puts("")
IO.puts("Language    | Kernel | Glue/Integration | Total | Dependencies")
IO.puts(String.duplicate("-", 70))
IO.puts("CUDA C      |  ~20   |      ~130        |  ~150 | nvcc, CUDA toolkit")
IO.puts("Rust-CUDA   |  ~20   |      ~295        |  ~315 | cargo, rustler, cudarc")
IO.puts("Triton AOT  |  ~15   |      ~370        |  ~385 | Triton (build), gcc, CUDA")
IO.puts("TK (CUDA)   |  ~50   |      ~160        |  ~210 | nvcc, gcc, CUDA (+ TK headers)")
IO.puts("CuPy/CCCL   |  ~65   |      ~200        |  ~265 | CuPy, msgpack, Python")
IO.puts("Julia       |  ~25   |      ~250        |  ~275 | Julia, CUDA.jl, MsgPack.jl")
IO.puts("Futhark     |  ~15   |      ~190        |  ~205 | futhark compiler")
IO.puts("Mojo        |  ~30   |      ~245        |  ~275 | Mojo SDK (or NumPy fallback)")
IO.puts("Bend        |  ~10   |      ~100        |  ~110 | bend-lang (cargo)")
IO.puts("")

IO.puts(String.duplicate("=", 70))
IO.puts("Developer Experience")
IO.puts(String.duplicate("=", 70))
IO.puts("")
IO.puts("Language    | Readability | Debuggability | Ecosystem | Stability")
IO.puts(String.duplicate("-", 70))
IO.puts("CUDA C      | Medium      | Good (nsight) | Excellent | Stable")
IO.puts("Rust-CUDA   | Medium      | Good (cudarc) | Good      | Stable")
IO.puts("Triton AOT  | High        | Good (Python) | Excellent | Stable")
IO.puts("TK (CUDA)   | Medium      | Good (nsight) | Medium    | Stable (sm_80+)")
IO.puts("CuPy/CCCL   | High        | Good (Python) | Good      | Stable")
IO.puts("Julia       | High        | Good (REPL)   | Good      | Stable")
IO.puts("Futhark     | High        | Limited       | Tiny      | Stable")
IO.puts("Mojo        | High        | Poor          | Growing   | Pre-1.0")
IO.puts("Bend        | Medium      | Very poor     | Tiny      | Experimental")
IO.puts("")

Output.success("Unified benchmark complete")
IO.puts("")
IO.puts("For detailed analysis, see: docs/research/KERNEL_LANGUAGE_COMPARISON.md")
