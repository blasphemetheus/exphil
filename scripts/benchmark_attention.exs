#!/usr/bin/env elixir
# Benchmark attention implementations for choosing effective defaults
#
# Usage:
#   mix run scripts/benchmark_attention.exs [--seq-lens 32,64,128,256] [--iterations 50]
#
# This compares:
# - Standard scaled dot-product attention (O(n²) memory)
# - Chunked attention (lower peak memory, same algorithm)
# - Memory-efficient attention (online softmax, O(n) memory)
# - NIF FlashAttention (native Rust/CUDA, forward-only)
#
# The benchmark tests different sequence lengths to help choose the right
# implementation for your use case:
# - Short sequences (32-64): Standard attention is fine
# - Medium sequences (128-256): Memory-efficient starts to shine
# - Long sequences (512+): Memory-efficient or NIF required

alias ExPhil.Networks.Attention
alias ExPhil.Training.Output

require Output

Output.banner("Attention Implementation Benchmark")

# Parse CLI arguments
args = System.argv()

default_seq_lens = [32, 64, 128, 256]
seq_lens =
  case Enum.find_index(args, &(&1 == "--seq-lens")) do
    nil -> default_seq_lens
    idx ->
      args
      |> Enum.at(idx + 1, "")
      |> String.split(",")
      |> Enum.map(&String.to_integer/1)
  end

num_iterations =
  case Enum.find_index(args, &(&1 == "--iterations")) do
    nil -> 50
    idx -> args |> Enum.at(idx + 1, "50") |> String.to_integer()
  end

warmup_iterations = 5

# Configuration
batch_size = 8
hidden_dim = 256
num_heads = 4
head_dim = 64
chunk_size = 32

# Use GPU if available
Nx.default_backend(EXLA.Backend)

IO.puts("\nConfiguration:")
IO.puts("  Batch size:    #{batch_size}")
IO.puts("  Hidden dim:    #{hidden_dim}")
IO.puts("  Num heads:     #{num_heads}")
IO.puts("  Head dim:      #{head_dim}")
IO.puts("  Chunk size:    #{chunk_size}")
IO.puts("  Sequence lens: #{Enum.join(seq_lens, ", ")}")
IO.puts("  Iterations:    #{num_iterations}")
IO.puts("  Warmup:        #{warmup_iterations}")
IO.puts("")

# Check NIF availability
nif_available =
  try do
    ExPhil.Native.FlashAttention.backend_info()
    true
  rescue
    _ -> false
  end

nif_backend = if nif_available, do: ExPhil.Native.FlashAttention.backend_info(), else: "unavailable"
IO.puts("NIF Backend: #{nif_backend}")

if nif_available and ExPhil.Native.FlashAttention.cuda_available?() do
  IO.puts("  CUDA: ✓ Enabled (GPU acceleration)")
else
  IO.puts("  CUDA: ✗ Disabled (CPU fallback)")
end

IO.puts("")

# Benchmark helper
benchmark = fn name, impl_fn, seq_len ->
  key = Nx.Random.key(42)
  {query, key} = Nx.Random.uniform(key, shape: {batch_size, seq_len, hidden_dim}, type: :f32)
  {key_tensor, key} = Nx.Random.uniform(key, shape: {batch_size, seq_len, hidden_dim}, type: :f32)
  {value, _key} = Nx.Random.uniform(key, shape: {batch_size, seq_len, hidden_dim}, type: :f32)

  # Warmup
  for _ <- 1..warmup_iterations do
    try do
      impl_fn.(query, key_tensor, value) |> Nx.backend_transfer(Nx.BinaryBackend)
    rescue
      _ -> :error
    end
  end

  # Benchmark
  start = System.monotonic_time(:microsecond)

  successful =
    Enum.reduce(1..num_iterations, 0, fn _, succ ->
      try do
        impl_fn.(query, key_tensor, value) |> Nx.backend_transfer(Nx.BinaryBackend)
        succ + 1
      rescue
        _ -> succ
      end
    end)

  elapsed = System.monotonic_time(:microsecond) - start
  errors = num_iterations - successful

  if successful > 0 do
    avg_time = elapsed / successful
    {name, avg_time, successful, errors}
  else
    {name, :error, 0, num_iterations}
  end
end

# Run benchmarks for each sequence length
for seq_len <- seq_lens do
  IO.puts("=" |> String.duplicate(90))
  IO.puts("Sequence Length: #{seq_len}")
  IO.puts("=" |> String.duplicate(90))
  IO.puts("")
  IO.puts(String.pad_trailing("Implementation", 35) <> " | " <>
          String.pad_trailing("Avg Time", 15) <> " | " <>
          String.pad_trailing("Success", 10) <> " | " <>
          "Relative")
  IO.puts("-" |> String.duplicate(90))

  # Standard attention
  standard_result = benchmark.("Standard (O(n²))", fn q, k, v ->
    mask = Attention.causal_mask(seq_len)
    Attention.scaled_dot_product_attention(q, k, v, mask: mask)
  end, seq_len)

  # Chunked attention
  chunked_result = benchmark.("Chunked (chunk=#{chunk_size})", fn q, k, v ->
    mask = Attention.causal_mask(seq_len)
    Attention.chunked_attention(q, k, v, mask: mask, chunk_size: chunk_size)
  end, seq_len)

  # Memory-efficient attention
  mem_eff_result = benchmark.("Memory-Efficient (O(n))", fn q, k, v ->
    Attention.memory_efficient_attention(q, k, v, chunk_size: chunk_size, causal: true)
  end, seq_len)

  # NIF FlashAttention (if available)
  nif_result =
    if nif_available do
      benchmark.("NIF FlashAttention", fn q, k, v ->
        # NIF expects [batch, seq, num_heads, head_dim]
        # We have [batch, seq, hidden_dim] where hidden_dim = num_heads * head_dim
        # Need to reshape for NIF
        q_reshaped = Nx.reshape(q, {batch_size, seq_len, num_heads, head_dim})
        k_reshaped = Nx.reshape(k, {batch_size, seq_len, num_heads, head_dim})
        v_reshaped = Nx.reshape(v, {batch_size, seq_len, num_heads, head_dim})

        {:ok, output} = ExPhil.Native.FlashAttention.forward(q_reshaped, k_reshaped, v_reshaped, causal: true)
        # Reshape back to [batch, seq, hidden_dim]
        Nx.reshape(output, {batch_size, seq_len, hidden_dim})
      end, seq_len)
    else
      {"NIF FlashAttention", :unavailable, 0, 0}
    end

  results = [standard_result, chunked_result, mem_eff_result, nif_result]

  # Find baseline (standard attention time) for relative comparison
  baseline_time =
    case standard_result do
      {_, time, _, _} when is_number(time) -> time
      _ -> 1.0
    end

  # Print results
  for {name, time, successful, _errors} <- results do
    time_str =
      case time do
        :error -> "ERROR"
        :unavailable -> "N/A"
        t when is_number(t) -> "#{Float.round(t, 1)}μs"
      end

    success_str = "#{successful}/#{num_iterations}"

    relative_str =
      case time do
        t when is_number(t) and baseline_time > 0 ->
          ratio = t / baseline_time
          if ratio < 1.0 do
            "#{Float.round(1.0 / ratio, 2)}x faster"
          else
            "#{Float.round(ratio, 2)}x slower"
          end
        _ -> "-"
      end

    IO.puts(
      String.pad_trailing(name, 35) <> " | " <>
      String.pad_trailing(time_str, 15) <> " | " <>
      String.pad_trailing(success_str, 10) <> " | " <>
      relative_str
    )
  end

  IO.puts("")
end

# Summary
IO.puts("=" |> String.duplicate(90))
IO.puts("SUMMARY & RECOMMENDATIONS")
IO.puts("=" |> String.duplicate(90))
IO.puts("")

IO.puts("""
Memory Complexity:
  • Standard:         O(n²) - Full attention matrix in memory
  • Chunked:          O(n × chunk) - Lower peak, same total
  • Memory-Efficient: O(n) - Online softmax, true linear memory
  • NIF FlashAttention: O(n) - CUDA kernel with tiled computation

Speed Characteristics (CPU):
  • Standard:         Fastest - optimized XLA kernels
  • Chunked:          ~2x slower - more kernel launches
  • Memory-Efficient: ~2-3x slower - online softmax overhead
  • NIF (CPU):        Slowest - data copy overhead Elixir↔Rust

Speed Characteristics (GPU with CUDA):
  • NIF FlashAttention: 10-100x faster than CPU implementations
  • Avoids materializing O(n²) attention matrix in VRAM
  • Designed for real-time inference at 60 FPS

Recommendations:
  • Training (GPU):    Use standard attention (XLA handles memory)
  • Training (memory): Use memory-efficient for long sequences
  • Inference (GPU):   Use NIF FlashAttention (lowest latency)
  • Inference (CPU):   Use standard attention (NIF has copy overhead)

When to use each:
  • Standard:         Default choice, fastest on CPU, handles seq ≤ 256 well
  • Chunked:          When hitting OOM with standard attention
  • Memory-Efficient: Long sequences (512+) where memory is critical
  • NIF:              Real-time inference with Ampere+ GPU
""")

IO.puts("✅ Benchmark complete!")
