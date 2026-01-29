#!/usr/bin/env elixir
# Benchmark attention implementations for choosing effective defaults
#
# Usage:
#   mix run scripts/benchmark_attention.exs [--seq-lens 32,64,128,256] [--iterations 50] [--quiet]
#
# Options:
#   --quiet       Suppress warnings and info logs (only show errors)
#   --seq-lens    Comma-separated sequence lengths to test
#   --iterations  Number of benchmark iterations per implementation
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

# Parse CLI arguments
args = System.argv()

# Quiet mode - suppress warnings
if "--quiet" in args do
  Logger.configure(level: :error)
end

Output.banner("Attention Implementation Benchmark")

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

# =============================================================================
# Memory Calculation Helpers
# =============================================================================

# Calculate theoretical peak memory for attention implementations
# Returns bytes needed for the largest intermediate tensor

defmodule MemoryCalc do
  @doc """
  Standard attention peak memory: full attention matrix [batch, seq, seq]
  Plus Q, K, V, and output tensors
  """
  def standard(batch, seq_len, hidden_dim, dtype_bytes \\ 4) do
    # Attention scores: [batch, seq, seq]
    attention_matrix = batch * seq_len * seq_len * dtype_bytes
    # Q, K, V, Output: each [batch, seq, hidden_dim]
    tensors = 4 * batch * seq_len * hidden_dim * dtype_bytes
    attention_matrix + tensors
  end

  @doc """
  Chunked attention peak memory: chunked attention matrix [batch, chunk, seq]
  Still processes full K/V but only chunk of Q at a time
  """
  def chunked(batch, seq_len, hidden_dim, chunk_size, dtype_bytes \\ 4) do
    # Chunked attention scores: [batch, chunk_size, seq]
    attention_chunk = batch * chunk_size * seq_len * dtype_bytes
    # Q, K, V, Output: each [batch, seq, hidden_dim]
    tensors = 4 * batch * seq_len * hidden_dim * dtype_bytes
    attention_chunk + tensors
  end

  @doc """
  Memory-efficient attention peak memory: only [batch, seq, chunk] scores
  Plus running accumulators for online softmax
  """
  def memory_efficient(batch, seq_len, hidden_dim, chunk_size, dtype_bytes \\ 4) do
    # Chunk scores: [batch, seq_q, chunk_size]
    chunk_scores = batch * seq_len * chunk_size * dtype_bytes
    # Accumulators: max [batch, seq], sum [batch, seq], output [batch, seq, hidden]
    acc_max = batch * seq_len * dtype_bytes
    acc_sum = batch * seq_len * dtype_bytes
    acc_output = batch * seq_len * hidden_dim * dtype_bytes
    # Q, K, V (but K/V accessed in chunks)
    tensors = 3 * batch * seq_len * hidden_dim * dtype_bytes
    chunk_scores + acc_max + acc_sum + acc_output + tensors
  end

  @doc """
  NIF FlashAttention: tiled computation, never materializes full matrix
  Similar to memory-efficient but more optimized
  """
  def nif_flash(batch, seq_len, num_heads, head_dim, tile_size \\ 32, dtype_bytes \\ 4) do
    hidden_dim = num_heads * head_dim
    # Per-tile shared memory (K and V tiles)
    tile_kv = 2 * tile_size * head_dim * dtype_bytes
    # Per-query accumulators (in registers, but count them)
    query_acc = head_dim * dtype_bytes * 2  # output + running sum
    # Q, K, V, Output tensors
    tensors = 4 * batch * seq_len * hidden_dim * dtype_bytes
    # Multiply tile memory by number of concurrent tiles (batch * heads)
    tile_kv * batch * num_heads + query_acc * batch * num_heads + tensors
  end

  @doc """
  Format bytes as human-readable string
  """
  def format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  def format_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"
  def format_bytes(bytes) when bytes < 1024 * 1024 * 1024, do: "#{Float.round(bytes / (1024 * 1024), 1)} MB"
  def format_bytes(bytes), do: "#{Float.round(bytes / (1024 * 1024 * 1024), 2)} GB"
end

# =============================================================================
# Benchmark Helper
# =============================================================================

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

# =============================================================================
# Memory Analysis Section
# =============================================================================

IO.puts("=" |> String.duplicate(100))
IO.puts("THEORETICAL PEAK MEMORY BY SEQUENCE LENGTH")
IO.puts("=" |> String.duplicate(100))
IO.puts("")
IO.puts(
  String.pad_trailing("Seq Len", 10) <> " | " <>
  String.pad_trailing("Standard", 12) <> " | " <>
  String.pad_trailing("Chunked", 12) <> " | " <>
  String.pad_trailing("Mem-Eff", 12) <> " | " <>
  String.pad_trailing("NIF Flash", 12) <> " | " <>
  "Savings"
)
IO.puts("-" |> String.duplicate(100))

for seq_len <- seq_lens do
  standard_mem = MemoryCalc.standard(batch_size, seq_len, hidden_dim)
  chunked_mem = MemoryCalc.chunked(batch_size, seq_len, hidden_dim, chunk_size)
  mem_eff_mem = MemoryCalc.memory_efficient(batch_size, seq_len, hidden_dim, chunk_size)
  nif_mem = MemoryCalc.nif_flash(batch_size, seq_len, num_heads, head_dim)

  savings = Float.round(standard_mem / mem_eff_mem, 1)

  IO.puts(
    String.pad_trailing("#{seq_len}", 10) <> " | " <>
    String.pad_trailing(MemoryCalc.format_bytes(standard_mem), 12) <> " | " <>
    String.pad_trailing(MemoryCalc.format_bytes(chunked_mem), 12) <> " | " <>
    String.pad_trailing(MemoryCalc.format_bytes(mem_eff_mem), 12) <> " | " <>
    String.pad_trailing(MemoryCalc.format_bytes(nif_mem), 12) <> " | " <>
    "#{savings}x less"
  )
end

IO.puts("")
IO.puts("Note: Standard attention memory grows O(n²) with sequence length.")
IO.puts("      Memory-efficient grows O(n) - critical for long sequences.")
IO.puts("")

# =============================================================================
# Timing Benchmarks
# =============================================================================

for seq_len <- seq_lens do
  IO.puts("=" |> String.duplicate(100))
  IO.puts("Sequence Length: #{seq_len}")
  IO.puts("=" |> String.duplicate(100))
  IO.puts("")

  # Calculate memory for this seq_len
  standard_mem = MemoryCalc.standard(batch_size, seq_len, hidden_dim)
  chunked_mem = MemoryCalc.chunked(batch_size, seq_len, hidden_dim, chunk_size)
  mem_eff_mem = MemoryCalc.memory_efficient(batch_size, seq_len, hidden_dim, chunk_size)
  nif_mem = MemoryCalc.nif_flash(batch_size, seq_len, num_heads, head_dim)

  IO.puts(
    String.pad_trailing("Implementation", 30) <> " | " <>
    String.pad_trailing("Avg Time", 12) <> " | " <>
    String.pad_trailing("Peak Memory", 12) <> " | " <>
    String.pad_trailing("Success", 8) <> " | " <>
    "Relative"
  )
  IO.puts("-" |> String.duplicate(100))

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
        q_reshaped = Nx.reshape(q, {batch_size, seq_len, num_heads, head_dim})
        k_reshaped = Nx.reshape(k, {batch_size, seq_len, num_heads, head_dim})
        v_reshaped = Nx.reshape(v, {batch_size, seq_len, num_heads, head_dim})

        {:ok, output} = ExPhil.Native.FlashAttention.forward(q_reshaped, k_reshaped, v_reshaped, causal: true)
        Nx.reshape(output, {batch_size, seq_len, hidden_dim})
      end, seq_len)
    else
      {"NIF FlashAttention", :unavailable, 0, 0}
    end

  results = [
    {standard_result, standard_mem},
    {chunked_result, chunked_mem},
    {mem_eff_result, mem_eff_mem},
    {nif_result, nif_mem}
  ]

  # Find baseline (standard attention time) for relative comparison
  baseline_time =
    case standard_result do
      {_, time, _, _} when is_number(time) -> time
      _ -> 1.0
    end

  # Print results
  for {{name, time, successful, _errors}, memory} <- results do
    time_str =
      case time do
        :error -> "ERROR"
        :unavailable -> "N/A"
        t when is_number(t) -> "#{Float.round(t, 1)}μs"
      end

    mem_str = MemoryCalc.format_bytes(memory)
    success_str = "#{successful}/#{num_iterations}"

    relative_str =
      case time do
        t when is_number(t) and baseline_time > 0 ->
          ratio = t / baseline_time
          if ratio < 1.0 do
            "#{Float.round(1.0 / ratio, 2)}x faster"
          else
            "#{Float.round(ratio, 2)}x"
          end
        _ -> "-"
      end

    IO.puts(
      String.pad_trailing(name, 30) <> " | " <>
      String.pad_trailing(time_str, 12) <> " | " <>
      String.pad_trailing(mem_str, 12) <> " | " <>
      String.pad_trailing(success_str, 8) <> " | " <>
      relative_str
    )
  end

  IO.puts("")
end

# =============================================================================
# Summary
# =============================================================================

IO.puts("=" |> String.duplicate(100))
IO.puts("SUMMARY & RECOMMENDATIONS")
IO.puts("=" |> String.duplicate(100))
IO.puts("")

IO.puts("""
Memory Complexity (theoretical peak for attention computation):
  • Standard:         O(batch × seq²) - Full attention matrix
  • Chunked:          O(batch × chunk × seq) - Chunked Q processing
  • Memory-Efficient: O(batch × seq × chunk) - Online softmax, never full matrix
  • NIF FlashAttention: O(batch × heads × tile²) - Tiled GPU computation

Speed vs Memory Trade-off (CPU):
  • Standard:         Fastest, highest memory
  • Chunked:          ~2x slower, ~seq/chunk memory reduction
  • Memory-Efficient: ~2-3x slower, best memory efficiency
  • NIF (CPU):        Slower due to data copy overhead

When Memory Matters:
  • seq=64:   Standard uses ~#{MemoryCalc.format_bytes(MemoryCalc.standard(batch_size, 64, hidden_dim))} peak
  • seq=256:  Standard uses ~#{MemoryCalc.format_bytes(MemoryCalc.standard(batch_size, 256, hidden_dim))} peak
  • seq=1024: Standard uses ~#{MemoryCalc.format_bytes(MemoryCalc.standard(batch_size, 1024, hidden_dim))} peak
  • seq=4096: Standard uses ~#{MemoryCalc.format_bytes(MemoryCalc.standard(batch_size, 4096, hidden_dim))} peak

Recommendations:
  • seq ≤ 128:  Use standard attention (memory is fine)
  • seq 128-512: Consider memory-efficient if memory-constrained
  • seq > 512:  Use memory-efficient or NIF with CUDA
  • Real-time:  NIF with CUDA for lowest latency (when available)
""")

IO.puts("✅ Benchmark complete!")
