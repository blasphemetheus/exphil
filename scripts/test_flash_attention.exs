#!/usr/bin/env elixir
# Test script for FlashAttention Python bridge
#
# Usage:
#   mix run scripts/test_flash_attention.exs
#
# Requirements:
#   pip install torch flash-attn --no-build-isolation msgpack

alias ExPhil.Bridge.FlashAttentionPort
alias ExPhil.Networks.Attention
alias ExPhil.Training.Output

Output.banner("FlashAttention Python Bridge Test")

# Start the server
IO.puts("\n1. Starting FlashAttention server...")
{:ok, _pid} = FlashAttentionPort.start_link()
Process.sleep(1500)  # Give it time to initialize

# Check server info
IO.puts("\n2. Checking server status...")
case FlashAttentionPort.info() do
  {:ok, info} ->
    IO.puts("   PyTorch version: #{info["pytorch_version"]}")
    IO.puts("   CUDA available: #{info["cuda_available"]}")
    IO.puts("   Device: #{info["device"]}")
    IO.puts("   flash_attn available: #{info["flash_attn_available"]}")

  {:error, reason} ->
    IO.puts("   ERROR: #{inspect(reason)}")
    System.halt(1)
end

# Test forward pass
IO.puts("\n3. Testing forward pass...")

batch = 2
seq_len = 64
dim = 256

# Create test tensors
key = Nx.Random.key(42)
{query, key} = Nx.Random.normal(key, shape: {batch, seq_len, dim})
{kv, key} = Nx.Random.normal(key, shape: {batch, seq_len, dim})
{value, _key} = Nx.Random.normal(key, shape: {batch, seq_len, dim})

IO.puts("   Input shapes: query=#{inspect(Nx.shape(query))}, key=#{inspect(Nx.shape(kv))}, value=#{inspect(Nx.shape(value))}")

# Run with Python bridge (flash or fallback)
case FlashAttentionPort.forward(query, kv, value, causal: true, use_flash: true) do
  {:ok, result} ->
    IO.puts("   Output shape: #{inspect(Nx.shape(result))}")
    IO.puts("   Output range: [#{Nx.to_number(Nx.reduce_min(result))}, #{Nx.to_number(Nx.reduce_max(result))}]")

  {:error, reason} ->
    IO.puts("   ERROR: #{inspect(reason)}")
end

# Compare with Pure Nx implementation
IO.puts("\n4. Comparing with Pure Nx implementations...")

# Standard attention
mask = Attention.causal_mask(seq_len)
nx_standard = Attention.scaled_dot_product_attention(query, kv, value, mask: mask)

# Memory-efficient attention
nx_mea = Attention.memory_efficient_attention(query, kv, value, causal: true, chunk_size: 32)

# Python bridge
{:ok, python_result} = FlashAttentionPort.forward(query, kv, value, causal: true, use_flash: false)

# Compare results
diff_mea_vs_std = Nx.abs(Nx.subtract(nx_mea, nx_standard)) |> Nx.reduce_max() |> Nx.to_number()
diff_py_vs_std = Nx.abs(Nx.subtract(python_result, nx_standard)) |> Nx.reduce_max() |> Nx.to_number()

IO.puts("   MEA vs Standard max diff: #{Float.round(diff_mea_vs_std, 6)}")
IO.puts("   Python vs Standard max diff: #{Float.round(diff_py_vs_std, 6)}")

# Run benchmark
IO.puts("\n5. Running benchmark...")
case FlashAttentionPort.benchmark(batch: 4, seq_len: 128, dim: 256, num_iters: 10) do
  {:ok, results} ->
    IO.puts("   Config: batch=#{results["config"]["batch"]}, seq_len=#{results["config"]["seq_len"]}, dim=#{results["config"]["dim"]}")
    IO.puts("   Standard attention: #{Float.round(results["standard_ms"], 2)} ms")

    if results["flash_ms"] do
      IO.puts("   Flash attention: #{Float.round(results["flash_ms"], 2)} ms")
      IO.puts("   Speedup: #{Float.round(results["speedup"], 2)}x")
    else
      IO.puts("   Flash attention: NOT AVAILABLE")
    end

  {:error, reason} ->
    IO.puts("   ERROR: #{inspect(reason)}")
end

# Quick latency comparison including Elixir overhead
IO.puts("\n6. End-to-end latency comparison (including serialization)...")

iterations = 5

# Warm up
FlashAttentionPort.forward(query, kv, value, causal: true)
Attention.scaled_dot_product_attention(query, kv, value, mask: mask)
Attention.memory_efficient_attention(query, kv, value, causal: true)

# Benchmark Pure Nx standard
{nx_time, _} = :timer.tc(fn ->
  for _ <- 1..iterations do
    Attention.scaled_dot_product_attention(query, kv, value, mask: mask)
  end
end)
nx_ms = nx_time / iterations / 1000

# Benchmark Pure Nx MEA
{mea_time, _} = :timer.tc(fn ->
  for _ <- 1..iterations do
    Attention.memory_efficient_attention(query, kv, value, causal: true)
  end
end)
mea_ms = mea_time / iterations / 1000

# Benchmark Python bridge
{py_time, _} = :timer.tc(fn ->
  for _ <- 1..iterations do
    FlashAttentionPort.forward(query, kv, value, causal: true)
  end
end)
py_ms = py_time / iterations / 1000

IO.puts("   Pure Nx Standard: #{Float.round(nx_ms, 2)} ms")
IO.puts("   Pure Nx MEA:      #{Float.round(mea_ms, 2)} ms")
IO.puts("   Python Bridge:    #{Float.round(py_ms, 2)} ms")

IO.puts("\n" <> String.duplicate("=", 60))
IO.puts("Test complete!")

if py_ms < nx_ms do
  IO.puts("Python bridge is #{Float.round(nx_ms / py_ms, 2)}x faster than Pure Nx")
else
  IO.puts("Pure Nx is #{Float.round(py_ms / nx_ms, 2)}x faster than Python bridge")
  IO.puts("(Serialization overhead exceeds computation savings at this scale)")
end
