#!/usr/bin/env elixir
# Benchmark fused operations vs unfused equivalents
#
# Usage:
#   mix run scripts/benchmark_fused_ops.exs
#
# This measures the speedup from kernel fusion for:
# - Dense + Activation (relu, silu, gelu)
# - LayerNorm + Activation
# - FFN blocks (dense → activation → dense)
# - Softmax (fused stable vs manual)
# - Gated Linear Units (SwiGLU, GeGLU)

alias ExPhil.Networks.FusedOps
alias ExPhil.Training.Output

require Output

Output.banner("Fused Operations Benchmark")

# Use GPU if available
Nx.default_backend(EXLA.Backend)

# Benchmark configuration
batch_size = 32
seq_len = 60
hidden_dim = 512
ffn_dim = 2048
num_iterations = 100
warmup_iterations = 10

IO.puts("\nConfiguration:")
IO.puts("  Batch size:    #{batch_size}")
IO.puts("  Sequence len:  #{seq_len}")
IO.puts("  Hidden dim:    #{hidden_dim}")
IO.puts("  FFN dim:       #{ffn_dim}")
IO.puts("  Iterations:    #{num_iterations}")
IO.puts("  Warmup:        #{warmup_iterations}")
IO.puts("")

# Helper to run benchmark
benchmark = fn name, fused_fn, unfused_fn ->
  # Warmup
  for _ <- 1..warmup_iterations do
    fused_fn.() |> Nx.backend_transfer(Nx.BinaryBackend)
    unfused_fn.() |> Nx.backend_transfer(Nx.BinaryBackend)
  end

  # Benchmark fused
  fused_start = System.monotonic_time(:microsecond)
  for _ <- 1..num_iterations do
    fused_fn.() |> Nx.backend_transfer(Nx.BinaryBackend)
  end
  fused_time = (System.monotonic_time(:microsecond) - fused_start) / num_iterations

  # Benchmark unfused
  unfused_start = System.monotonic_time(:microsecond)
  for _ <- 1..num_iterations do
    unfused_fn.() |> Nx.backend_transfer(Nx.BinaryBackend)
  end
  unfused_time = (System.monotonic_time(:microsecond) - unfused_start) / num_iterations

  speedup = unfused_time / fused_time

  IO.puts("#{String.pad_trailing(name, 30)} | Fused: #{Float.round(fused_time, 1)}μs | Unfused: #{Float.round(unfused_time, 1)}μs | Speedup: #{Float.round(speedup, 2)}x")

  {name, fused_time, unfused_time, speedup}
end

IO.puts("=" |> String.duplicate(85))
IO.puts("Operation                      | Fused Time      | Unfused Time    | Speedup")
IO.puts("=" |> String.duplicate(85))

results = []

# Generate random tensors using Nx.Random
key = Nx.Random.key(42)

# =============================================================================
# Dense + Activation benchmarks
# =============================================================================

{input_2d, key} = Nx.Random.uniform(key, shape: {batch_size, hidden_dim}, type: :f32)
{weight, key} = Nx.Random.uniform(key, shape: {hidden_dim, hidden_dim}, type: :f32)
{bias, key} = Nx.Random.uniform(key, shape: {hidden_dim}, type: :f32)

# Dense + ReLU
results = [benchmark.(
  "Dense + ReLU",
  fn -> FusedOps.dense_activation(input_2d, weight, bias, :relu) end,
  fn ->
    input_2d
    |> Nx.dot(weight)
    |> Nx.add(bias)
    |> Nx.max(0)
  end
) | results]

# Dense + SiLU
results = [benchmark.(
  "Dense + SiLU",
  fn -> FusedOps.dense_activation(input_2d, weight, bias, :silu) end,
  fn ->
    x = input_2d |> Nx.dot(weight) |> Nx.add(bias)
    Nx.multiply(x, Nx.sigmoid(x))
  end
) | results]

# Dense + GELU
results = [benchmark.(
  "Dense + GELU",
  fn -> FusedOps.dense_activation(input_2d, weight, bias, :gelu) end,
  fn ->
    x = input_2d |> Nx.dot(weight) |> Nx.add(bias)
    Nx.multiply(x, Nx.multiply(0.5, Nx.add(1.0, Nx.erf(Nx.multiply(x, 0.7071067811865476)))))
  end
) | results]

# =============================================================================
# LayerNorm + Activation benchmarks
# =============================================================================

{input_3d, key} = Nx.Random.uniform(key, shape: {batch_size, seq_len, hidden_dim}, type: :f32)
{gamma, key} = Nx.Random.uniform(key, shape: {hidden_dim}, type: :f32)
{beta, key} = Nx.Random.uniform(key, shape: {hidden_dim}, type: :f32)

# LayerNorm + ReLU
results = [benchmark.(
  "LayerNorm + ReLU",
  fn -> FusedOps.layernorm_activation(input_3d, gamma, beta, :relu) end,
  fn ->
    mean = Nx.mean(input_3d, axes: [-1], keep_axes: true)
    var = Nx.variance(input_3d, axes: [-1], keep_axes: true)
    normalized = Nx.divide(Nx.subtract(input_3d, mean), Nx.sqrt(Nx.add(var, 1.0e-5)))
    scaled = Nx.add(Nx.multiply(normalized, gamma), beta)
    Nx.max(scaled, 0)
  end
) | results]

# LayerNorm + SiLU
results = [benchmark.(
  "LayerNorm + SiLU",
  fn -> FusedOps.layernorm_activation(input_3d, gamma, beta, :silu) end,
  fn ->
    mean = Nx.mean(input_3d, axes: [-1], keep_axes: true)
    var = Nx.variance(input_3d, axes: [-1], keep_axes: true)
    normalized = Nx.divide(Nx.subtract(input_3d, mean), Nx.sqrt(Nx.add(var, 1.0e-5)))
    scaled = Nx.add(Nx.multiply(normalized, gamma), beta)
    Nx.multiply(scaled, Nx.sigmoid(scaled))
  end
) | results]

# =============================================================================
# FFN benchmarks
# =============================================================================

{w1, key} = Nx.Random.uniform(key, shape: {hidden_dim, ffn_dim}, type: :f32)
{b1, key} = Nx.Random.uniform(key, shape: {ffn_dim}, type: :f32)
{w2, key} = Nx.Random.uniform(key, shape: {ffn_dim, hidden_dim}, type: :f32)
{b2, key} = Nx.Random.uniform(key, shape: {hidden_dim}, type: :f32)

# FFN with GELU
results = [benchmark.(
  "FFN (Dense→GELU→Dense)",
  fn -> FusedOps.fused_ffn(input_2d, w1, b1, w2, b2, :gelu) end,
  fn ->
    x = input_2d |> Nx.dot(w1) |> Nx.add(b1)
    hidden = Nx.multiply(x, Nx.multiply(0.5, Nx.add(1.0, Nx.erf(Nx.multiply(x, 0.7071067811865476)))))
    hidden |> Nx.dot(w2) |> Nx.add(b2)
  end
) | results]

# FFN with SiLU
results = [benchmark.(
  "FFN (Dense→SiLU→Dense)",
  fn -> FusedOps.fused_ffn(input_2d, w1, b1, w2, b2, :silu) end,
  fn ->
    x = input_2d |> Nx.dot(w1) |> Nx.add(b1)
    hidden = Nx.multiply(x, Nx.sigmoid(x))
    hidden |> Nx.dot(w2) |> Nx.add(b2)
  end
) | results]

# =============================================================================
# Softmax benchmarks
# =============================================================================

{logits, key} = Nx.Random.uniform(key, shape: {batch_size, seq_len, hidden_dim}, type: :f32)

# Stable softmax
results = [benchmark.(
  "Softmax (stable)",
  fn -> FusedOps.fused_softmax(logits) end,
  fn ->
    max_logits = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_logits)
    exp_shifted = Nx.exp(shifted)
    Nx.divide(exp_shifted, Nx.sum(exp_shifted, axes: [-1], keep_axes: true))
  end
) | results]

# Log-softmax
results = [benchmark.(
  "Log-Softmax (stable)",
  fn -> FusedOps.fused_log_softmax(logits) end,
  fn ->
    max_logits = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_logits)
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
    Nx.subtract(shifted, log_sum_exp)
  end
) | results]

# =============================================================================
# Gated Linear Unit benchmarks
# =============================================================================

{w_gate, key} = Nx.Random.uniform(key, shape: {hidden_dim, ffn_dim}, type: :f32)
{w_up, _key} = Nx.Random.uniform(key, shape: {hidden_dim, ffn_dim}, type: :f32)

# SwiGLU
results = [benchmark.(
  "SwiGLU",
  fn -> FusedOps.swiglu(input_2d, w_gate, w_up) end,
  fn ->
    gate = Nx.dot(input_2d, w_gate)
    gate_activated = Nx.multiply(gate, Nx.sigmoid(gate))
    up = Nx.dot(input_2d, w_up)
    Nx.multiply(gate_activated, up)
  end
) | results]

# GeGLU
results = [benchmark.(
  "GeGLU",
  fn -> FusedOps.geglu(input_2d, w_gate, w_up) end,
  fn ->
    gate = Nx.dot(input_2d, w_gate)
    gate_activated = Nx.multiply(gate, Nx.multiply(0.5, Nx.add(1.0, Nx.erf(Nx.multiply(gate, 0.7071067811865476)))))
    up = Nx.dot(input_2d, w_up)
    Nx.multiply(gate_activated, up)
  end
) | results]

IO.puts("=" |> String.duplicate(85))

# Summary
results = Enum.reverse(results)
avg_speedup = results |> Enum.map(&elem(&1, 3)) |> Enum.sum() |> Kernel./(length(results))

IO.puts("\n📊 Summary:")
IO.puts("  Total operations benchmarked: #{length(results)}")
IO.puts("  Average speedup: #{Float.round(avg_speedup, 2)}x")

fastest = Enum.max_by(results, &elem(&1, 3))
slowest = Enum.min_by(results, &elem(&1, 3))

IO.puts("  Best speedup:  #{elem(fastest, 0)} (#{Float.round(elem(fastest, 3), 2)}x)")
IO.puts("  Least speedup: #{elem(slowest, 0)} (#{Float.round(elem(slowest, 3), 2)}x)")

IO.puts("\n✅ Benchmark complete!")
