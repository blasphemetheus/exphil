#!/usr/bin/env elixir
# Benchmark: True Mamba vs GatedSSM
#
# Run on GPU pod:
#   mix run scripts/benchmark_mamba_vs_gated.exs
#
# This compares the true Mamba (with parallel associative scan) against
# the simplified GatedSSM to measure the impact of the parallel scan on GPU.

alias ExPhil.Networks.{Mamba, GatedSSM}
alias ExPhil.Training.Output

Output.banner("Mamba vs GatedSSM Benchmark")

# Configuration
embed_size = 287
hidden_size = 256
state_size = 16
seq_len = 60
batch_size = 32
warmup = 5
iterations = 20

Output.config([
  {"Embed size", embed_size},
  {"Hidden size", hidden_size},
  {"State size", state_size},
  {"Sequence length", seq_len},
  {"Batch size", batch_size},
  {"Warmup iterations", warmup},
  {"Timed iterations", iterations}
])

# Check for GPU
backend = Nx.default_backend()
Output.puts("\nBackend: #{inspect(backend)}")

if match?({EXLA.Backend, _}, backend) do
  Output.success("EXLA backend detected - GPU acceleration available")
else
  Output.warning("Not using EXLA - results will be CPU-only")
end

# Helper to format numbers with commas
format_number = fn n ->
  n
  |> round()
  |> Integer.to_string()
  |> String.graphemes()
  |> Enum.reverse()
  |> Enum.chunk_every(3)
  |> Enum.join(",")
  |> String.reverse()
end

# Benchmark function
benchmark = fn name, pred_fn, params, input ->
  Output.puts("\nBenchmarking #{name}...")

  # Warmup (with progress)
  for i <- 1..warmup do
    Output.progress_bar(i, warmup, label: "  Warmup")
    pred_fn.(params, input)
  end
  Output.progress_done()

  # Timed runs
  times = for i <- 1..iterations do
    Output.progress_bar(i, iterations, label: "  Timing")
    {time_us, _} = :timer.tc(fn ->
      result = pred_fn.(params, input)
      # Force synchronization for GPU
      if match?({EXLA.Backend, _}, Nx.default_backend()) do
        Nx.backend_transfer(result, Nx.BinaryBackend)
      else
        result
      end
    end)
    time_us / 1000  # Convert to ms
  end
  Output.progress_done()

  mean = Enum.sum(times) / length(times)
  min_t = Enum.min(times)
  max_t = Enum.max(times)
  std = :math.sqrt(Enum.sum(Enum.map(times, fn t -> (t - mean) * (t - mean) end)) / length(times))

  %{name: name, mean: mean, min: min_t, max: max_t, std: std}
end

# Build models
Output.step(1, 4, "Building GatedSSM model")
gated_model = GatedSSM.build(
  embed_size: embed_size,
  hidden_size: hidden_size,
  state_size: state_size,
  num_layers: 2,
  window_size: seq_len
)
{gated_init, gated_pred} = Axon.build(gated_model, mode: :inference)
gated_params = gated_init.(Nx.template({batch_size, seq_len, embed_size}, :f32), Axon.ModelState.empty())

Output.step(2, 4, "Building Mamba model (true parallel scan)")
mamba_model = Mamba.build(
  embed_size: embed_size,
  hidden_size: hidden_size,
  state_size: state_size,
  num_layers: 2,
  window_size: seq_len
)
{mamba_init, mamba_pred} = Axon.build(mamba_model, mode: :inference)
mamba_params = mamba_init.(Nx.template({batch_size, seq_len, embed_size}, :f32), Axon.ModelState.empty())

# Create input
Output.step(3, 4, "Creating input tensor")
key = Nx.Random.key(42)
{input, _} = Nx.Random.uniform(key, shape: {batch_size, seq_len, embed_size}, type: :f32)

# Run benchmarks
Output.step(4, 4, "Running benchmarks")

gated_stats = benchmark.("GatedSSM", gated_pred, gated_params, input)
mamba_stats = benchmark.("Mamba (parallel scan)", mamba_pred, mamba_params, input)

# Results
Output.puts("\n" <> String.duplicate("=", 60))
Output.puts("RESULTS")
Output.puts(String.duplicate("=", 60))

print_stats = fn stats ->
  Output.puts("  #{stats.name}:")
  Output.puts("    Mean:   #{Float.round(stats.mean, 2)} ms")
  Output.puts("    Std:    #{Float.round(stats.std, 2)} ms")
  Output.puts("    Min:    #{Float.round(stats.min, 2)} ms")
  Output.puts("    Max:    #{Float.round(stats.max, 2)} ms")
end

print_stats.(gated_stats)
print_stats.(mamba_stats)

# Comparison
Output.puts("\n" <> String.duplicate("-", 60))
Output.puts("COMPARISON")
Output.puts(String.duplicate("-", 60))

ratio = mamba_stats.mean / gated_stats.mean
fps_gated = 1000 / gated_stats.mean
fps_mamba = 1000 / mamba_stats.mean

Output.puts("")
if ratio > 1 do
  Output.puts("  Mamba is #{Float.round(ratio, 2)}x SLOWER than GatedSSM")
else
  Output.puts("  Mamba is #{Float.round(1/ratio, 2)}x FASTER than GatedSSM")
end

Output.puts("")
Output.puts("  Throughput (batch=#{batch_size}):")
Output.puts("    GatedSSM: #{Float.round(fps_gated, 1)} batches/sec (#{Float.round(fps_gated * batch_size, 0)} samples/sec)")
Output.puts("    Mamba:    #{Float.round(fps_mamba, 1)} batches/sec (#{Float.round(fps_mamba * batch_size, 0)} samples/sec)")

Output.puts("")
Output.puts("  Real-time capability (60 FPS = 16.67ms max):")
if gated_stats.mean < 16.67 do
  Output.success("    GatedSSM: YES (#{Float.round(gated_stats.mean, 2)}ms < 16.67ms)")
else
  Output.warning("    GatedSSM: NO (#{Float.round(gated_stats.mean, 2)}ms > 16.67ms)")
end
if mamba_stats.mean < 16.67 do
  Output.success("    Mamba:    YES (#{Float.round(mamba_stats.mean, 2)}ms < 16.67ms)")
else
  Output.warning("    Mamba:    NO (#{Float.round(mamba_stats.mean, 2)}ms > 16.67ms)")
end

# Parameter counts
Output.puts("")
Output.puts("  Parameter counts:")
gated_count = GatedSSM.param_count(embed_size: embed_size, hidden_size: hidden_size, state_size: state_size, num_layers: 2)
mamba_count = Mamba.param_count(embed_size: embed_size, hidden_size: hidden_size, state_size: state_size, num_layers: 2)
Output.puts("    GatedSSM: #{format_number.(gated_count)}")
Output.puts("    Mamba:    #{format_number.(mamba_count)}")

Output.puts("")
Output.success("Benchmark complete!")
