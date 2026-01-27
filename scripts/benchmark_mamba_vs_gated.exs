#!/usr/bin/env elixir
# Benchmark: Mamba variants inference speed
#
# Run:  mix run scripts/benchmark_mamba_vs_gated.exs
#
# Compares:
# - GatedSSM: Simplified gated approximation (fastest)
# - Mamba: True parallel scan with Blelloch algorithm
# - MambaCumsum: Cumsum-based optimization

alias ExPhil.Networks.{Mamba, GatedSSM, MambaCumsum}
alias ExPhil.Training.Output

Output.banner("Mamba Variants Inference Benchmark")

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
  {"Sequence length", seq_len},
  {"Batch size", batch_size},
  {"Iterations", iterations}
])

# Check backend
backend = Nx.default_backend()
Output.puts("\nBackend: #{inspect(backend)}")
if match?({EXLA.Backend, _}, backend) do
  Output.success("GPU acceleration available")
else
  Output.warning("CPU only - results will be slow")
end

# Models to benchmark
models = [
  {:gated_ssm, "GatedSSM", fn opts -> GatedSSM.build(opts) end},
  {:mamba, "Mamba (Blelloch)", fn opts -> Mamba.build(opts) end},
  {:mamba_cumsum, "MambaCumsum", fn opts -> MambaCumsum.build(opts) end}
]

model_opts = [
  embed_size: embed_size,
  hidden_size: hidden_size,
  state_size: state_size,
  num_layers: 2,
  window_size: seq_len
]

# Create input
Output.puts("\nCreating input tensor...")
key = Nx.Random.key(42)
{input, _} = Nx.Random.uniform(key, shape: {batch_size, seq_len, embed_size}, type: :f32)

# Benchmark each model
results =
  for {id, name, builder} <- models do
    Output.puts("\n" <> String.duplicate("-", 50))
    Output.puts("#{name}")
    Output.puts(String.duplicate("-", 50))

    try do
      Output.puts("  Building...")
      model = builder.(model_opts)
      {init_fn, pred_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({batch_size, seq_len, embed_size}, :f32), Axon.ModelState.empty())

      Output.puts("  Warmup (#{warmup} iterations)...")
      for _ <- 1..warmup, do: pred_fn.(params, input)

      Output.puts("  Timing (#{iterations} iterations)...")
      times = for _ <- 1..iterations do
        {time_us, _} = :timer.tc(fn ->
          result = pred_fn.(params, input)
          if match?({EXLA.Backend, _}, Nx.default_backend()) do
            Nx.backend_transfer(result, Nx.BinaryBackend)
          else
            result
          end
        end)
        time_us / 1000
      end

      mean = Enum.sum(times) / length(times)
      std = :math.sqrt(Enum.sum(Enum.map(times, &((&1 - mean) * (&1 - mean)))) / length(times))

      Output.puts("  Result: #{Float.round(mean, 2)} ± #{Float.round(std, 2)} ms")

      {id, name, mean, std, :ok}
    rescue
      e ->
        Output.error("  FAILED: #{Exception.message(e)}")
        {id, name, 0.0, 0.0, :error}
    end
  end

# Summary
Output.puts("\n" <> String.duplicate("=", 60))
Output.puts("SUMMARY")
Output.puts(String.duplicate("=", 60))

# Find baseline (GatedSSM)
{_, _, baseline_ms, _, _} = Enum.find(results, fn {id, _, _, _, _} -> id == :gated_ssm end)

Output.puts("\n| Model | Time (ms) | vs GatedSSM | 60 FPS? |")
Output.puts("|-------|-----------|-------------|---------|")

for {_id, name, mean, _std, status} <- results do
  if status == :ok and mean > 0 do
    ratio = if baseline_ms > 0, do: Float.round(mean / baseline_ms, 2), else: 0.0
    fps_ok = if mean < 16.67, do: "YES", else: "NO"
    Output.puts("| #{String.pad_trailing(name, 16)} | #{String.pad_leading(Float.to_string(Float.round(mean, 1)), 9)} | #{String.pad_leading("#{ratio}x", 11)} | #{String.pad_leading(fps_ok, 7)} |")
  else
    Output.puts("| #{String.pad_trailing(name, 16)} | FAILED |")
  end
end

Output.puts("\n60 FPS threshold: 16.67ms")
Output.puts("")
Output.success("Benchmark complete!")
