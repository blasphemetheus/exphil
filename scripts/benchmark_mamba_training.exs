#!/usr/bin/env elixir
# Benchmark: Mamba variants TRAINING speed
#
# Run on GPU pod:
#   mix run scripts/benchmark_mamba_training.exs
#
# Compares training speed (batches/sec including backward pass) for:
# - GatedSSM (simplified gated approximation)
# - Mamba (true parallel scan)
# - MambaCumsum (cumsum-based optimization)
#
# Also tests longer sequences (L=60, L=120, L=180) to show scaling

alias ExPhil.Training.{Imitation, Output}

Output.banner("Mamba Training Speed Benchmark")

# Configuration
embed_size = 287
hidden_size = 256
state_size = 16
batch_size = 32
warmup = 2
iterations = 5

# Test different sequence lengths
seq_lengths = [60, 120, 180]

Output.config([
  {"Embed size", embed_size},
  {"Hidden size", hidden_size},
  {"State size", state_size},
  {"Batch size", batch_size},
  {"Sequence lengths", Enum.join(seq_lengths, ", ")}
])

# Check for GPU
backend = Nx.default_backend()
Output.puts("\nBackend: #{inspect(backend)}")

if match?({EXLA.Backend, _}, backend) do
  Output.success("EXLA backend detected - GPU acceleration available")
else
  Output.warning("Not using EXLA - results will be CPU-only")
end

# Architectures to benchmark
architectures = [
  {:gated_ssm, "GatedSSM (simplified)"},
  {:mamba, "Mamba (parallel scan)"},
  {:mamba_cumsum, "MambaCumsum (cumsum)"}
]

# Create fake training batch
create_batch = fn seq_len ->
  # States: [batch, seq_len, embed_size]
  key = Nx.Random.key(42)
  {states, key} = Nx.Random.uniform(key, shape: {batch_size, seq_len, embed_size}, type: :f32)

  # Labels (fake)
  {buttons, key} = Nx.Random.randint(key, 0, 2, shape: {batch_size, 8}, type: :u8)
  {main_x, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size}, type: :u8)
  {main_y, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size}, type: :u8)
  {c_x, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size}, type: :u8)
  {c_y, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size}, type: :u8)
  {shoulder, _key} = Nx.Random.randint(key, 0, 5, shape: {batch_size}, type: :u8)

  %{
    states: states,
    buttons: buttons,
    main_x: main_x,
    main_y: main_y,
    c_x: c_x,
    c_y: c_y,
    shoulder: shoulder
  }
end

# Benchmark function for training speed
benchmark_training = fn backbone, seq_len ->
  Output.puts("    Building trainer...")

  trainer =
    Imitation.new(
      embed_size: embed_size,
      hidden_sizes: [hidden_size, hidden_size],
      temporal: true,
      backbone: backbone,
      window_size: seq_len,
      hidden_size: hidden_size,
      state_size: state_size,
      num_layers: 2
    )

  batch = create_batch.(seq_len)

  Output.puts("    Warmup (#{warmup} steps)...")
  # Warmup - includes JIT compilation
  trainer =
    Enum.reduce(1..warmup, trainer, fn _, t ->
      {new_trainer, _metrics} = Imitation.train_step(t, batch, nil)
      new_trainer
    end)

  Output.puts("    Timing (#{iterations} steps)...")
  # Timed runs
  {total_time_us, _} = :timer.tc(fn ->
    Enum.reduce(1..iterations, trainer, fn _, t ->
      {new_trainer, _metrics} = Imitation.train_step(t, batch, nil)

      # Force GPU sync
      if match?({EXLA.Backend, _}, Nx.default_backend()) do
        # Access a value to force computation
        _ = Nx.to_number(Nx.sum(new_trainer.policy_params["input_projection"]["kernel"]))
      end

      new_trainer
    end)
  end)

  avg_time_ms = total_time_us / iterations / 1000
  batches_per_sec = 1000 / avg_time_ms
  samples_per_sec = batches_per_sec * batch_size

  %{
    avg_time_ms: avg_time_ms,
    batches_per_sec: batches_per_sec,
    samples_per_sec: samples_per_sec
  }
end

# Run benchmarks
results = %{}

for {backbone, name} <- architectures do
  Output.puts("\n" <> String.duplicate("=", 60))
  Output.puts("#{name}")
  Output.puts(String.duplicate("=", 60))

  for seq_len <- seq_lengths do
    Output.puts("\n  Sequence length: #{seq_len}")

    try do
      stats = benchmark_training.(backbone, seq_len)
      Output.puts("    Result: #{Float.round(stats.avg_time_ms, 1)}ms/batch, #{Float.round(stats.batches_per_sec, 2)} batch/s")
    rescue
      e ->
        Output.error("    FAILED: #{Exception.message(e)}")
    end
  end
end

# Summary table
Output.puts("\n" <> String.duplicate("=", 60))
Output.puts("SUMMARY: Training Speed (batches/sec)")
Output.puts(String.duplicate("=", 60))
Output.puts("\n| Architecture | L=60 | L=120 | L=180 | Scaling |")
Output.puts("|--------------|------|-------|-------|---------|")

# Re-run to collect data for table
for {backbone, name} <- architectures do
  results_row =
    for seq_len <- seq_lengths do
      try do
        stats = benchmark_training.(backbone, seq_len)
        Float.round(stats.batches_per_sec, 2)
      rescue
        _ -> 0.0
      end
    end

  [l60, l120, l180] = results_row
  scaling = if l60 > 0 and l180 > 0, do: Float.round(l60 / l180, 2), else: 0.0

  Output.puts("| #{String.pad_trailing(name, 12)} | #{l60} | #{l120} | #{l180} | #{scaling}x |")
end

Output.puts("")
Output.puts("Lower scaling factor = better (closer to O(log L) vs O(L))")
Output.puts("Ideal Mamba scaling for L=60→180: ~1.3x (log₂(180)/log₂(60) ≈ 1.27)")
Output.puts("Sequential scaling for L=60→180: 3.0x")

Output.puts("")
Output.success("Benchmark complete!")
