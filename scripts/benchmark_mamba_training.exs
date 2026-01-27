#!/usr/bin/env elixir
# Benchmark: Mamba variants TRAINING speed
#
# Run on GPU pod:
#   mix run scripts/benchmark_mamba_training.exs
#   mix run scripts/benchmark_mamba_training.exs --only mamba_hs
#   mix run scripts/benchmark_mamba_training.exs --only gated_ssm,mamba --seq 60,120
#
# Available IDs: gated_ssm, mamba, mamba_cumsum, mamba_hillis_steele, mamba_ssd

alias ExPhil.Training.{Imitation, Output}

# Parse flags
{opts, _, _} = OptionParser.parse(System.argv(), strict: [only: :string, seq: :string])
only_filter = case opts[:only] do
  nil -> nil
  str -> str |> String.split(",") |> Enum.map(&String.trim/1) |> MapSet.new()
end
seq_filter = case opts[:seq] do
  nil -> nil
  str -> str |> String.split(",") |> Enum.map(&String.trim/1) |> Enum.map(&String.to_integer/1)
end

Output.banner("Mamba Training Speed Benchmark")

# Configuration
embed_size = 287
hidden_size = 256
state_size = 16
batch_size = 32
warmup = 2
iterations = 5

# Test different sequence lengths
seq_lengths = seq_filter || [60, 120, 180]

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

# All available architectures
all_architectures = [
  {:gated_ssm, "GatedSSM (simplified)"},
  {:mamba, "Mamba (Blelloch)"},
  {:mamba_cumsum, "MambaCumsum (Blelloch)"},
  {:mamba_hillis_steele, "Mamba (Hillis-Steele)"},
  {:mamba_ssd, "Mamba (SSD)"}
]

# Filter architectures if --only specified
architectures = case only_filter do
  nil -> all_architectures
  filter ->
    filtered = Enum.filter(all_architectures, fn {id, _} -> MapSet.member?(filter, Atom.to_string(id)) end)
    if filtered == [] do
      Output.error("No architectures matched filter: #{Enum.join(filter, ", ")}")
      Output.puts("Available IDs: #{Enum.map_join(all_architectures, ", ", fn {id, _} -> id end)}")
      System.halt(1)
    end
    filtered
end

Output.puts("Testing #{length(architectures)} architecture(s): #{Enum.map_join(architectures, ", ", fn {id, _} -> id end)}")

# Create fake training batch
create_batch = fn seq_len ->
  # States: [batch, seq_len, embed_size]
  key = Nx.Random.key(42)
  {states, key} = Nx.Random.uniform(key, shape: {batch_size, seq_len, embed_size}, type: :f32)

  # Labels (fake) - must be nested under :actions key
  {buttons, key} = Nx.Random.randint(key, 0, 2, shape: {batch_size, 8}, type: :u8)
  {main_x, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size}, type: :u8)
  {main_y, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size}, type: :u8)
  {c_x, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size}, type: :u8)
  {c_y, key} = Nx.Random.randint(key, 0, 17, shape: {batch_size}, type: :u8)
  {shoulder, _key} = Nx.Random.randint(key, 0, 5, shape: {batch_size}, type: :u8)

  %{
    states: states,
    actions: %{
      buttons: buttons,
      main_x: main_x,
      main_y: main_y,
      c_x: c_x,
      c_y: c_y,
      shoulder: shoulder
    }
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

      # Force GPU sync by accessing a parameter value
      if match?({EXLA.Backend, _}, Nx.default_backend()) do
        # Extract params data from ModelState struct
        %Axon.ModelState{data: params_data} = new_trainer.policy_params
        {_name, first_param} = params_data |> Map.to_list() |> List.first()
        {_key, tensor} = first_param |> Map.to_list() |> List.first()
        _ = Nx.to_number(Nx.sum(tensor))
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

# Build dynamic header based on seq_lengths
header_cols = Enum.map(seq_lengths, fn l -> "L=#{l}" end) ++ ["Scaling"]
header = "| Architecture | " <> Enum.join(header_cols, " | ") <> " |"
separator = "|" <> String.duplicate("-", 14) <> "|" <> Enum.map_join(1..length(header_cols), "|", fn _ -> "------" end) <> "|"
Output.puts("\n" <> header)
Output.puts(separator)

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

  first = List.first(results_row) || 0.0
  last = List.last(results_row) || 0.0
  scaling = if first > 0 and last > 0 and length(results_row) > 1, do: Float.round(first / last, 2), else: 0.0

  row_data = Enum.map(results_row, &to_string/1) ++ ["#{scaling}x"]
  Output.puts("| #{String.pad_trailing(name, 12)} | " <> Enum.join(row_data, " | ") <> " |")
end

Output.puts("")
Output.puts("Lower scaling factor = better (closer to O(log L) vs O(L))")
Output.puts("Ideal Mamba scaling for L=60→180: ~1.3x (log₂(180)/log₂(60) ≈ 1.27)")
Output.puts("Sequential scaling for L=60→180: 3.0x")

Output.puts("")
Output.success("Benchmark complete!")
