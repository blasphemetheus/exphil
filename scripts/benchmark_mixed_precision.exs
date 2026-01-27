#!/usr/bin/env elixir
# Benchmark: BF16 precision vs Mixed Precision (FP32 master weights)
#
# Usage:
#   mix run scripts/benchmark_mixed_precision.exs
#
# Compares training speed and numerical stability between:
# 1. --precision bf16 (default) - All tensors in BF16
# 2. --mixed-precision - FP32 master weights + BF16 compute

alias ExPhil.Training.{Imitation, Data, Output}
alias ExPhil.Embeddings

Output.banner("Mixed Precision Benchmark")

# Use GPU
Nx.default_backend(EXLA.Backend)

# Configuration
num_frames = 5000
batch_size = 256
num_batches = 20
warmup_batches = 3
hidden_sizes = [512, 256]

IO.puts("\nConfiguration:")
IO.puts("  Frames: #{num_frames}")
IO.puts("  Batch size: #{batch_size}")
IO.puts("  Batches: #{num_batches}")
IO.puts("  Warmup: #{warmup_batches}")
IO.puts("  Hidden sizes: #{inspect(hidden_sizes)}")

# Create synthetic dataset
Output.puts("\nCreating synthetic dataset...")

# Generate synthetic game states
embed_config = Embeddings.config()

frames = for i <- 1..num_frames do
  # Create a minimal game state with random positions
  game_state = %ExPhil.Bridge.GameState{
    frame: i,
    stage: 31,  # Battlefield
    players: %{
      1 => %ExPhil.Bridge.Player{
        character: 10,  # Mewtwo
        x: :rand.uniform() * 200 - 100,
        y: :rand.uniform() * 100,
        percent: :rand.uniform() * 150,
        stock: 4,
        facing: if(:rand.uniform() > 0.5, do: 1, else: -1),
        action: :rand.uniform(400),
        action_frame: :rand.uniform(30),
        invulnerable: false,
        on_ground: :rand.uniform() > 0.3,
        jumps_left: :rand.uniform(3),
        shield_strength: 60.0,
        speed_air_x_self: :rand.uniform() * 2 - 1,
        speed_ground_x_self: :rand.uniform() * 2 - 1,
        speed_y_self: :rand.uniform() * 2 - 1,
        speed_x_attack: 0.0,
        speed_y_attack: 0.0,
        hitstun_frames_left: 0,
        nana: nil
      },
      2 => %ExPhil.Bridge.Player{
        character: 2,  # Fox
        x: :rand.uniform() * 200 - 100,
        y: :rand.uniform() * 100,
        percent: :rand.uniform() * 150,
        stock: 4,
        facing: if(:rand.uniform() > 0.5, do: 1, else: -1),
        action: :rand.uniform(400),
        action_frame: :rand.uniform(30),
        invulnerable: false,
        on_ground: :rand.uniform() > 0.3,
        jumps_left: :rand.uniform(3),
        shield_strength: 60.0,
        speed_air_x_self: :rand.uniform() * 2 - 1,
        speed_ground_x_self: :rand.uniform() * 2 - 1,
        speed_y_self: :rand.uniform() * 2 - 1,
        speed_x_attack: 0.0,
        speed_y_attack: 0.0,
        hitstun_frames_left: 0,
        nana: nil
      }
    },
    projectiles: [],
    items: []
  }

  # Create random controller state for labels
  controller = %ExPhil.Bridge.ControllerState{
    button_a: :rand.uniform() > 0.9,
    button_b: :rand.uniform() > 0.95,
    button_x: :rand.uniform() > 0.95,
    button_y: :rand.uniform() > 0.95,
    button_z: :rand.uniform() > 0.97,
    button_l: :rand.uniform() > 0.97,
    button_r: :rand.uniform() > 0.95,
    button_d_up: false,
    main_stick: %{x: :rand.uniform(), y: :rand.uniform()},
    c_stick: %{x: 0.5, y: 0.5},
    l_shoulder: 0.0,
    r_shoulder: 0.0
  }

  %{game_state: game_state, controller: controller}
end

dataset = %Data{
  frames: frames,
  metadata: %{},
  embed_config: embed_config,
  size: num_frames,
  embedded_frames: nil,
  embedded_sequences: nil,
  player_registry: nil
}

dataset = Data.precompute_frame_embeddings(dataset, show_progress: false)

IO.puts("  Dataset size: #{dataset.size} frames")
{_frames, embed_size} = Nx.shape(dataset.embedded_frames)
IO.puts("  Embedding size: #{embed_size}")

# Benchmark function
run_benchmark = fn name, config_opts ->
  IO.puts("\n" <> String.duplicate("=", 60))
  IO.puts("Benchmark: #{name}")
  IO.puts(String.duplicate("=", 60))

  # Create trainer using Imitation.new/1
  trainer = Imitation.new(Keyword.merge([
    embed_config: embed_config,
    hidden_sizes: hidden_sizes,
    learning_rate: 1.0e-4,
    epochs: 1
  ], config_opts))

  # Get batches
  batches = Data.batched(dataset, batch_size: batch_size, shuffle: true)
  |> Enum.take(warmup_batches + num_batches)

  # Warmup
  IO.puts("  Warming up (#{warmup_batches} batches)...")
  {trainer, _} = Enum.reduce(Enum.take(batches, warmup_batches), {trainer, []}, fn batch, {tr, _} ->
    {new_tr, metrics} = Imitation.train_step(tr, batch, nil)
    {new_tr, metrics}
  end)

  # Force sync
  :erlang.garbage_collect()

  # Benchmark
  IO.puts("  Running benchmark (#{num_batches} batches)...")

  start_time = System.monotonic_time(:millisecond)

  {final_trainer, losses} =
    batches
    |> Enum.drop(warmup_batches)
    |> Enum.reduce({trainer, []}, fn batch, {tr, losses} ->
      {new_tr, metrics} = Imitation.train_step(tr, batch, nil)
      # Force computation to complete
      loss_val = Nx.to_number(metrics.loss)
      {new_tr, [loss_val | losses]}
    end)

  end_time = System.monotonic_time(:millisecond)
  total_ms = end_time - start_time

  # Calculate stats
  losses = Enum.reverse(losses)
  avg_loss = Enum.sum(losses) / length(losses)
  final_loss = List.last(losses)
  ms_per_batch = total_ms / num_batches
  samples_per_sec = batch_size * num_batches / (total_ms / 1000)

  # Check for NaN/Inf
  has_nan = Enum.any?(losses, fn l -> l != l or l == :infinity or l == :neg_infinity end)

  IO.puts("\n  Results:")
  IO.puts("    Total time: #{total_ms} ms")
  IO.puts("    Time per batch: #{Float.round(ms_per_batch, 2)} ms")
  IO.puts("    Samples/sec: #{Float.round(samples_per_sec, 1)}")
  IO.puts("    Avg loss: #{Float.round(avg_loss, 6)}")
  IO.puts("    Final loss: #{Float.round(final_loss, 6)}")
  IO.puts("    NaN/Inf detected: #{has_nan}")

  %{
    name: name,
    total_ms: total_ms,
    ms_per_batch: ms_per_batch,
    samples_per_sec: samples_per_sec,
    avg_loss: avg_loss,
    final_loss: final_loss,
    has_nan: has_nan
  }
end

# Run benchmarks
results = []

# 1. BF16 only (current default)
results = [run_benchmark.("BF16 Only (default)", [precision: :bf16, mixed_precision: false]) | results]

# 2. Mixed Precision (FP32 master + BF16 compute)
results = [run_benchmark.("Mixed Precision (FP32 master)", [precision: :bf16, mixed_precision: true]) | results]

# 3. FP32 baseline (for comparison)
results = [run_benchmark.("FP32 Baseline", [precision: :f32, mixed_precision: false]) | results]

results = Enum.reverse(results)

# Summary
IO.puts("\n" <> String.duplicate("=", 60))
IO.puts("SUMMARY")
IO.puts(String.duplicate("=", 60))

IO.puts("\n| Mode | ms/batch | samples/sec | Final Loss | NaN? |")
IO.puts("|------|----------|-------------|------------|------|")

for r <- results do
  nan_str = if r.has_nan, do: "YES", else: "no"
  IO.puts("| #{String.pad_trailing(r.name, 25)} | #{String.pad_leading(Float.round(r.ms_per_batch, 1) |> to_string(), 8)} | #{String.pad_leading(Float.round(r.samples_per_sec, 0) |> trunc() |> to_string(), 11)} | #{String.pad_leading(Float.round(r.final_loss, 4) |> to_string(), 10)} | #{nan_str} |")
end

# Speedup calculation
fp32 = Enum.find(results, & &1.name =~ "FP32")
bf16 = Enum.find(results, & &1.name =~ "BF16 Only")
mixed = Enum.find(results, & &1.name =~ "Mixed")

if fp32 && bf16 do
  speedup = fp32.ms_per_batch / bf16.ms_per_batch
  IO.puts("\nBF16 vs FP32 speedup: #{Float.round(speedup, 2)}x")
end

if fp32 && mixed do
  speedup = fp32.ms_per_batch / mixed.ms_per_batch
  IO.puts("Mixed vs FP32 speedup: #{Float.round(speedup, 2)}x")
end

if bf16 && mixed do
  overhead = (mixed.ms_per_batch - bf16.ms_per_batch) / bf16.ms_per_batch * 100
  IO.puts("Mixed precision overhead vs BF16: #{Float.round(overhead, 1)}%")
end

IO.puts("\n✅ Benchmark complete!")
