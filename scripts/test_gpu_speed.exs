#!/usr/bin/env elixir
# Test script to verify GPU training speed
# Run: mix run scripts/test_gpu_speed.exs

alias ExPhil.Training.Imitation

IO.puts("=" |> String.duplicate(60))
IO.puts("GPU Training Speed Test")
IO.puts("=" |> String.duplicate(60))

# Config
embed_size = 1204
batch_size = 512
num_batches = 10

IO.puts("\nConfig:")
IO.puts("  Embed size: #{embed_size}")
IO.puts("  Batch size: #{batch_size}")
IO.puts("  Test batches: #{num_batches}")

# Create trainer
IO.puts("\nInitializing trainer...")
trainer = Imitation.new(
  embed_size: embed_size,
  hidden_sizes: [512, 512, 256],
  temporal: false,
  learning_rate: 1.0e-4
)
param_count = trainer.config[:param_count] || 0
IO.puts("  ✓ Trainer initialized (#{Float.round(param_count / 1_000_000, 2)}M params)")

# Generate test batch
IO.puts("\nGenerating test batch...")
key = Nx.Random.key(42)
{states, key} = Nx.Random.uniform(key, shape: {batch_size, embed_size}, type: :f32)

# Generate actions
{buttons_f, key} = Nx.Random.uniform(key, shape: {batch_size, 8}, type: :f32)
buttons = buttons_f |> Nx.multiply(2) |> Nx.floor() |> Nx.as_type(:s32)
{main_x_f, key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
main_x = main_x_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)
{main_y_f, key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
main_y = main_y_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)
{c_x_f, key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
c_x = c_x_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)
{c_y_f, key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
c_y = c_y_f |> Nx.multiply(17) |> Nx.floor() |> Nx.as_type(:s32)
{shoulder_f, _key} = Nx.Random.uniform(key, shape: {batch_size}, type: :f32)
shoulder = shoulder_f |> Nx.multiply(4) |> Nx.floor() |> Nx.as_type(:s32)

actions = %{
  buttons: buttons,
  main_x: main_x,
  main_y: main_y,
  c_x: c_x,
  c_y: c_y,
  shoulder: shoulder
}

batch = %{states: states, actions: actions}
IO.puts("  ✓ Batch generated")

# Check tensor backend
IO.puts("\nTensor backends:")
IO.puts("  States: #{inspect(states.data.__struct__)}")
IO.puts("  Buttons: #{inspect(buttons.data.__struct__)}")

# Warm up (JIT compilation)
IO.puts("\n⏳ JIT compiling (first batch)...")
{jit_time_us, {_trainer, _metrics}} = :timer.tc(fn ->
  Imitation.train_step(trainer, batch, nil)
end)
jit_time_s = jit_time_us / 1_000_000
IO.puts("  ✓ JIT complete in #{Float.round(jit_time_s, 1)}s")

# Benchmark
IO.puts("\nRunning #{num_batches} batches...")
{total_time_us, _} = :timer.tc(fn ->
  Enum.reduce(1..num_batches, trainer, fn i, t ->
    {new_t, metrics} = Imitation.train_step(t, batch, nil)
    loss = Nx.to_number(metrics.loss)
    IO.puts("  Batch #{i}/#{num_batches}: loss=#{Float.round(loss, 4)}")
    new_t
  end)
end)

total_time_ms = total_time_us / 1000
avg_time_ms = total_time_ms / num_batches

IO.puts("\n" <> String.duplicate("=", 60))
IO.puts("RESULTS")
IO.puts(String.duplicate("=", 60))
IO.puts("  Total time: #{Float.round(total_time_ms / 1000, 2)}s")
IO.puts("  Avg per batch: #{Float.round(avg_time_ms, 1)}ms")
IO.puts("")

cond do
  avg_time_ms < 500 ->
    IO.puts("  ✅ FAST: #{Float.round(avg_time_ms, 1)}ms/batch - GPU is working correctly!")
  avg_time_ms < 2000 ->
    IO.puts("  ⚠️  MEDIUM: #{Float.round(avg_time_ms, 1)}ms/batch - GPU working but with overhead")
  avg_time_ms < 10000 ->
    IO.puts("  ⚠️  SLOW: #{Float.round(avg_time_ms, 1)}ms/batch - likely CPU→GPU transfer each batch")
  true ->
    IO.puts("  ❌ VERY SLOW: #{Float.round(avg_time_ms, 1)}ms/batch - GPU may not be used")
end

IO.puts("")
IO.puts("Expected performance:")
IO.puts("  < 500ms/batch  = GPU tensors staying on GPU (optimal)")
IO.puts("  1-5s/batch     = CPU→GPU transfer each batch (BinaryBackend copy)")
IO.puts("  > 10s/batch    = No GPU usage (0% GPU util bug)")
IO.puts(String.duplicate("=", 60))
