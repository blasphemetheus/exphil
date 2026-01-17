#!/usr/bin/env elixir
# Training script: Parse replays and run imitation learning
#
# Usage:
#   mix run scripts/train_from_replays.exs [options]
#
# Options:
#   --replays PATH    - Path to replay directory (default: ../replays)
#   --epochs N        - Number of training epochs (default: 10)
#   --batch-size N    - Batch size (default: 64)
#   --max-files N     - Max replay files to use (default: all)
#   --checkpoint PATH - Save checkpoint to path
#   --player PORT     - Player port to learn from (1-4, default: 1)

require Logger

alias ExPhil.Data.Peppi
alias ExPhil.Training.{Data, Imitation}
alias ExPhil.Embeddings

# Parse command line arguments
args = System.argv()

opts = [
  replays: Enum.find_value(args, "/home/dori/git/melee/replays", fn
    "--replays" <> _ -> nil
    arg ->
      idx = Enum.find_index(args, &(&1 == "--replays"))
      if idx, do: Enum.at(args, idx + 1), else: nil
  end) || "/home/dori/git/melee/replays",
  epochs: String.to_integer(Enum.find_value(args, "10", fn
    arg ->
      idx = Enum.find_index(args, &(&1 == "--epochs"))
      if idx, do: Enum.at(args, idx + 1), else: nil
  end) || "10"),
  batch_size: String.to_integer(Enum.find_value(args, "64", fn
    arg ->
      idx = Enum.find_index(args, &(&1 == "--batch-size"))
      if idx, do: Enum.at(args, idx + 1), else: nil
  end) || "64"),
  max_files: case Enum.find_index(args, &(&1 == "--max-files")) do
    nil -> nil
    idx -> String.to_integer(Enum.at(args, idx + 1))
  end,
  checkpoint: Enum.find_value(args, "checkpoints/imitation_latest.axon", fn
    arg ->
      idx = Enum.find_index(args, &(&1 == "--checkpoint"))
      if idx, do: Enum.at(args, idx + 1), else: nil
  end) || "checkpoints/imitation_latest.axon",
  player_port: String.to_integer(Enum.find_value(args, "1", fn
    arg ->
      idx = Enum.find_index(args, &(&1 == "--player"))
      if idx, do: Enum.at(args, idx + 1), else: nil
  end) || "1")
]

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║              ExPhil Imitation Learning Training                ║
╚════════════════════════════════════════════════════════════════╝

Configuration:
  Replays:     #{opts[:replays]}
  Epochs:      #{opts[:epochs]}
  Batch Size:  #{opts[:batch_size]}
  Max Files:   #{opts[:max_files] || "all"}
  Player Port: #{opts[:player_port]}
  Checkpoint:  #{opts[:checkpoint]}

""")

# Step 1: Find and parse replays
IO.puts("Step 1: Parsing replays...")

replay_files = Path.wildcard(Path.join(opts[:replays], "**/*.slp"))
replay_files = if opts[:max_files], do: Enum.take(replay_files, opts[:max_files]), else: replay_files

IO.puts("  Found #{length(replay_files)} replay files")

# Parse replays in parallel and collect training frames
{parse_time, all_frames} = :timer.tc(fn ->
  replay_files
  |> Task.async_stream(
    fn path ->
      case Peppi.parse(path, player_port: opts[:player_port]) do
        {:ok, replay} ->
          frames = Peppi.to_training_frames(replay, player_port: opts[:player_port])
          {path, length(frames), frames}
        {:error, reason} ->
          IO.puts("  ⚠ Failed to parse #{Path.basename(path)}: #{reason}")
          {path, 0, []}
      end
    end,
    max_concurrency: System.schedulers_online(),
    timeout: :infinity
  )
  |> Enum.reduce({0, 0, []}, fn
    {:ok, {_path, frame_count, frames}}, {total_files, total_frames, all_frames} ->
      {total_files + 1, total_frames + frame_count, [frames | all_frames]}
    {:exit, _reason}, acc ->
      acc
  end)
  |> then(fn {files, frames, frame_lists} ->
    IO.puts("  Parsed #{files} files, #{frames} total frames")
    List.flatten(frame_lists)
  end)
end)

IO.puts("  Parse time: #{Float.round(parse_time / 1_000_000, 2)}s")
IO.puts("  Total training frames: #{length(all_frames)}")

if length(all_frames) == 0 do
  IO.puts("\n❌ No training frames found. Check replay files and player port.")
  System.halt(1)
end

# Step 2: Create dataset
IO.puts("\nStep 2: Creating dataset...")

dataset = Data.from_frames(all_frames)
{train_dataset, val_dataset} = Data.split(dataset, ratio: 0.9)

IO.puts("  Training frames: #{train_dataset.size}")
IO.puts("  Validation frames: #{val_dataset.size}")

# Show some statistics
stats = Data.stats(train_dataset)
IO.puts("\n  Button press rates:")
for {button, rate} <- Enum.sort(stats.button_rates) do
  bar = String.duplicate("█", round(rate * 50))
  IO.puts("    #{button |> to_string() |> String.pad_trailing(6)}: #{bar} #{Float.round(rate * 100, 1)}%")
end

# Step 3: Initialize trainer
IO.puts("\nStep 3: Initializing model...")

embed_size = Embeddings.embedding_size()
IO.puts("  Embedding size: #{embed_size}")

trainer = Imitation.new(
  embed_size: embed_size,
  hidden_sizes: [512, 512],
  learning_rate: 1.0e-4,
  batch_size: opts[:batch_size]
)

IO.puts("  Model initialized with #{opts[:batch_size]} batch size")

# Step 4: Training loop
IO.puts("\nStep 4: Training for #{opts[:epochs]} epochs...")
IO.puts("─" |> String.duplicate(60))

start_time = System.monotonic_time(:second)

final_trainer = Enum.reduce(1..opts[:epochs], trainer, fn epoch, current_trainer ->
  epoch_start = System.monotonic_time(:second)

  # Create batched dataset for this epoch
  batches = Data.batched(train_dataset,
    batch_size: opts[:batch_size],
    shuffle: true,
    drop_last: true
  )
  |> Enum.to_list()

  num_batches = length(batches)

  # Train epoch
  {updated_trainer, epoch_losses} = Enum.reduce(Enum.with_index(batches), {current_trainer, []}, fn {batch, batch_idx}, {t, losses} ->
    {_predict_fn, loss_fn} = Imitation.build_loss_fn(t.policy_model)
    {new_trainer, metrics} = Imitation.train_step(t, batch, loss_fn)

    # Progress indicator every 10%
    if rem(batch_idx + 1, max(1, div(num_batches, 10))) == 0 do
      pct = round((batch_idx + 1) / num_batches * 100)
      IO.write("\r  Epoch #{epoch}: #{pct}% (loss: #{Float.round(metrics.loss, 4)})")
    end

    {new_trainer, [metrics.loss | losses]}
  end)

  epoch_time = System.monotonic_time(:second) - epoch_start
  avg_loss = Enum.sum(epoch_losses) / length(epoch_losses)

  # Validation
  val_batches = Data.batched(val_dataset, batch_size: opts[:batch_size], shuffle: false)
  val_metrics = Imitation.evaluate(updated_trainer, val_batches)

  IO.puts("\r  Epoch #{epoch}/#{opts[:epochs]}: train_loss=#{Float.round(avg_loss, 4)} val_loss=#{Float.round(val_metrics.loss, 4)} (#{epoch_time}s)")

  updated_trainer
end)

total_time = System.monotonic_time(:second) - start_time
IO.puts("─" |> String.duplicate(60))
IO.puts("Training complete in #{total_time}s")

# Step 5: Save checkpoint
IO.puts("\nStep 5: Saving checkpoint...")
case Imitation.save_checkpoint(final_trainer, opts[:checkpoint]) do
  :ok -> IO.puts("  ✓ Saved to #{opts[:checkpoint]}")
  {:error, reason} -> IO.puts("  ✗ Failed: #{inspect(reason)}")
end

# Also export policy for inference
policy_path = String.replace(opts[:checkpoint], ".axon", "_policy.bin")
case Imitation.export_policy(final_trainer, policy_path) do
  :ok -> IO.puts("  ✓ Policy exported to #{policy_path}")
  {:error, reason} -> IO.puts("  ✗ Failed: #{inspect(reason)}")
end

# Summary
IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                      Training Complete!                        ║
╚════════════════════════════════════════════════════════════════╝

Summary:
  Replays parsed: #{length(replay_files)}
  Training frames: #{train_dataset.size}
  Epochs completed: #{opts[:epochs]}
  Final training loss: #{Float.round(Enum.sum(Enum.take(final_trainer.metrics.loss, 10)) / 10, 4)}
  Checkpoint: #{opts[:checkpoint]}

Next steps:
  1. Test the model: mix run scripts/test_model.exs --checkpoint #{opts[:checkpoint]}
  2. Continue training: mix run scripts/train_from_replays.exs --checkpoint #{opts[:checkpoint]}
  3. Run against CPU: mix run scripts/eval.exs

""")
