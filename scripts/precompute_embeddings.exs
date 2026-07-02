#!/usr/bin/env elixir
# Pre-compute frame embeddings and save to cache.
# Run this BEFORE training so the training process starts with a clean GPU.
#
# Usage:
#   mix run scripts/precompute_embeddings.exs --replays ./replays/huggingface --max-files 200
#   mix run scripts/precompute_embeddings.exs --replays ./replays/huggingface  # all files
#
# Then train (guaranteed cache hit, no GPU fragmentation):
#   mix run scripts/train_from_replays.exs --backbone mamba --temporal --max-files 200 ...
#
# Why: The embedding step creates/frees ~340 GPU tensors, fragmenting XLA's BFC allocator
# pool. Training then can't find a contiguous block for the backward pass. Running
# embeddings in a separate process means the GPU pool is pristine when training starts.

alias ExPhil.Training.{Data, Config, Output, EmbeddingCache}
alias ExPhil.Data.Peppi
alias ExPhil.Embeddings

# Parse args
args = System.argv()

replays_dir = case Enum.find_index(args, &(&1 == "--replays")) do
  nil -> "./replays"
  i -> Enum.at(args, i + 1, "./replays")
end

max_files = case Enum.find_index(args, &(&1 == "--max-files")) do
  nil -> nil
  i -> String.to_integer(Enum.at(args, i + 1, "0"))
end

cache_dir = case Enum.find_index(args, &(&1 == "--cache-dir")) do
  nil -> "cache/embeddings"
  i -> Enum.at(args, i + 1, "cache/embeddings")
end

player_port = case Enum.find_index(args, &(&1 == "--player-port")) do
  nil -> 1
  i -> String.to_integer(Enum.at(args, i + 1, "1"))
end

Output.banner("ExPhil Embedding Precomputation")

# Step 1: Find replay files
Output.step(1, 4, "Finding replay files")
replay_dir = Path.expand(replays_dir)
files = Path.wildcard(Path.join(replay_dir, "**/*.slp"))
files = if max_files, do: Enum.take(files, max_files), else: files
Output.puts("  Found #{length(files)} .slp files in #{replay_dir}")

if files == [] do
  Output.error("No .slp files found!")
  System.halt(1)
end

# Step 2: Parse replays
Output.step(2, 4, "Parsing replays")
frames = Enum.flat_map(Enum.with_index(files), fn {path, idx} ->
  if rem(idx, 50) == 0 do
    IO.write(:stderr, "\r  Parsing: #{idx}/#{length(files)}\e[K")
  end
  try do
    case Peppi.parse(path, player_port: player_port) do
      {:ok, replay} -> Peppi.to_training_frames(replay, player_port: player_port)
      _ -> []
    end
  rescue
    _ -> []
  end
end)
IO.write(:stderr, "\r\e[K")
Output.puts("  #{length(frames)} frames from #{length(files)} files")

if frames == [] do
  Output.error("No frames parsed!")
  System.halt(1)
end

# Step 3: Create dataset and compute embeddings
Output.step(3, 4, "Computing frame embeddings")
dataset = Data.from_frames(frames)

embed_config = dataset.embed_config
cache_key = EmbeddingCache.cache_key(embed_config, files, temporal: false)
Output.puts("  Cache key: #{cache_key}")

# Check if already cached
case EmbeddingCache.load(cache_key, cache_dir: cache_dir) do
  {:ok, _} ->
    Output.success("Already cached! Nothing to do.")
    System.halt(0)

  {:error, _} ->
    Output.puts("  Cache miss — computing embeddings...")
end

# Compute on GPU (fast)
embedded = Data.precompute_frame_embeddings(dataset, show_progress: true)

# Step 4: Save to cache
Output.step(4, 4, "Saving to cache")
case EmbeddingCache.save(cache_key, embedded.embedded_frames, cache_dir: cache_dir) do
  :ok ->
    size_mb = Nx.byte_size(embedded.embedded_frames) / 1_000_000
    Output.success("Saved #{Float.round(size_mb, 1)} MB to #{cache_dir}/#{cache_key}")

  {:error, reason} ->
    Output.error("Failed to save: #{inspect(reason)}")
    System.halt(1)
end

Output.puts("\nDone! Training will now get a cache hit and skip GPU embedding.")
