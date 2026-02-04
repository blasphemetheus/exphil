defmodule ExPhil.Training.ChunkPipeline do
  @moduledoc """
  Pipelined chunk preparation for streaming mode.

  Prepares chunk N+1 while training on chunk N, hiding preparation latency
  behind GPU compute time. This dramatically reduces total training time
  when chunk preparation (parsing + embedding) dominates.

  ## How It Works

  ```
  Without pipelining:
    Chunk 1: [──parse──][────embed────][─train─]
    Chunk 2:                                    [──parse──][────embed────][─train─]
    Total: 2 × (parse + embed + train)

  With pipelining:
    Chunk 1: [──parse──][────embed────][─train─]
    Chunk 2:            [──parse──][────embed────][─train─]
    Total: parse + embed + N × max(embed, train) + train
  ```

  ## Embedding Cache

  When `cache_embeddings: true`, embeddings are saved to disk after first computation
  and reloaded on subsequent epochs. This eliminates re-embedding overhead:

  ```
  Epoch 1: [parse][embed 10m][train] → save cache
  Epoch 2: [parse][load 5s][train]   → 100x faster!
  ```

  ## Memory Trade-off

  Pipelining keeps two chunks in memory simultaneously:
  - Current chunk being trained
  - Next chunk being prepared

  For typical streaming configs (100 files/chunk, ~1M frames), expect ~4GB extra.

  ## Usage

      # Enable in training script
      mix run scripts/train_from_replays.exs --stream-chunk-size 100 --pipeline-chunks

      # With caching (recommended for multi-epoch training)
      mix run scripts/train_from_replays.exs --stream-chunk-size 100 --cache-streaming

      # Programmatic usage
      ChunkPipeline.stream_prepared_chunks(file_chunks,
        chunk_opts: [...],
        dataset_opts: [...],
        buffer_size: 1,
        cache_embeddings: true,
        embed_config: embed_config
      )
  """

  alias ExPhil.Training.{Streaming, Output, EmbeddingCache, Data}

  require Logger

  @doc """
  Create a stream of prepared datasets with look-ahead pipelining.

  While the consumer processes chunk N, this prepares chunk N+1 in the background.
  This hides chunk preparation time (parsing + embedding) behind training time.

  ## Options

  - `:chunk_opts` - Options passed to `Streaming.parse_chunk/2`
  - `:dataset_opts` - Options passed to `Streaming.create_dataset/2`
  - `:buffer_size` - Number of chunks to prepare ahead (default: 1)
  - `:show_progress` - Whether to show chunk preparation progress (default: true)
  - `:cache_embeddings` - Whether to cache embeddings to disk (default: false)
  - `:cache_dir` - Directory for embedding cache (default: "cache/embeddings")
  - `:embed_config` - Embedding config for cache key generation (required if caching)

  ## Returns

  A stream that yields `{dataset, chunk_idx, errors}` tuples, where:
  - `dataset` - The prepared `Data.t()` with precomputed embeddings
  - `chunk_idx` - The 1-based chunk index (for progress display)
  - `errors` - List of `{path, reason}` parsing errors
  """
  @spec stream_prepared_chunks([[String.t()]], keyword()) :: Enumerable.t()
  def stream_prepared_chunks(file_chunks, opts \\ []) do
    chunk_opts = Keyword.get(opts, :chunk_opts, [])
    dataset_opts = Keyword.get(opts, :dataset_opts, [])
    buffer_size = Keyword.get(opts, :buffer_size, 1)
    show_progress = Keyword.get(opts, :show_progress, true)
    cache_embeddings = Keyword.get(opts, :cache_embeddings, false)
    cache_dir = Keyword.get(opts, :cache_dir, "cache/embeddings")
    embed_config = Keyword.get(opts, :embed_config)
    total_chunks = length(file_chunks)

    # Build preparation options
    prep_opts = %{
      chunk_opts: chunk_opts,
      dataset_opts: dataset_opts,
      show_progress: show_progress,
      cache_embeddings: cache_embeddings,
      cache_dir: cache_dir,
      embed_config: embed_config
    }

    Stream.resource(
      # Init: start preparing first buffer_size chunks
      fn ->
        initial_tasks =
          file_chunks
          |> Enum.take(buffer_size)
          |> Enum.with_index(1)
          |> Enum.map(fn {chunk, idx} ->
            start_chunk_preparation(chunk, idx, total_chunks, prep_opts)
          end)

        remaining_chunks =
          file_chunks
          |> Enum.drop(buffer_size)
          |> Enum.with_index(buffer_size + 1)

        {initial_tasks, remaining_chunks, total_chunks, prep_opts}
      end,

      # Next: yield prepared chunk, start next preparation
      fn state ->
        {tasks, remaining, total, p_opts} = state

        case tasks do
          [] ->
            {:halt, state}

          [current_task | rest_tasks] ->
            # Wait for current chunk to finish preparing
            {dataset, chunk_idx, errors} = Task.await(current_task, :infinity)

            # Start preparing next chunk if available
            {new_tasks, new_remaining} =
              case remaining do
                [{next_chunk, next_idx} | rest_remaining] ->
                  new_task = start_chunk_preparation(next_chunk, next_idx, total, p_opts)
                  {rest_tasks ++ [new_task], rest_remaining}

                [] ->
                  {rest_tasks, []}
              end

            new_state = {new_tasks, new_remaining, total, p_opts}
            {[{dataset, chunk_idx, errors}], new_state}
        end
      end,

      # Cleanup: cancel any pending tasks
      fn {tasks, _, _, _} ->
        Enum.each(tasks, fn task ->
          Task.shutdown(task, :brutal_kill)
        end)
      end
    )
  end

  @doc """
  Create a batch stream from pipelined chunks.

  Combines `stream_prepared_chunks/2` with batch creation, yielding a flat
  stream of training batches across all chunks.

  ## Options

  All options from `stream_prepared_chunks/2`, plus:
  - `:batch_size` - Batch size for training (required)
  - `:temporal` - Whether to use temporal/sequence batching
  - `:shuffle` - Whether to shuffle batches within each chunk (default: true)
  - `:drop_last` - Whether to drop incomplete final batch (default: false)
  - Additional options passed to `Data.batched/2` or `Data.batched_sequences/2`
  """
  @spec stream_batches([[String.t()]], keyword()) :: Enumerable.t()
  def stream_batches(file_chunks, opts \\ []) do
    batch_size = Keyword.fetch!(opts, :batch_size)
    temporal = Keyword.get(opts, :temporal, false)
    shuffle = Keyword.get(opts, :shuffle, true)
    drop_last = Keyword.get(opts, :drop_last, false)

    # Separate pipeline opts from batch opts
    pipeline_opts = Keyword.take(opts, [:chunk_opts, :dataset_opts, :buffer_size, :show_progress])

    # Additional batch options
    batch_opts =
      opts
      |> Keyword.drop([
        :chunk_opts,
        :dataset_opts,
        :buffer_size,
        :show_progress,
        :batch_size,
        :temporal
      ])
      |> Keyword.merge(batch_size: batch_size, shuffle: shuffle, drop_last: drop_last)

    file_chunks
    |> stream_prepared_chunks(pipeline_opts)
    |> Stream.flat_map(fn {dataset, _chunk_idx, _errors} ->
      if dataset.size == 0 do
        []
      else
        if temporal do
          ExPhil.Training.Data.batched_sequences(dataset, batch_opts)
        else
          ExPhil.Training.Data.batched(dataset, batch_opts)
        end
      end
    end)
  end

  # Start async chunk preparation with optional caching
  defp start_chunk_preparation(chunk_files, chunk_idx, total_chunks, prep_opts) do
    Task.async(fn ->
      %{
        chunk_opts: chunk_opts,
        dataset_opts: dataset_opts,
        show_progress: show_progress,
        cache_embeddings: cache_embeddings,
        cache_dir: cache_dir,
        embed_config: embed_config
      } = prep_opts

      # Show progress (note: output may interleave with training progress)
      if show_progress do
        Output.puts("  🔄 Preparing chunk #{chunk_idx}/#{total_chunks} (#{length(chunk_files)} files)...")
      end

      # Generate cache key if caching is enabled
      cache_key =
        if cache_embeddings and embed_config do
          # Use sorted file paths for deterministic key
          sorted_files = chunk_files |> Enum.map(&normalize_path/1) |> Enum.sort()
          EmbeddingCache.cache_key(embed_config, sorted_files, dataset_opts)
        end

      # Check cache
      cached_embeddings =
        if cache_key && EmbeddingCache.exists?(cache_key, cache_dir: cache_dir) do
          if show_progress do
            Output.puts("    📦 Loading cached embeddings...")
          end

          case EmbeddingCache.load(cache_key, cache_dir: cache_dir) do
            {:ok, embeddings} ->
              if show_progress do
                Output.puts("    ✓ Cache hit!")
              end
              embeddings

            {:error, _} ->
              nil
          end
        end

      # Parse files (always needed for frame data/labels)
      parse_opts = Keyword.put(chunk_opts, :show_progress, false)
      {:ok, frames, errors} = Streaming.parse_chunk(chunk_files, parse_opts)

      if length(errors) > 0 and show_progress do
        Output.warning("Chunk #{chunk_idx}: #{length(errors)} file(s) failed to parse")
      end

      # Create dataset - either with cached embeddings or compute fresh
      dataset =
        if cached_embeddings do
          # Use cached embeddings - skip embedding computation
          create_dataset_with_cached_embeddings(frames, cached_embeddings, dataset_opts)
        else
          # Compute embeddings fresh
          dataset = Streaming.create_dataset(frames, dataset_opts)

          # Save to cache if enabled
          if cache_key && dataset.embedded_frames do
            if show_progress do
              Output.puts("    💾 Saving embeddings to cache...")
            end

            EmbeddingCache.save(cache_key, dataset.embedded_frames, cache_dir: cache_dir)
          end

          dataset
        end

      if show_progress do
        cache_status = if cached_embeddings, do: " (cached)", else: ""
        Output.puts("  ✓ Chunk #{chunk_idx}/#{total_chunks} ready (#{dataset.size} sequences#{cache_status})")
      end

      {dataset, chunk_idx, errors}
    end)
  end

  # Create a dataset using pre-loaded cached embeddings
  defp create_dataset_with_cached_embeddings(frames, cached_embeddings, opts) do
    temporal = Keyword.get(opts, :temporal, false)
    window_size = Keyword.get(opts, :window_size, 60)
    stride = Keyword.get(opts, :stride, 1)
    embed_config = Keyword.get(opts, :embed_config)
    player_registry = Keyword.get(opts, :player_registry)

    # Build base dataset from frames (without embedding)
    from_frames_opts = []
    from_frames_opts = if embed_config, do: [{:embed_config, embed_config} | from_frames_opts], else: from_frames_opts
    from_frames_opts = if player_registry, do: [{:player_registry, player_registry} | from_frames_opts], else: from_frames_opts

    base_dataset = Data.from_frames(frames, from_frames_opts)

    # Attach cached embeddings
    dataset = %{base_dataset | embedded_frames: cached_embeddings}

    # Transfer embeddings to GPU for fast batching
    gpu_embeddings = Nx.backend_transfer(cached_embeddings, EXLA.Backend)
    dataset = %{dataset | embedded_frames: gpu_embeddings}

    # Convert to sequences if temporal
    if temporal do
      Data.sequences_from_frame_embeddings(
        dataset,
        gpu_embeddings,
        window_size: window_size,
        stride: stride,
        show_progress: false
      )
    else
      dataset
    end
  end

  # Normalize path for cache key (handle {path, port} tuples)
  defp normalize_path({path, _port}), do: path
  defp normalize_path(path) when is_binary(path), do: path
end
