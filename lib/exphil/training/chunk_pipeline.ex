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

  ## Memory Trade-off

  Pipelining keeps two chunks in memory simultaneously:
  - Current chunk being trained
  - Next chunk being prepared

  For typical streaming configs (100 files/chunk, ~1M frames), expect ~4GB extra.

  ## Usage

      # Enable in training script
      mix run scripts/train_from_replays.exs --stream-chunk-size 100 --pipeline-chunks

      # Programmatic usage
      ChunkPipeline.stream_prepared_chunks(file_chunks,
        chunk_opts: [...],
        dataset_opts: [...],
        buffer_size: 1
      )
  """

  alias ExPhil.Training.{Streaming, Output}

  @doc """
  Create a stream of prepared datasets with look-ahead pipelining.

  While the consumer processes chunk N, this prepares chunk N+1 in the background.
  This hides chunk preparation time (parsing + embedding) behind training time.

  ## Options

  - `:chunk_opts` - Options passed to `Streaming.parse_chunk/2`
  - `:dataset_opts` - Options passed to `Streaming.create_dataset/2`
  - `:buffer_size` - Number of chunks to prepare ahead (default: 1)
  - `:show_progress` - Whether to show chunk preparation progress (default: true)

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
    total_chunks = length(file_chunks)

    Stream.resource(
      # Init: start preparing first buffer_size chunks
      fn ->
        initial_tasks =
          file_chunks
          |> Enum.take(buffer_size)
          |> Enum.with_index(1)
          |> Enum.map(fn {chunk, idx} ->
            start_chunk_preparation(chunk, idx, total_chunks, chunk_opts, dataset_opts, show_progress)
          end)

        remaining_chunks =
          file_chunks
          |> Enum.drop(buffer_size)
          |> Enum.with_index(buffer_size + 1)

        {initial_tasks, remaining_chunks, total_chunks, chunk_opts, dataset_opts, show_progress}
      end,

      # Next: yield prepared chunk, start next preparation
      fn state ->
        {tasks, remaining, total, c_opts, d_opts, progress} = state

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
                  new_task =
                    start_chunk_preparation(
                      next_chunk,
                      next_idx,
                      total,
                      c_opts,
                      d_opts,
                      progress
                    )

                  {rest_tasks ++ [new_task], rest_remaining}

                [] ->
                  {rest_tasks, []}
              end

            new_state = {new_tasks, new_remaining, total, c_opts, d_opts, progress}
            {[{dataset, chunk_idx, errors}], new_state}
        end
      end,

      # Cleanup: cancel any pending tasks
      fn {tasks, _, _, _, _, _} ->
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

  # Start async chunk preparation
  defp start_chunk_preparation(chunk_files, chunk_idx, total_chunks, chunk_opts, dataset_opts, show_progress) do
    Task.async(fn ->
      # Show progress (note: output may interleave with training progress)
      if show_progress do
        Output.puts("  🔄 Preparing chunk #{chunk_idx}/#{total_chunks} (#{length(chunk_files)} files)...")
      end

      # Parse files
      parse_opts = Keyword.put(chunk_opts, :show_progress, false)
      {:ok, frames, errors} = Streaming.parse_chunk(chunk_files, parse_opts)

      if length(errors) > 0 and show_progress do
        Output.warning("Chunk #{chunk_idx}: #{length(errors)} file(s) failed to parse")
      end

      # Create dataset with embeddings
      # Note: show_progress for embedding is controlled separately
      dataset = Streaming.create_dataset(frames, dataset_opts)

      if show_progress do
        Output.puts("  ✓ Chunk #{chunk_idx}/#{total_chunks} ready (#{dataset.size} sequences)")
      end

      {dataset, chunk_idx, errors}
    end)
  end
end
