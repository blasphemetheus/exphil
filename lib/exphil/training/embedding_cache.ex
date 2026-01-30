defmodule ExPhil.Training.EmbeddingCache do
  @moduledoc """
  Disk caching for precomputed embeddings.

  Saves precomputed embeddings to disk so they can be reused across training runs
  without recomputing. Cache keys are based on:
  - Embedding config (action_mode, nana_mode, stage_mode, etc.)
  - Replay file list (sorted paths)
  - Window size (for temporal embeddings)

  ## Usage

      # Check for cached embeddings
      cache_key = EmbeddingCache.cache_key(embed_config, replay_files, opts)

      case EmbeddingCache.load(cache_key) do
        {:ok, embeddings} ->
          # Use cached embeddings
          embeddings

        {:error, :not_found} ->
          # Compute and cache
          embeddings = compute_embeddings(...)
          EmbeddingCache.save(cache_key, embeddings)
          embeddings
      end

  ## Cache Location

  Default: `cache/embeddings/`
  Override with `EXPHIL_CACHE_DIR` env var or `:cache_dir` option.
  """

  require Logger

  @default_cache_dir "cache/embeddings"
  # Max bytes per chunk (~500MB) - well under Erlang's term_to_binary limit
  @max_chunk_bytes 500_000_000

  @doc """
  Generate a cache key from embedding config and replay files.

  The key is a hash of:
  - Sorted replay file paths
  - Embedding config (serialized)
  - Optional: window_size, stride for temporal
  - Optional: augmented, num_noisy_variants, noise_scale for augmented caches

  ## Options
    - `:temporal` - Whether this is temporal (sequence) data
    - `:window_size` - Window size for temporal data
    - `:stride` - Stride for temporal data
    - `:augmented` - Whether this is an augmented cache (default: false)
    - `:num_noisy_variants` - Number of noisy variants (for augmented)
    - `:noise_scale` - Noise scale (for augmented)
  """
  @spec cache_key(map(), [String.t()], keyword()) :: String.t()
  def cache_key(embed_config, replay_files, opts \\ []) do
    window_size = Keyword.get(opts, :window_size)
    stride = Keyword.get(opts, :stride, 1)
    temporal = Keyword.get(opts, :temporal, false)

    # Augmentation options
    augmented = Keyword.get(opts, :augmented, false)
    num_noisy_variants = Keyword.get(opts, :num_noisy_variants)
    noise_scale = Keyword.get(opts, :noise_scale)

    # Sort files for deterministic hashing
    sorted_files = Enum.sort(replay_files)

    # Serialize config to stable format
    config_data = serialize_config(embed_config)

    # Compute actual embedding size (includes padding logic)
    # This ensures cache invalidates if embedding_size() implementation changes
    actual_embed_size = ExPhil.Embeddings.Game.embedding_size(embed_config)

    # Build hash input
    hash_input = %{
      files: sorted_files,
      config: config_data,
      # Include computed embedding size to catch padding/alignment changes
      embedding_size: actual_embed_size,
      temporal: temporal,
      window_size: window_size,
      stride: stride,
      # Include augmentation params so augmented caches have different keys
      augmented: augmented,
      num_noisy_variants: num_noisy_variants,
      noise_scale: noise_scale
    }

    # Generate SHA256 hash
    hash_input
    |> :erlang.term_to_binary()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> String.slice(0, 16)
  end

  @doc """
  Save embeddings to cache.

  For frame embeddings: saves an :array of tensors
  For sequence embeddings: saves a %Data{} struct with embedded_sequences

  Large tensors are automatically chunked to avoid Erlang's term_to_binary limits.
  """
  @spec save(String.t(), term(), keyword()) :: :ok | {:error, term()}
  def save(cache_key, embeddings, opts \\ []) do
    cache_dir = get_cache_dir(opts)
    File.mkdir_p!(cache_dir)

    # Check if we need chunked saving for large stacked tensors
    case embeddings do
      tensor when is_struct(tensor, Nx.Tensor) ->
        save_tensor(cache_key, tensor, cache_dir, opts)

      other ->
        save_single_file(cache_key, other, cache_dir)
    end
  end

  # Save a tensor, chunking if necessary
  defp save_tensor(cache_key, tensor, cache_dir, _opts) do
    {num_frames, _rest} = shape_head_rest(Nx.shape(tensor))
    bytes_per_frame = Nx.byte_size(tensor) / num_frames
    frames_per_chunk = max(1, floor(@max_chunk_bytes / bytes_per_frame))

    if num_frames <= frames_per_chunk do
      # Small enough for single file
      save_single_file(cache_key, tensor, cache_dir)
    else
      # Need to chunk
      save_chunked(cache_key, tensor, cache_dir, frames_per_chunk)
    end
  end

  defp shape_head_rest(shape) do
    [head | rest] = Tuple.to_list(shape)
    {head, List.to_tuple(rest)}
  end

  # Save as single file (original behavior)
  defp save_single_file(cache_key, embeddings, cache_dir) do
    path = cache_path(cache_key, cache_dir)
    data = prepare_for_save(embeddings)

    case File.write(path, :erlang.term_to_binary(data, [:compressed])) do
      :ok ->
        size_mb = File.stat!(path).size / 1_000_000
        Logger.info("[EmbeddingCache] Saved #{cache_key} (#{Float.round(size_mb, 1)} MB)")
        :ok

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Save large tensor in chunks
  defp save_chunked(cache_key, tensor, cache_dir, frames_per_chunk) do
    {num_frames, _} = shape_head_rest(Nx.shape(tensor))
    num_chunks = ceil(num_frames / frames_per_chunk)
    shape = Nx.shape(tensor)
    type = Nx.type(tensor)

    Logger.info("[EmbeddingCache] Saving #{cache_key} in #{num_chunks} chunks...")

    # Save each chunk
    chunk_info =
      for chunk_idx <- 0..(num_chunks - 1) do
        start_idx = chunk_idx * frames_per_chunk
        end_idx = min((chunk_idx + 1) * frames_per_chunk, num_frames)
        chunk_frames = end_idx - start_idx

        # Slice the tensor
        chunk_tensor = Nx.slice_along_axis(tensor, start_idx, chunk_frames, axis: 0)
        chunk_binary = Nx.to_binary(chunk_tensor)

        # Save chunk file
        chunk_path = chunk_path(cache_key, chunk_idx, cache_dir)
        File.write!(chunk_path, chunk_binary)

        %{
          index: chunk_idx,
          start: start_idx,
          frames: chunk_frames,
          size_bytes: byte_size(chunk_binary)
        }
      end

    # Save manifest
    manifest = %{
      type: :chunked_stacked_embeddings,
      shape: Tuple.to_list(shape),
      dtype: type,
      num_frames: num_frames,
      num_chunks: num_chunks,
      frames_per_chunk: frames_per_chunk,
      chunks: chunk_info
    }

    manifest_path = manifest_path(cache_key, cache_dir)
    File.write!(manifest_path, :erlang.term_to_binary(manifest, [:compressed]))

    total_mb =
      chunk_info
      |> Enum.map(& &1.size_bytes)
      |> Enum.sum()
      |> Kernel./(1_000_000)

    Logger.info(
      "[EmbeddingCache] Saved #{cache_key} (#{Float.round(total_mb, 1)} MB in #{num_chunks} chunks)"
    )

    :ok
  end

  @doc """
  Load embeddings from cache.

  Automatically handles both single-file and chunked formats.
  """
  @spec load(String.t(), keyword()) :: {:ok, term()} | {:error, :not_found | term()}
  def load(cache_key, opts \\ []) do
    cache_dir = get_cache_dir(opts)
    path = cache_path(cache_key, cache_dir)
    manifest = manifest_path(cache_key, cache_dir)

    cond do
      # Check for chunked format first (manifest file)
      File.exists?(manifest) ->
        load_chunked(cache_key, cache_dir)

      # Fall back to single file
      File.exists?(path) ->
        load_single_file(cache_key, cache_dir)

      true ->
        {:error, :not_found}
    end
  end

  defp load_single_file(cache_key, cache_dir) do
    path = cache_path(cache_key, cache_dir)

    case File.read(path) do
      {:ok, binary} ->
        data = :erlang.binary_to_term(binary)
        embeddings = restore_from_load(data)
        size_mb = byte_size(binary) / 1_000_000
        Logger.info("[EmbeddingCache] Loaded #{cache_key} (#{Float.round(size_mb, 1)} MB)")
        {:ok, embeddings}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp load_chunked(cache_key, cache_dir) do
    manifest_file = manifest_path(cache_key, cache_dir)

    with {:ok, manifest_binary} <- File.read(manifest_file),
         manifest <- :erlang.binary_to_term(manifest_binary) do
      %{
        shape: shape_list,
        dtype: dtype,
        num_chunks: num_chunks,
        chunks: chunks
      } = manifest

      Logger.info("[EmbeddingCache] Loading #{cache_key} from #{num_chunks} chunks...")

      total_mb =
        chunks
        |> Enum.map(& &1.size_bytes)
        |> Enum.sum()
        |> Kernel./(1_000_000)

      # For large chunked data, keep on CPU and let training transfer batches
      # GPU concatenation of 10GB+ data causes OOM during the concat operation
      # (needs 2x memory: accumulated + new chunk + result)
      Logger.info("[EmbeddingCache] Loading to CPU (GPU transfer happens per-batch during training)")

      sorted_chunks = Enum.sort_by(chunks, & &1.index)

      # Force all operations to use CPU (BinaryBackend) to avoid GPU OOM
      # EXLA may be the default backend which would try to allocate 10GB+ on GPU
      tensor = Nx.with_default_backend(Nx.BinaryBackend, fn ->
        # Load all chunks to CPU first (fast), then concatenate on CPU
        chunk_tensors =
          sorted_chunks
          |> Enum.with_index()
          |> Enum.map(fn {chunk_info, idx} ->
            if rem(idx, 5) == 0 do
              Logger.info("[EmbeddingCache] Loading chunk #{idx + 1}/#{num_chunks}...")
            end

            chunk_file = chunk_path(cache_key, chunk_info.index, cache_dir)
            {:ok, binary} = File.read(chunk_file)

            # Reconstruct chunk shape: {chunk_frames, ...rest}
            [_total_frames | rest] = shape_list
            chunk_shape = List.to_tuple([chunk_info.frames | rest])

            Nx.from_binary(binary, dtype) |> Nx.reshape(chunk_shape)
          end)

        Logger.info("[EmbeddingCache] Concatenating #{num_chunks} chunks on CPU...")
        Nx.concatenate(chunk_tensors, axis: 0)
      end)

      Logger.info(
        "[EmbeddingCache] Loaded #{cache_key} (#{Float.round(total_mb, 1)} MB from #{num_chunks} chunks)"
      )

      {:ok, tensor}
    end
  end

  @doc """
  Check if cache exists for a key.

  Checks for both single-file and chunked formats.
  """
  @spec exists?(String.t(), keyword()) :: boolean()
  def exists?(cache_key, opts \\ []) do
    cache_dir = get_cache_dir(opts)
    path = cache_path(cache_key, cache_dir)
    manifest = manifest_path(cache_key, cache_dir)

    File.exists?(path) or File.exists?(manifest)
  end

  @doc """
  Delete a cached embedding.

  Handles both single-file and chunked formats.
  """
  @spec invalidate(String.t(), keyword()) :: :ok | {:error, term()}
  def invalidate(cache_key, opts \\ []) do
    cache_dir = get_cache_dir(opts)

    # Delete single file if exists
    path = cache_path(cache_key, cache_dir)
    File.rm(path)

    # Delete chunked files if exist
    manifest_file = manifest_path(cache_key, cache_dir)

    if File.exists?(manifest_file) do
      case File.read(manifest_file) do
        {:ok, binary} ->
          manifest = :erlang.binary_to_term(binary)

          # Delete all chunk files
          for chunk <- manifest.chunks do
            File.rm(chunk_path(cache_key, chunk.index, cache_dir))
          end

          # Delete manifest
          File.rm(manifest_file)

        _ ->
          :ok
      end
    end

    :ok
  end

  @doc """
  List all cached embeddings.

  Shows both single-file and chunked caches with total size.
  """
  @spec list(keyword()) :: [%{key: String.t(), size_mb: float(), mtime: NaiveDateTime.t(), chunked: boolean()}]
  def list(opts \\ []) do
    cache_dir = get_cache_dir(opts)

    if File.exists?(cache_dir) do
      files = File.ls!(cache_dir)

      # Find single-file caches (.emb files without corresponding .manifest)
      single_file_caches =
        files
        |> Enum.filter(&String.ends_with?(&1, ".emb"))
        |> Enum.map(fn filename ->
          key = String.replace_suffix(filename, ".emb", "")
          manifest_exists = "#{key}.manifest" in files

          unless manifest_exists do
            path = Path.join(cache_dir, filename)
            stat = File.stat!(path)

            %{
              key: key,
              size_mb: Float.round(stat.size / 1_000_000, 1),
              mtime: NaiveDateTime.from_erl!(stat.mtime),
              chunked: false
            }
          end
        end)
        |> Enum.reject(&is_nil/1)

      # Find chunked caches (.manifest files)
      chunked_caches =
        files
        |> Enum.filter(&String.ends_with?(&1, ".manifest"))
        |> Enum.map(fn filename ->
          key = String.replace_suffix(filename, ".manifest", "")
          manifest_file = Path.join(cache_dir, filename)
          stat = File.stat!(manifest_file)

          # Calculate total size from chunks
          total_size =
            case File.read(manifest_file) do
              {:ok, binary} ->
                manifest = :erlang.binary_to_term(binary)

                manifest.chunks
                |> Enum.map(& &1.size_bytes)
                |> Enum.sum()

              _ ->
                0
            end

          %{
            key: key,
            size_mb: Float.round(total_size / 1_000_000, 1),
            mtime: NaiveDateTime.from_erl!(stat.mtime),
            chunked: true
          }
        end)

      (single_file_caches ++ chunked_caches)
      |> Enum.sort_by(& &1.mtime, {:desc, NaiveDateTime})
    else
      []
    end
  end

  @doc """
  Clear all cached embeddings.

  Removes both single-file (.emb) and chunked (.manifest + _chunk_*.bin) formats.
  """
  @spec clear(keyword()) :: :ok
  def clear(opts \\ []) do
    cache_dir = get_cache_dir(opts)

    if File.exists?(cache_dir) do
      cache_dir
      |> File.ls!()
      |> Enum.filter(fn filename ->
        String.ends_with?(filename, ".emb") or
          String.ends_with?(filename, ".manifest") or
          String.contains?(filename, "_chunk_")
      end)
      |> Enum.each(fn filename ->
        File.rm!(Path.join(cache_dir, filename))
      end)
    end

    :ok
  end

  # Private helpers

  defp cache_path(cache_key, cache_dir) do
    Path.join(cache_dir, "#{cache_key}.emb")
  end

  defp manifest_path(cache_key, cache_dir) do
    Path.join(cache_dir, "#{cache_key}.manifest")
  end

  defp chunk_path(cache_key, chunk_index, cache_dir) do
    Path.join(cache_dir, "#{cache_key}_chunk_#{chunk_index}.bin")
  end

  defp get_cache_dir(opts) do
    Keyword.get(opts, :cache_dir) ||
      System.get_env("EXPHIL_CACHE_DIR") ||
      @default_cache_dir
  end

  # Serialize embed config to a stable, hashable format
  defp serialize_config(config) when is_struct(config) do
    config
    |> Map.from_struct()
    |> serialize_config()
  end

  defp serialize_config(config) when is_map(config) do
    config
    |> Enum.sort_by(fn {k, _} -> to_string(k) end)
    |> Enum.map(fn {k, v} -> {to_string(k), serialize_value(v)} end)
  end

  defp serialize_value(v) when is_struct(v), do: serialize_config(v)
  defp serialize_value(v) when is_map(v), do: serialize_config(v)
  defp serialize_value(v), do: v

  # Prepare embeddings for disk storage
  # Convert Nx tensors to binary format

  # NEW: Stacked tensor format {num_frames, embed_size} - much more efficient
  defp prepare_for_save(embeddings) when is_struct(embeddings, Nx.Tensor) do
    shape = Nx.shape(embeddings)
    type = Nx.type(embeddings)
    binary = Nx.to_binary(embeddings)

    {:stacked_frame_embeddings, %{binary: binary, shape: shape, type: type}}
  end

  # LEGACY: :array of individual tensors (for backwards compatibility)
  defp prepare_for_save(embeddings) when is_tuple(embeddings) and elem(embeddings, 0) == :array do
    # :array of tensors - convert each to binary
    size = :array.size(embeddings)

    tensors =
      for i <- 0..(size - 1) do
        tensor = :array.get(i, embeddings)
        Nx.to_binary(tensor)
      end

    # Get shape from first tensor
    first = :array.get(0, embeddings)
    shape = Nx.shape(first)
    type = Nx.type(first)

    {:frame_embeddings, %{tensors: tensors, shape: shape, type: type, size: size}}
  end

  defp prepare_for_save(%{embedded_sequences: seqs} = dataset) when seqs != nil do
    # Dataset with embedded sequences
    size = :array.size(seqs)

    tensors =
      for i <- 0..(size - 1) do
        tensor = :array.get(i, seqs)
        Nx.to_binary(tensor)
      end

    first = :array.get(0, seqs)
    shape = Nx.shape(first)
    type = Nx.type(first)

    # Also save the frames and metadata needed to reconstruct
    {:sequence_embeddings,
     %{
       tensors: tensors,
       shape: shape,
       type: type,
       size: size,
       frames: dataset.frames,
       metadata: dataset.metadata,
       embed_config: dataset.embed_config,
       dataset_size: dataset.size
     }}
  end

  defp prepare_for_save(other), do: {:raw, other}

  # Restore embeddings from loaded data

  # NEW: Stacked tensor format - direct restore
  defp restore_from_load({:stacked_frame_embeddings, %{binary: binary, shape: shape, type: type}}) do
    Nx.from_binary(binary, type) |> Nx.reshape(shape)
  end

  # LEGACY: Convert old array format to stacked tensor for consistency
  # This ensures all cached data works with the new fast batching code
  defp restore_from_load({:frame_embeddings, %{tensors: tensors, shape: shape, type: type}}) do
    # Restore individual tensors and stack them
    restored_tensors =
      Enum.map(tensors, fn binary ->
        Nx.from_binary(binary, type) |> Nx.reshape(shape)
      end)

    # Stack into single tensor {num_frames, embed_size}
    Nx.stack(restored_tensors)
  end

  defp restore_from_load(
         {:sequence_embeddings,
          %{
            tensors: tensors,
            shape: shape,
            type: type,
            frames: frames,
            metadata: metadata,
            embed_config: embed_config,
            dataset_size: dataset_size
          }}
       ) do
    array =
      tensors
      |> Enum.with_index()
      |> Enum.reduce(:array.new(), fn {binary, i}, arr ->
        tensor = Nx.from_binary(binary, type) |> Nx.reshape(shape)
        :array.set(i, tensor, arr)
      end)

    %ExPhil.Training.Data{
      frames: frames,
      metadata: metadata,
      embed_config: embed_config,
      size: dataset_size,
      embedded_sequences: array
    }
  end

  defp restore_from_load({:raw, data}), do: data
end
