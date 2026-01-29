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
  """
  @spec save(String.t(), term(), keyword()) :: :ok | {:error, term()}
  def save(cache_key, embeddings, opts \\ []) do
    cache_dir = get_cache_dir(opts)
    path = cache_path(cache_key, cache_dir)

    File.mkdir_p!(cache_dir)

    # Convert to binary-safe format
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

  @doc """
  Load embeddings from cache.
  """
  @spec load(String.t(), keyword()) :: {:ok, term()} | {:error, :not_found | term()}
  def load(cache_key, opts \\ []) do
    cache_dir = get_cache_dir(opts)
    path = cache_path(cache_key, cache_dir)

    if File.exists?(path) do
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
    else
      {:error, :not_found}
    end
  end

  @doc """
  Check if cache exists for a key.
  """
  @spec exists?(String.t(), keyword()) :: boolean()
  def exists?(cache_key, opts \\ []) do
    cache_dir = get_cache_dir(opts)
    path = cache_path(cache_key, cache_dir)
    File.exists?(path)
  end

  @doc """
  Delete a cached embedding.
  """
  @spec invalidate(String.t(), keyword()) :: :ok | {:error, term()}
  def invalidate(cache_key, opts \\ []) do
    cache_dir = get_cache_dir(opts)
    path = cache_path(cache_key, cache_dir)

    case File.rm(path) do
      :ok -> :ok
      {:error, :enoent} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  List all cached embeddings.
  """
  @spec list(keyword()) :: [%{key: String.t(), size_mb: float(), mtime: NaiveDateTime.t()}]
  def list(opts \\ []) do
    cache_dir = get_cache_dir(opts)

    if File.exists?(cache_dir) do
      cache_dir
      |> File.ls!()
      |> Enum.filter(&String.ends_with?(&1, ".emb"))
      |> Enum.map(fn filename ->
        path = Path.join(cache_dir, filename)
        stat = File.stat!(path)
        key = String.replace_suffix(filename, ".emb", "")

        %{
          key: key,
          size_mb: Float.round(stat.size / 1_000_000, 1),
          mtime: NaiveDateTime.from_erl!(stat.mtime)
        }
      end)
      |> Enum.sort_by(& &1.mtime, {:desc, NaiveDateTime})
    else
      []
    end
  end

  @doc """
  Clear all cached embeddings.
  """
  @spec clear(keyword()) :: :ok
  def clear(opts \\ []) do
    cache_dir = get_cache_dir(opts)

    if File.exists?(cache_dir) do
      cache_dir
      |> File.ls!()
      |> Enum.filter(&String.ends_with?(&1, ".emb"))
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
