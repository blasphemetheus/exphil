defmodule ExPhil.Training.MmapEmbeddings do
  @moduledoc """
  Memory-mapped embedding storage for training on datasets larger than RAM.

  Instead of loading all embeddings into memory, this module stores embeddings
  in a binary file and uses memory-mapping for random access. This allows
  training on datasets that exceed available RAM.

  ## How it works

  1. **Save phase**: Embeddings are written to a contiguous binary file with a header
  2. **Load phase**: File is opened with `:raw` mode for efficient sequential reads
  3. **Access**: Specific batches are read by calculating byte offsets

  ## File format

  ```
  [Header: 32 bytes]
    - Magic: 4 bytes ("EMBD")
    - Version: 4 bytes (1)
    - Num frames: 8 bytes (u64)
    - Embed size: 8 bytes (u64)
    - Dtype: 4 bytes (f32=1, f16=2, bf16=3)
    - Reserved: 4 bytes

  [Data: num_frames * embed_size * dtype_bytes]
    - Contiguous f32/f16/bf16 values
  ```

  ## Usage

  ```elixir
  # Save embeddings to disk
  MmapEmbeddings.save(embeddings_tensor, "embeddings.bin")

  # Load handle (doesn't read all data into memory)
  {:ok, handle} = MmapEmbeddings.open("embeddings.bin")

  # Read specific batch of embeddings
  batch_embeddings = MmapEmbeddings.read_batch(handle, [0, 5, 10, 15])

  # Close when done
  MmapEmbeddings.close(handle)
  ```

  ## Performance characteristics

  - **Memory**: O(batch_size) instead of O(dataset_size)
  - **Latency**: ~1-2ms per batch read (SSD), ~10-20ms (HDD)
  - **Throughput**: Limited by disk I/O, not memory bandwidth
  - **Best for**: Datasets > 50% of available RAM
  """

  require Logger

  @magic "EMBD"
  @version 1
  @header_size 32

  @type dtype :: :f32 | :f16 | :bf16
  @type handle :: %{
          fd: :file.io_device(),
          num_frames: non_neg_integer(),
          embed_size: non_neg_integer(),
          dtype: dtype(),
          bytes_per_value: 2 | 4,
          bytes_per_frame: non_neg_integer(),
          path: String.t()
        }

  @doc """
  Save embeddings tensor to a memory-mappable file.

  ## Parameters

    - `embeddings` - Nx tensor of shape `{num_frames, embed_size}`
    - `path` - Output file path

  ## Options

    - `:dtype` - Data type to save as (default: same as input tensor)
    - `:show_progress` - Show progress bar (default: true)

  ## Returns

    - `:ok` on success
    - `{:error, reason}` on failure
  """
  @spec save(Nx.Tensor.t(), String.t(), keyword()) :: :ok | {:error, term()}
  def save(embeddings, path, opts \\ []) do
    show_progress = Keyword.get(opts, :show_progress, true)

    # Validate input
    case Nx.shape(embeddings) do
      {num_frames, embed_size} ->
        # Get dtype from tensor or options
        tensor_type = Nx.type(embeddings)
        dtype = Keyword.get(opts, :dtype, nx_type_to_dtype(tensor_type))
        # bytes_per_value used for header, but header uses dtype_to_code directly
        _bytes_per_value = dtype_to_bytes(dtype)

        if show_progress do
          Logger.info("[MmapEmbeddings] Saving #{num_frames} frames (#{embed_size} dims) to #{path}")
        end

        # Ensure directory exists
        dir = Path.dirname(path)
        File.mkdir_p!(dir)

        # Write file
        case File.open(path, [:write, :binary, :raw]) do
          {:ok, fd} ->
            try do
              # Write header
              header = build_header(num_frames, embed_size, dtype)
              :ok = :file.write(fd, header)

              # Convert to target dtype if needed
              embeddings_typed =
                if nx_type_to_dtype(tensor_type) == dtype do
                  embeddings
                else
                  Nx.as_type(embeddings, dtype_to_nx_type(dtype))
                end

              # Write data
              # Copy to binary backend for serialization
              embeddings_cpu = Nx.backend_copy(embeddings_typed, Nx.BinaryBackend)
              binary_data = Nx.to_binary(embeddings_cpu)
              :ok = :file.write(fd, binary_data)

              if show_progress do
                file_size_mb = (byte_size(binary_data) + @header_size) / 1_000_000
                Logger.info("[MmapEmbeddings] Saved #{Float.round(file_size_mb, 1)} MB to #{path}")
              end

              :ok
            after
              :file.close(fd)
            end

          {:error, reason} ->
            {:error, {:file_open, reason}}
        end

      shape ->
        {:error, {:invalid_shape, shape, "expected {num_frames, embed_size}"}}
    end
  end

  @doc """
  Open an embeddings file for reading.

  Returns a handle that can be used with `read_batch/2` and `read_frame/2`.
  The file is opened but data is not loaded into memory until accessed.

  ## Returns

    - `{:ok, handle}` on success
    - `{:error, reason}` on failure
  """
  @spec open(String.t()) :: {:ok, handle()} | {:error, term()}
  def open(path) do
    case File.open(path, [:read, :binary, :raw]) do
      {:ok, fd} ->
        # Read and parse header
        case :file.read(fd, @header_size) do
          {:ok, header_data} ->
            case parse_header(header_data) do
              {:ok, {num_frames, embed_size, dtype}} ->
                bytes_per_value = dtype_to_bytes(dtype)
                bytes_per_frame = embed_size * bytes_per_value

                handle = %{
                  fd: fd,
                  num_frames: num_frames,
                  embed_size: embed_size,
                  dtype: dtype,
                  bytes_per_value: bytes_per_value,
                  bytes_per_frame: bytes_per_frame,
                  path: path
                }

                {:ok, handle}

              {:error, reason} ->
                :file.close(fd)
                {:error, reason}
            end

          {:error, reason} ->
            :file.close(fd)
            {:error, {:read_header, reason}}
        end

      {:error, reason} ->
        {:error, {:file_open, reason}}
    end
  end

  @doc """
  Close an embeddings file handle.
  """
  @spec close(handle()) :: :ok
  def close(%{fd: fd}) do
    :file.close(fd)
    :ok
  end

  @doc """
  Read a batch of embeddings by frame indices.

  ## Parameters

    - `handle` - Handle from `open/1`
    - `indices` - List of frame indices to read

  ## Returns

  Nx tensor of shape `{length(indices), embed_size}`
  """
  @spec read_batch(handle(), [non_neg_integer()]) :: Nx.Tensor.t()
  def read_batch(handle, indices) when is_list(indices) do
    %{
      fd: fd,
      embed_size: embed_size,
      dtype: dtype,
      bytes_per_frame: bytes_per_frame
    } = handle

    nx_type = dtype_to_nx_type(dtype)

    # Read each frame
    frames =
      Enum.map(indices, fn idx ->
        offset = @header_size + idx * bytes_per_frame
        {:ok, data} = :file.pread(fd, offset, bytes_per_frame)
        Nx.from_binary(data, nx_type)
      end)

    # Stack into batch tensor
    Nx.stack(frames)
    |> Nx.reshape({length(indices), embed_size})
  end

  @doc """
  Read a single frame embedding.

  ## Parameters

    - `handle` - Handle from `open/1`
    - `idx` - Frame index

  ## Returns

  Nx tensor of shape `{embed_size}`
  """
  @spec read_frame(handle(), non_neg_integer()) :: Nx.Tensor.t()
  def read_frame(handle, idx) do
    %{
      fd: fd,
      embed_size: embed_size,
      dtype: dtype,
      bytes_per_frame: bytes_per_frame
    } = handle

    offset = @header_size + idx * bytes_per_frame
    {:ok, data} = :file.pread(fd, offset, bytes_per_frame)

    Nx.from_binary(data, dtype_to_nx_type(dtype))
    |> Nx.reshape({embed_size})
  end

  @doc """
  Get metadata about the embeddings file.
  """
  @spec info(handle()) :: map()
  def info(handle) do
    %{
      path: handle.path,
      num_frames: handle.num_frames,
      embed_size: handle.embed_size,
      dtype: handle.dtype,
      size_mb: (handle.num_frames * handle.bytes_per_frame + @header_size) / 1_000_000
    }
  end

  @doc """
  Check if embeddings exist at the given path and are valid.
  """
  @spec exists?(String.t()) :: boolean()
  def exists?(path) do
    case open(path) do
      {:ok, handle} ->
        close(handle)
        true

      {:error, _} ->
        false
    end
  end

  # ==========================================================================
  # Private helpers
  # ==========================================================================

  defp build_header(num_frames, embed_size, dtype) do
    dtype_code = dtype_to_code(dtype)

    <<
      @magic::binary,
      @version::little-unsigned-32,
      num_frames::little-unsigned-64,
      embed_size::little-unsigned-64,
      dtype_code::little-unsigned-32,
      0::little-unsigned-32
    >>
  end

  defp parse_header(<<
         "EMBD"::binary,
         version::little-unsigned-32,
         num_frames::little-unsigned-64,
         embed_size::little-unsigned-64,
         dtype_code::little-unsigned-32,
         _reserved::little-unsigned-32
       >>) do
    if version != @version do
      {:error, {:unsupported_version, version}}
    else
      case code_to_dtype(dtype_code) do
        {:ok, dtype} -> {:ok, {num_frames, embed_size, dtype}}
        error -> error
      end
    end
  end

  defp parse_header(_), do: {:error, :invalid_header}

  defp dtype_to_code(:f32), do: 1
  defp dtype_to_code(:f16), do: 2
  defp dtype_to_code(:bf16), do: 3

  defp code_to_dtype(1), do: {:ok, :f32}
  defp code_to_dtype(2), do: {:ok, :f16}
  defp code_to_dtype(3), do: {:ok, :bf16}
  defp code_to_dtype(code), do: {:error, {:unknown_dtype_code, code}}

  defp dtype_to_bytes(:f32), do: 4
  defp dtype_to_bytes(:f16), do: 2
  defp dtype_to_bytes(:bf16), do: 2

  defp dtype_to_nx_type(:f32), do: {:f, 32}
  defp dtype_to_nx_type(:f16), do: {:f, 16}
  defp dtype_to_nx_type(:bf16), do: {:bf, 16}

  defp nx_type_to_dtype({:f, 32}), do: :f32
  defp nx_type_to_dtype({:f, 16}), do: :f16
  defp nx_type_to_dtype({:bf, 16}), do: :bf16
  defp nx_type_to_dtype(_), do: :f32  # Default to f32 for other types
end
