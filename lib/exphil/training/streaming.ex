defmodule ExPhil.Training.Streaming do
  @moduledoc """
  Streaming/chunked data loading for memory-efficient training on large datasets.

  When training on hundreds or thousands of replay files, loading all frames into
  memory at once can exceed available RAM. This module provides utilities to process
  files in chunks, allowing training on arbitrarily large datasets.

  ## How It Works

  Instead of loading all files at once:
  1. Split files into chunks (e.g., 30 files per chunk)
  2. For each epoch, iterate through all chunks
  3. For each chunk: parse → embed → train → free memory

  ## Usage

      # In training script
      if opts[:stream_chunk_size] do
        Streaming.train_with_chunks(replay_files, trainer, opts)
      else
        # Standard training (load all at once)
      end

  ## Trade-offs

  - **Memory**: Bounded by chunk size, not total dataset size
  - **Speed**: ~10-20% slower due to repeated I/O (can be mitigated with prefetching)
  - **Validation**: Requires sampling or separate validation set
  """

  alias ExPhil.Data.Peppi
  alias ExPhil.Training.{Data, Output}

  require Logger

  @doc """
  Split files into chunks of the given size.

  Returns a list of file chunks, where each chunk is a list of file paths.

  ## Examples

      iex> Streaming.chunk_files(["a.slp", "b.slp", "c.slp"], 2)
      [["a.slp", "b.slp"], ["c.slp"]]
  """
  @spec chunk_files([String.t() | {String.t(), integer()}], pos_integer()) :: [
          [String.t() | {String.t(), integer()}]
        ]
  def chunk_files(files, chunk_size)
      when is_list(files) and is_integer(chunk_size) and chunk_size > 0 do
    Enum.chunk_every(files, chunk_size)
  end

  @doc """
  Parse a chunk of replay files and return training frames.

  This is a standalone version of the parsing logic from train_from_replays.exs,
  designed to be called for each chunk independently.

  ## Options

  - `:player_port` - Default player port (1-4)
  - `:port_map` - Map of {path, port} tuples to their ports (for character-based selection)
  - `:dual_port` - Whether to parse both ports
  - `:frame_delay` - Frame delay for training
  - `:show_progress` - Whether to show parsing progress (default: true)
  """
  @spec parse_chunk([String.t() | {String.t(), integer()}], keyword()) :: {:ok, list(), list()}
  def parse_chunk(files, opts \\ []) do
    player_port = Keyword.get(opts, :player_port, 1)
    port_map = Keyword.get(opts, :port_map, %{})
    dual_port = Keyword.get(opts, :dual_port, false)
    frame_delay = ExPhil.Training.LabelDelay.resolve!(opts)[:label_delay]
    show_progress = Keyword.get(opts, :show_progress, true)
    # Name conditioning: the corpus is anonymized in-file ("Master Player"),
    # so identity comes from the FILENAME bracket tags (FilenameTags).
    # subject_character = the imitated player's display name ("Fox").
    subject_character = Keyword.get(opts, :subject_character)

    if show_progress do
      Output.puts("    Parsing #{length(files)} files...")
    end

    {all_frames, errors} =
      files
      |> Task.async_stream(
        fn path_or_tuple ->
          # Handle both {path, port} tuples (from character filter) and plain paths
          {path, target_port} =
            case path_or_tuple do
              {p, port} ->
                {p, port}

              p when is_binary(p) ->
                if dual_port do
                  # dual_port will parse both
                  {p, nil}
                else
                  {p, Map.get(port_map, p, player_port)}
                end
            end

          if dual_port do
            parse_dual_port(path, frame_delay)
          else
            case Peppi.parse(path, player_port: target_port) do
              {:ok, replay} ->
                # Opponent = the other occupied port. Passing only
                # player_port left opponent_port at its default of 2 — for
                # a port-2 subject that meant opponent == subject: the real
                # opponent was DROPPED from every game_state, distance was
                # 0, and the embedding (own_port hardcoded 1) saw an
                # all-zero self with the subject in the opponent slot (the
                # 09-01 corpus corruption, ~44% of the fox corpus).
                # :remap_ports then normalizes to %{1 => subject, 2 => opp}
                # so the embedding/reward port-1 convention holds.
                opp_port = opponent_port_for(replay, target_port)

                frames =
                  Peppi.to_training_frames(replay,
                    player_port: target_port,
                    opponent_port: opp_port,
                    remap_ports: true,
                    frame_delay: frame_delay
                  )

                frames = maybe_filename_tags(frames, path, subject_character, target_port)

                {:ok, path, length(frames), frames}

              {:error, reason} ->
                {:error, path, reason}
            end
          end
        end,
        max_concurrency: System.schedulers_online(),
        timeout: :infinity
      )
      |> Enum.reduce({[], []}, fn
        {:ok, {:ok, _path, _count, frames}}, {all_frames, errors} ->
          {[frames | all_frames], errors}

        {:ok, {:error, path, reason}}, {all_frames, errors} ->
          {all_frames, [{path, reason} | errors]}

        {:exit, reason}, {all_frames, errors} ->
          {all_frames, [{:unknown, {:exit, reason}} | errors]}
      end)

    {:ok, List.flatten(all_frames), Enum.reverse(errors)}
  end

  # Filename-derived identity (ExPhil.Data.FilenameTags): when the in-file
  # netplay name is the anonymization placeholder, the bracket tag attached
  # to the subject's character in the FILENAME is the only identity signal.
  # DITTOS STAY ANONYMOUS here: positional resolution was MEASURED
  # unreliable (69.2% agreement, n=558 — style_fingerprint.exs
  # --validate-order, 09-04); a ~31%-wrong identity label is worse than
  # none. The intelligent path is fingerprint-based 2-way assignment
  # (STYLE_IDENTITY.md) via the --player-tag-map override, not position.
  # No-op when subject_character is nil, the file has real in-file tags, or
  # the filename is unparseable/ambiguous (frames keep their placeholder ->
  # name_id 0, the anonymous bucket).
  defp maybe_filename_tags(frames, _path, nil, _port), do: frames
  defp maybe_filename_tags([], _path, _subject, _port), do: []

  defp maybe_filename_tags([first | _] = frames, path, subject_character, _port) do
    if ExPhil.Data.FilenameTags.placeholder?(first[:player_tag]) do
      case ExPhil.Data.FilenameTags.subject_tag(path, subject_character) do
        nil -> frames
        tag -> Enum.map(frames, &Map.put(&1, :player_tag, tag))
      end
    else
      frames
    end
  end

  # Opponent = first other port present in the frames (falls back to the
  # 1<->2 convention when the replay somehow has a single player).
  defp opponent_port_for(replay, player_port) do
    replay.frames
    |> Enum.find_value(fn fr ->
      case fr.players |> Map.keys() |> Enum.reject(&(&1 == player_port)) do
        [o | _] -> o
        [] -> nil
      end
    end)
    |> case do
      nil -> if player_port == 1, do: 2, else: 1
      o -> o
    end
  end

  # Parse both ports for dual-port training
  defp parse_dual_port(path, frame_delay) do
    case Peppi.metadata(path) do
      {:ok, meta} ->
        ports = Enum.map(meta.players, & &1.port)

        all_frames =
          Enum.flat_map(ports, fn port ->
            case Peppi.parse(path, player_port: port) do
              {:ok, replay} ->
                Peppi.to_training_frames(replay,
                  player_port: port,
                  opponent_port: opponent_port_for(replay, port),
                  remap_ports: true,
                  frame_delay: frame_delay
                )

              {:error, _} ->
                []
            end
          end)

        {:ok, path, length(all_frames), all_frames}

      {:error, reason} ->
        {:error, path, reason}
    end
  end

  @doc """
  Create a dataset from parsed frames with optional sequence conversion and embedding.

  ## Options

  - `:temporal` - Whether to use temporal/sequence training
  - `:window_size` - Window size for sequences (default: 60)
  - `:stride` - Stride for sequence sampling (default: 1)
  - `:precompute` - Whether to precompute embeddings (default: true)
  - `:embed_config` - Embedding configuration (optional)
  """
  @spec create_dataset(list(), keyword()) :: Data.t()
  def create_dataset(frames, opts \\ []) do
    temporal = Keyword.get(opts, :temporal, false)
    window_size = Keyword.get(opts, :window_size, 60)
    stride = Keyword.get(opts, :stride, 1)
    embed_config = Keyword.get(opts, :embed_config)
    player_registry = Keyword.get(opts, :player_registry)
    # Precompute embeddings per-chunk for efficient GPU utilization.
    # This is NOT wasteful - it's much faster than on-the-fly computation:
    # - Precompute: one batched GPU operation per chunk
    # - On-the-fly: embedding computed every batch (CPU-bound, slow)
    precompute = Keyword.get(opts, :precompute, true)
    show_progress = Keyword.get(opts, :show_progress, false)

    # Build from_frames options with embed_config and player_registry if provided
    from_frames_opts = []

    from_frames_opts =
      if embed_config,
        do: [{:embed_config, embed_config} | from_frames_opts],
        else: from_frames_opts

    from_frames_opts =
      if player_registry,
        do: [{:player_registry, player_registry} | from_frames_opts],
        else: from_frames_opts

    dataset = Data.from_frames(frames, from_frames_opts)

    # Precompute embeddings for this chunk if enabled
    if precompute do
      if temporal do
        # LAZY temporal layout (2026-08-25 rewrite): keep the FLAT frame list
        # plus one {num_frames, dim} embedded tensor; batched_sequences(lazy:
        # true) slices windows on the fly, exactly like the standard pipeline.
        # The old eager path here embedded AFTER to_sequences (so it embedded
        # one last-frame per sequence and sliced windows off a too-short
        # tensor) and finished by writing :sequence_embeddings — a field the
        # Data struct doesn't have. It could never have completed a chunk.
        embedded_dataset = Data.precompute_frame_embeddings(dataset, show_progress: show_progress)
        gpu_embeddings = Nx.backend_transfer(embedded_dataset.embedded_frames, EXLA.Backend)

        %{
          embedded_dataset
          | embedded_frames: gpu_embeddings,
            metadata:
              Map.merge(embedded_dataset.metadata || %{}, %{
                temporal: true,
                window_size: window_size,
                stride: stride
              })
        }
      else
        # For single-frame: just precompute frame embeddings
        embedded_dataset = Data.precompute_frame_embeddings(dataset, show_progress: show_progress)
        # Transfer to GPU for fast Nx.take during batching
        gpu_embeddings = Nx.backend_transfer(embedded_dataset.embedded_frames, EXLA.Backend)
        %{embedded_dataset | embedded_frames: gpu_embeddings}
      end
    else
      # No precompute: legacy eager sequence structs (embeds at batch time)
      if temporal do
        Data.to_sequences(dataset, window_size: window_size, stride: stride)
      else
        dataset
      end
    end
  end

  @doc """
  Compute the number of total training examples across all chunks.

  This is useful for progress tracking and should be called once at startup.
  Uses sampling to estimate if there are many files.
  """
  @spec estimate_total_examples([String.t() | {String.t(), integer()}], keyword()) ::
          {:ok, integer()}
  def estimate_total_examples(files, opts \\ []) do
    sample_size = min(10, length(files))

    if sample_size == 0 do
      {:ok, 0}
    else
      # Sample a few files to estimate frames per file
      sample = Enum.take_random(files, sample_size)
      {:ok, sample_frames, _} = parse_chunk(sample, Keyword.put(opts, :show_progress, false))
      avg_frames_per_file = length(sample_frames) / sample_size
      estimated_total = round(avg_frames_per_file * length(files))
      {:ok, estimated_total}
    end
  end

  @doc """
  Format streaming configuration for display.
  """
  @spec format_config(integer(), integer()) :: String.t()
  def format_config(chunk_size, total_files) do
    num_chunks = ceil(total_files / chunk_size)
    "#{num_chunks} chunks of #{chunk_size} files (#{total_files} total)"
  end
end
