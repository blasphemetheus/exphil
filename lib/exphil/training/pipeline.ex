defmodule ExPhil.Training.Pipeline do
  @moduledoc """
  Data pipeline for training — from replay files to ready-to-train batches.

  Encapsulates the full data flow: find replays → validate → parse → embed →
  cache → split train/val → precompute val batches. The training loop only
  interacts with `batch_stream/2` and the struct fields.

  ## Usage

      {:ok, pipeline} = Pipeline.setup(opts)
      {batch_stream, num_batches} = Pipeline.batch_stream(pipeline, epoch: 1)

  ## Streaming Mode

  For datasets too large to fit in memory, pass `stream_chunk_size: N`.
  Replays are parsed in chunks during training instead of upfront. No
  validation set is available in streaming mode.
  """

  require Logger

  alias ExPhil.Training.{Data, Output}
  alias ExPhil.Training.{Streaming, CharacterBalance, PlayerRegistry}
  alias ExPhil.Data.Peppi
  alias ExPhil.Embeddings

  defstruct [
    # Datasets (nil in streaming mode)
    :train_dataset,
    :val_dataset,

    # Precomputed validation batches (on CPU, transferred per-batch during eval)
    :val_batches,

    # Test dataset (held-out, only evaluated at end of training)
    :test_dataset,

    # Replay info
    :replay_files,
    :replay_stats,

    # Embedding config
    :embed_config,

    # Resolved options (button_pos_weight tensor, character_weights, etc.)
    :resolved_opts,

    # Streaming mode state
    :streaming,
    :file_chunks,
    :streaming_chunk_opts,
    :streaming_dataset_opts,

    # Augmentation
    :augment_fn,

    # Character balancing weights
    :character_weights,

    # Player registry (for style-conditional training)
    :player_registry,

    # Batch count estimate (for progress display)
    :estimated_batches
  ]

  @type t :: %__MODULE__{}

  # ============================================================================
  # Setup
  # ============================================================================

  @doc """
  Set up the full data pipeline.

  Finds, validates, and parses replay files. Creates train/val datasets with
  precomputed embeddings. Resolves config options. Returns a ready-to-use
  pipeline struct.

  ## Options

  All training config options are accepted. Key ones:
  - `:replays` — Path to replay directory
  - `:max_files` — Limit number of files
  - `:temporal` — Enable temporal/sequence training
  - `:backbone` — Architecture type
  - `:batch_size` — Batch size
  - `:val_split` — Fraction for validation (default: 0.1)
  - `:stream_chunk_size` — Enable streaming mode with N files per chunk
  - `:cache_embeddings` — Cache frame embeddings to disk (default: true)
  - `:lazy_sequences` — Slice sequences on-the-fly (default: true for temporal)
  """
  @spec setup(keyword()) :: {:ok, t()} | {:error, term()}
  def setup(opts) do
    streaming = opts[:stream_chunk_size] != nil and opts[:stream_chunk_size] > 0

    with {:ok, replay_files, replay_stats} <- find_and_validate_replays(opts),
         {:ok, pipeline} <- build_pipeline(replay_files, replay_stats, streaming, opts) do
      {:ok, pipeline}
    end
  end

  @doc "Same as setup/1 but raises on error."
  @spec setup!(keyword()) :: t()
  def setup!(opts) do
    case setup(opts) do
      {:ok, pipeline} -> pipeline
      {:error, reason} -> raise "Pipeline setup failed: #{inspect(reason)}"
    end
  end

  # ============================================================================
  # Batch Stream
  # ============================================================================

  @doc """
  Create a lazy batch stream for one epoch of training.

  Returns `{stream, num_batches}` where stream yields batch maps with
  `:states` and `:actions` keys.

  ## Options
  - `:shuffle` — Shuffle batches (default: true)
  - `:drop_last` — Drop incomplete final batch (default: true)
  """
  @spec batch_stream(t(), keyword()) :: {Enumerable.t(), non_neg_integer()}
  def batch_stream(%{streaming: true} = pipeline, opts) do
    batch_stream_streaming(pipeline, opts)
  end

  def batch_stream(%{streaming: false} = pipeline, opts) do
    batch_stream_standard(pipeline, opts)
  end

  @doc """
  Return the number of validation batches, or 0 if no validation set.
  """
  def val_batch_count(%{val_batches: nil}), do: 0
  def val_batch_count(%{val_batches: batches}), do: length(batches)

  @doc """
  Check if pipeline has a validation set.
  """
  def has_validation?(%{val_batches: nil}), do: false
  def has_validation?(%{val_batches: []}), do: false
  def has_validation?(_), do: true

  # ============================================================================
  # Private — Setup
  # ============================================================================

  defp find_and_validate_replays(opts) do
    replay_dir = Path.expand(opts[:replays] || "./replays")
    Output.step(1, 4, "Finding replays")

    files = Path.wildcard(Path.join(replay_dir, "**/*.slp"))

    if files == [] do
      {:error, "No .slp files found in #{replay_dir}"}
    else
      files = if opts[:max_files], do: Enum.take(files, opts[:max_files]), else: files

      # Character filtering
      files =
        if character = opts[:train_character] do
          Output.puts("  Filtering for character: #{character}")
          Enum.filter(files, fn path ->
            case Peppi.metadata(path) do
              {:ok, meta} ->
                Enum.any?(meta.players, fn p ->
                  to_string(p.character) |> String.downcase() == to_string(character) |> String.downcase()
                end)
              _ -> false
            end
          end)
        else
          files
        end

      # Stage filtering
      files =
        if (stages = opts[:stages]) && stages != [] do
          stage_list = if is_list(stages), do: stages, else: String.split(to_string(stages), ",")
          Enum.filter(files, fn path ->
            case Peppi.metadata(path) do
              {:ok, meta} -> to_string(meta.stage) in stage_list
              _ -> false
            end
          end)
        else
          files
        end

      # Quality filtering
      files =
        if min_quality = opts[:min_quality] do
          alias ExPhil.Training.ReplayQuality
          Enum.filter(files, fn path ->
            try do
              case Peppi.metadata(path) do
                {:ok, meta} ->
                  score = ReplayQuality.score(meta)
                  is_number(score) and score >= min_quality
                _ -> true
              end
            rescue
              _ -> true
            end
          end)
        else
          files
        end

      Output.puts("  #{length(files)} replay files")

      # Quick validation — filter obviously bad files
      {valid_files, stats} = validate_files(files, opts)

      {:ok, valid_files, stats}
    end
  end

  defp validate_files(files, opts) do
    if length(files) > 500 or (opts[:skip_validation] || false) do
      Output.puts("  Skipping file validation")
      {files, %{total: length(files), invalid: 0}}
    else
      # Quick metadata check
      {valid, invalid} =
        Enum.split_with(files, fn path ->
          case Peppi.metadata(path) do
            {:ok, _meta} -> true
            _ -> false
          end
        end)

      if length(invalid) > 0 and (opts[:show_errors] || false) do
        Output.warning("#{length(invalid)} invalid replay files will be skipped")
      end

      {valid, %{total: length(files), invalid: length(invalid)}}
    end
  end

  defp build_pipeline(replay_files, replay_stats, true = _streaming, opts) do
    # Streaming mode — don't load data upfront
    Output.step(2, 4, "Setting up streaming pipeline")

    chunk_size = opts[:stream_chunk_size]
    file_chunks = Streaming.chunk_files(replay_files, chunk_size)
    Output.puts("  #{length(file_chunks)} chunks of ~#{chunk_size} files")

    embed_config = Embeddings.config(
      action_mode: opts[:action_mode] || :learned,
      character_mode: opts[:character_mode] || :learned,
      stage_mode: opts[:stage_mode] || :one_hot_compact
    )

    # Estimate batch count
    estimated = estimate_streaming_batches(replay_files, opts)

    pipeline = %__MODULE__{
      replay_files: replay_files,
      replay_stats: replay_stats,
      embed_config: embed_config,
      streaming: true,
      file_chunks: file_chunks,
      streaming_chunk_opts: Keyword.take(opts, [
        :player_port, :dual_port, :frame_delay, :skip_errors, :show_errors
      ]),
      streaming_dataset_opts: Keyword.take(opts, [
        :temporal, :window_size, :stride, :precompute, :lazy_sequences
      ]) ++ [embed_config: embed_config],
      val_batches: nil,
      character_weights: nil,
      augment_fn: nil,
      estimated_batches: estimated,
      resolved_opts: opts
    }

    {:ok, pipeline}
  end

  defp build_pipeline(replay_files, replay_stats, false = _streaming, opts) do
    # Standard mode — load everything upfront
    Output.step(2, 4, "Parsing and embedding replays")

    # Parse all replays
    frames = parse_replays(replay_files, opts)

    if frames == [] do
      {:error, "No frames parsed from #{length(replay_files)} files"}
    else
      Output.puts("  #{length(frames)} frames from #{length(replay_files)} files")
      log_character_distribution(frames)

      # Build embed config
      embed_config = Embeddings.config(
        action_mode: opts[:action_mode] || :learned,
        character_mode: opts[:character_mode] || :learned,
        stage_mode: opts[:stage_mode] || :one_hot_compact,
        kmeans_centers: opts[:kmeans_centers]
      )

      # Create base dataset
      dataset = Data.from_frames(frames, embed_config: embed_config)

      # Player registry for style-conditional training
      player_registry =
        if opts[:learn_player_styles] do
          case PlayerRegistry.from_replays(replay_files) do
            {:ok, reg} -> reg
            _ -> nil
          end
        else
          nil
        end

      # Precompute embeddings on full dataset FIRST, then split.
      # This ensures train/val get correctly sliced embedded_frames tensors.
      Output.step(3, 4, "Computing embeddings")
      embedded_dataset =
        if opts[:cache_augmented] do
          Data.precompute_augmented_frame_embeddings_cached(dataset,
            cache: true,
            cache_dir: opts[:cache_dir] || "cache/embeddings",
            replay_files: replay_files,
            num_noisy_variants: opts[:num_noisy_variants] || 3,
            noise_scale: opts[:noise_scale] || 0.01,
            show_progress: true
          )
        else
          precompute_embeddings(dataset, replay_files, opts)
        end

      # Split AFTER embedding — each split gets the correct slice
      # For lazy sequences, don't shuffle — sequential order is required for windowed slicing
      split_opts = if opts[:lazy_sequences] != false, do: [shuffle: false], else: []
      {train_ds, val_ds} = split_dataset(embedded_dataset, Keyword.merge(opts, split_opts))
      Output.puts("  Train: #{train_ds.size} frames, Val: #{val_ds.size} frames")

      # Optionally convert to memory-mapped embeddings for huge datasets
      train_ds =
        if opts[:mmap_embeddings] && train_ds.embedded_frames do
          alias ExPhil.Training.MmapEmbeddings
          mmap_path = opts[:mmap_path] || "cache/mmap_embeddings.bin"
          Output.puts("  Saving embeddings to mmap file: #{mmap_path}")
          MmapEmbeddings.save(train_ds.embedded_frames, mmap_path)
          {:ok, handle} = MmapEmbeddings.open(mmap_path)
          %{train_ds | embedded_frames: nil, metadata: Map.put(train_ds.metadata, :mmap_handle, handle)}
        else
          train_ds
        end

      # Resolve config (button_pos_weight, character_weights)
      resolved_opts = resolve_opts(train_ds, opts)
      character_weights = compute_character_weights(train_ds, opts)
      augment_fn = build_augment_fn(opts)

      # Compute batch count
      estimated_batches = estimate_standard_batches(train_ds, opts)

      # Precompute val batches (on CPU for lazy mode)
      Output.step(4, 4, "Preparing validation")
      val_batches = precompute_val_batches(val_ds, opts)

      pipeline = %__MODULE__{
        train_dataset: train_ds,
        val_dataset: val_ds,
        val_batches: val_batches,
        replay_files: replay_files,
        replay_stats: replay_stats,
        embed_config: embed_config,
        streaming: false,
        character_weights: character_weights,
        augment_fn: augment_fn,
        player_registry: player_registry,
        estimated_batches: estimated_batches,
        resolved_opts: resolved_opts
      }

      {:ok, pipeline}
    end
  end

  # ============================================================================
  # Private — Parsing
  # ============================================================================

  defp parse_replays(files, opts) do
    player_port = opts[:player_port] || 1
    dual_port = opts[:dual_port] || false
    skip_errors = opts[:skip_errors] != false
    total = length(files)

    files
    |> Enum.with_index()
    |> Enum.flat_map(fn {path, idx} ->
      if rem(idx, max(div(total, 20), 1)) == 0 do
        pct = round(idx / total * 100)
        IO.write(:stderr, "\r  Parsing: #{pct}% (#{idx}/#{total})\e[K")
      end

      try do
        if dual_port do
          # Parse both ports — doubles training data
          parse_both_ports(path)
        else
          case Peppi.parse(path, player_port: player_port) do
            {:ok, replay} ->
              Peppi.to_training_frames(replay, player_port: player_port)
            _ -> []
          end
        end
      rescue
        e ->
          if not skip_errors, do: raise(e)
          []
      end
    end)
    |> tap(fn _ -> IO.write(:stderr, "\r\e[K") end)
  end

  defp parse_both_ports(path) do
    case Peppi.metadata(path) do
      {:ok, meta} ->
        Enum.flat_map(meta.players, fn p ->
          case Peppi.parse(path, player_port: p.port) do
            {:ok, replay} -> Peppi.to_training_frames(replay, player_port: p.port)
            _ -> []
          end
        end)
      _ -> []
    end
  end

  # ============================================================================
  # Private — Embeddings
  # ============================================================================

  defp precompute_embeddings(dataset, replay_files, opts) do
    cache_enabled = opts[:cache_embeddings] != false

    if cache_enabled do
      Data.precompute_frame_embeddings_cached(dataset,
        cache: true,
        cache_dir: opts[:cache_dir] || "cache/embeddings",
        force_recompute: opts[:no_cache] || false,
        replay_files: replay_files,
        show_progress: true
      )
    else
      Data.precompute_frame_embeddings(dataset, show_progress: true)
    end
  end

  # ============================================================================
  # Private — Split & Validation
  # ============================================================================

  defp split_dataset(dataset, opts) do
    val_split = opts[:val_split] || 0.1
    shuffle = Keyword.get(opts, :shuffle, true)

    if val_split > 0 and dataset.size > 0 do
      Data.split(dataset, ratio: 1.0 - val_split, shuffle: shuffle)
    else
      {dataset, %{dataset | frames: [], size: 0, embedded_frames: nil}}
    end
  end

  defp precompute_val_batches(val_ds, opts) do
    if val_ds.size == 0 do
      nil
    else
      IO.write(:stderr, "  Pre-computing validation batches...\e[K")
      start = System.monotonic_time(:millisecond)

      stream =
        if opts[:temporal] do
          Data.batched_sequences(val_ds,
            batch_size: opts[:batch_size] || 32,
            shuffle: false,
            lazy: opts[:lazy_sequences] != false,
            gpu: false,
            window_size: opts[:window_size] || 60,
            stride: opts[:stride] || 5
          )
        else
          Data.batched(val_ds, batch_size: opts[:batch_size] || 32, shuffle: false)
        end

      batches =
        stream
        |> Stream.with_index()
        |> Enum.map(fn {batch, idx} ->
          if rem(idx, 50) == 0 do
            IO.write(:stderr, "\r  Pre-computing validation batches... #{idx}\e[K")
          end
          batch
        end)

      elapsed = System.monotonic_time(:millisecond) - start
      IO.write(:stderr, "\r  Pre-computing validation batches... done (#{length(batches)} batches, #{div(elapsed, 1000)}s)\e[K\n")

      batches
    end
  end

  # ============================================================================
  # Private — Config Resolution
  # ============================================================================

  defp resolve_opts(train_ds, opts) do
    opts
    |> resolve_button_pos_weight(train_ds)
  end

  defp resolve_button_pos_weight(opts, train_ds) do
    case opts[:button_pos_weight] do
      :auto ->
        stats = Data.stats(train_ds)
        button_order = [:a, :b, :x, :y, :z, :l, :r, :d_up]
        rates = Enum.map(button_order, fn btn -> Map.get(stats.button_rates, btn, 0.0) end)
        rates_tensor = Nx.tensor(rates, type: :f32) |> Nx.backend_copy(Nx.BinaryBackend)
        pos_weight = ExPhil.Networks.Policy.Loss.compute_pos_weights_from_rates(rates_tensor)
        Output.puts("  Per-button pos_weight: #{inspect(Nx.to_flat_list(pos_weight) |> Enum.map(&Float.round(&1, 1)))}")
        Keyword.put(opts, :button_pos_weight, pos_weight)

      weights when is_list(weights) ->
        Keyword.put(opts, :button_pos_weight, Nx.tensor(weights, type: :f32) |> Nx.backend_copy(Nx.BinaryBackend))

      _ ->
        opts
    end
  end

  defp compute_character_weights(train_ds, opts) do
    if opts[:balance_characters] do
      CharacterBalance.compute_weights(train_ds.frames)
    else
      nil
    end
  end

  defp build_augment_fn(opts) do
    if opts[:augment] do
      alias ExPhil.Training.Augmentation

      mirror_prob = opts[:mirror_prob] || 0.5
      noise_prob = opts[:noise_prob] || 0.3
      noise_scale = opts[:noise_scale] || 0.01

      fn frame ->
        frame
        |> Augmentation.maybe_mirror(probability: mirror_prob)
        |> Augmentation.maybe_add_noise(probability: noise_prob, scale: noise_scale)
      end
    else
      nil
    end
  end

  # ============================================================================
  # Private — Batch Streams
  # ============================================================================

  defp batch_stream_standard(pipeline, opts) do
    ropts = pipeline.resolved_opts

    stream =
      if ropts[:temporal] do
        Data.batched_sequences(pipeline.train_dataset,
          batch_size: ropts[:batch_size] || 32,
          shuffle: Keyword.get(opts, :shuffle, true),
          drop_last: Keyword.get(opts, :drop_last, true),
          character_weights: pipeline.character_weights,
          action_oversample: ropts[:action_oversample],
          lazy: ropts[:lazy_sequences] != false,
          use_batch: ropts[:use_batch] || false,
          window_size: ropts[:window_size] || 60,
          stride: ropts[:stride] || 5,
          neutral_weight: Keyword.get(ropts, :neutral_weight, 0.25)
        )
      else
        Data.batched(pipeline.train_dataset,
          batch_size: ropts[:batch_size] || 32,
          shuffle: Keyword.get(opts, :shuffle, true),
          drop_last: Keyword.get(opts, :drop_last, true),
          augment_fn: pipeline.augment_fn,
          character_weights: pipeline.character_weights,
          frame_delay: ropts[:frame_delay] || 0,
          frame_delay_augment: ropts[:frame_delay_augment] || false,
          frame_delay_min: ropts[:frame_delay_min] || 0,
          frame_delay_max: ropts[:frame_delay_max] || 3
        )
      end

    {stream, pipeline.estimated_batches}
  end

  defp batch_stream_streaming(pipeline, _opts) do
    ropts = pipeline.resolved_opts
    chunk_opts = pipeline.streaming_chunk_opts
    dataset_opts = pipeline.streaming_dataset_opts

    stream =
      if ropts[:pipeline_chunks] do
        # Pipelined: parse chunk N+1 while training on chunk N
        alias ExPhil.Training.ChunkPipeline
        ChunkPipeline.stream_prepared_chunks(pipeline.file_chunks,
          chunk_opts: chunk_opts,
          dataset_opts: dataset_opts,
          batch_size: ropts[:batch_size] || 32,
          temporal: ropts[:temporal]
        )
      else
        pipeline.file_chunks
        |> Stream.flat_map(fn chunk ->
          {:ok, chunk_frames, _errors} = Streaming.parse_chunk(chunk, chunk_opts)
          chunk_dataset = Streaming.create_dataset(chunk_frames, dataset_opts)

          if ropts[:temporal] do
            Data.batched_sequences(chunk_dataset,
              batch_size: ropts[:batch_size] || 32,
              shuffle: true,
              drop_last: true
            )
          else
            Data.batched(chunk_dataset,
              batch_size: ropts[:batch_size] || 32,
              shuffle: true,
              drop_last: true
            )
          end
        end)
      end

    {stream, pipeline.estimated_batches}
  end

  # ============================================================================
  # Private — Batch Count Estimation
  # ============================================================================

  defp estimate_standard_batches(train_ds, opts) do
    batch_size = opts[:batch_size] || 32

    num_items =
      if opts[:temporal] and opts[:lazy_sequences] != false and train_ds.embedded_frames != nil do
        {num_frames, _} = Nx.shape(train_ds.embedded_frames)
        window = opts[:window_size] || 60
        stride = opts[:stride] || 5
        div(num_frames - window, stride) + 1
      else
        train_ds.size
      end

    div(num_items, batch_size)
  end

  defp estimate_streaming_batches(files, opts) do
    case Streaming.estimate_total_examples(files, opts) do
      {:ok, total} -> div(total, opts[:batch_size] || 32)
      _ -> div(length(files) * 3000, opts[:batch_size] || 32)
    end
  end

  # Data visibility: what the model actually trains on. The July-2026 sweeps
  # were run without knowing the character mix ("did it even train on Fox?" —
  # nobody could answer). Logs the trained player's character distribution
  # and the opponent mix, sorted by share.
  defp log_character_distribution(frames) do
    alias ExPhil.Training.CharacterBalance

    total = length(frames)
    self_counts = CharacterBalance.count_characters(frames)

    opp_counts =
      frames
      |> Enum.map(fn frame ->
        players = (frame[:game_state] && frame.game_state.players) || %{}
        # extract_character reads port 0/1 (self); the opponent is the other
        opp = Map.get(players, 2) || Map.get(players, 1)
        CharacterBalance.extract_character(%{game_state: %{players: %{1 => opp}}})
      end)
      |> Enum.reject(&is_nil/1)
      |> Enum.frequencies()

    fmt = fn counts ->
      counts
      |> Enum.sort_by(fn {_, n} -> -n end)
      |> Enum.take(8)
      |> Enum.map_join(", ", fn {char, n} ->
        "#{char} #{Float.round(n / max(total, 1) * 100, 1)}%"
      end)
    end

    Output.puts("  Trained player: #{fmt.(self_counts)}")
    Output.puts("  Opponents:      #{fmt.(opp_counts)}")
  rescue
    # Diagnostics must never kill training
    e -> Output.puts("  (character distribution unavailable: #{Exception.message(e)})")
  end
end
