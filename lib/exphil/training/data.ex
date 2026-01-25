defmodule ExPhil.Training.Data do
  @moduledoc """
  Data loading and batching utilities for training.

  Provides utilities for loading parsed replay data, creating batches,
  and iterating through datasets for both imitation learning and RL.

  ## Data Format

  Parsed replay data is expected to be in the format:
  ```
  %{
    frames: [
      %{
        game_state: %GameState{...},
        controller: %ControllerState{...}  # For imitation learning
      },
      ...
    ],
    metadata: %{
      player_port: 1,
      character: :mewtwo,
      opponent_port: 2,
      ...
    }
  }
  ```

  ## Usage

      # Load dataset from directory
      {:ok, dataset} = Data.load_dataset("replays/parsed/")

      # Create batched iterator
      batches = Data.batched(dataset, batch_size: 64, shuffle: true)

      # Iterate
      for batch <- batches do
        # batch.states: [batch_size, embed_size]
        # batch.actions: %{buttons: ..., main_x: ..., etc.}
      end
  """

  alias ExPhil.Embeddings
  alias ExPhil.Bridge.{GameState, ControllerState}
  alias ExPhil.Training.PlayerRegistry

  require Logger

  defstruct [
    :frames,              # List of frame data
    :metadata,            # Dataset metadata
    :embed_config,        # Embedding configuration
    :size,                # Number of frames
    :embedded_sequences,  # Pre-computed embeddings (optional, for temporal training)
    :embedded_frames,     # Pre-computed single-frame embeddings (optional, for MLP training)
    :player_registry      # Player tag -> ID mapping for style-conditional training
  ]

  @type frame :: %{
    game_state: GameState.t(),
    controller: ControllerState.t() | nil,
    action: map() | nil,
    player_tag: String.t() | nil,
    name_id: non_neg_integer() | nil
  }

  @type t :: %__MODULE__{
    frames: [frame()],
    metadata: map(),
    embed_config: map(),
    size: non_neg_integer(),
    player_registry: PlayerRegistry.t() | nil
  }

  # ============================================================================
  # Dataset Loading
  # ============================================================================

  @doc """
  Load dataset from a directory of parsed replay files.

  ## Options
    - `:embed_config` - Embedding configuration (default: standard config)
    - `:player_port` - Filter to specific player port
    - `:character` - Filter to specific character
    - `:max_files` - Maximum number of files to load
  """
  @spec load_dataset(Path.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def load_dataset(path, opts \\ []) do
    embed_config = Keyword.get_lazy(opts, :embed_config, fn ->
      Embeddings.config()
    end)

    try do
      frames = path
      |> list_replay_files()
      |> maybe_limit(Keyword.get(opts, :max_files))
      |> Enum.flat_map(&load_replay_file(&1, opts))
      |> filter_frames(opts)

      dataset = %__MODULE__{
        frames: frames,
        metadata: %{
          source_path: path,
          filters: Keyword.take(opts, [:player_port, :character])
        },
        embed_config: embed_config,
        size: length(frames)
      }

      Logger.info("Loaded dataset with #{dataset.size} frames from #{path}")
      {:ok, dataset}
    rescue
      e -> {:error, e}
    end
  end

  @doc """
  Create a dataset from a list of frames directly.

  Useful for testing or when data is already in memory.

  ## Options
    - `:embed_config` - Embedding configuration
    - `:metadata` - Dataset metadata
    - `:player_registry` - PlayerRegistry for style-conditional training
  """
  @spec from_frames([frame()], keyword()) :: t()
  def from_frames(frames, opts \\ []) do
    embed_config = Keyword.get_lazy(opts, :embed_config, fn ->
      Embeddings.config()
    end)

    player_registry = Keyword.get(opts, :player_registry)

    # Add name_id to frames if registry is provided
    frames =
      if player_registry do
        Enum.map(frames, fn frame ->
          player_tag = frame[:player_tag]
          name_id = PlayerRegistry.get_id(player_registry, player_tag)
          Map.put(frame, :name_id, name_id)
        end)
      else
        frames
      end

    %__MODULE__{
      frames: frames,
      metadata: Keyword.get(opts, :metadata, %{}),
      embed_config: embed_config,
      size: length(frames),
      player_registry: player_registry
    }
  end

  defp list_replay_files(path) do
    case File.ls(path) do
      {:ok, files} ->
        files
        |> Enum.filter(&String.ends_with?(&1, [".parsed", ".bin", ".term"]))
        |> Enum.map(&Path.join(path, &1))
        |> Enum.sort()

      {:error, reason} ->
        Logger.warning("Failed to list directory #{path}: #{reason}")
        []
    end
  end

  defp maybe_limit(files, nil), do: files
  defp maybe_limit(files, max), do: Enum.take(files, max)

  defp load_replay_file(path, _opts) do
    case File.read(path) do
      {:ok, binary} ->
        try do
          data = :erlang.binary_to_term(binary)
          Map.get(data, :frames, [])
        rescue
          _ ->
            Logger.warning("Failed to parse #{path}")
            []
        end

      {:error, reason} ->
        Logger.warning("Failed to read #{path}: #{reason}")
        []
    end
  end

  defp filter_frames(frames, opts) do
    frames
    |> maybe_filter_port(Keyword.get(opts, :player_port))
    |> maybe_filter_character(Keyword.get(opts, :character))
    |> Enum.filter(&valid_frame?/1)
  end

  defp maybe_filter_port(frames, nil), do: frames
  defp maybe_filter_port(frames, port) do
    Enum.filter(frames, fn frame ->
      get_in(frame, [:metadata, :player_port]) == port
    end)
  end

  defp maybe_filter_character(frames, nil), do: frames
  defp maybe_filter_character(frames, character) do
    Enum.filter(frames, fn frame ->
      get_in(frame, [:metadata, :character]) == character
    end)
  end

  defp valid_frame?(frame) do
    Map.has_key?(frame, :game_state) and
    (Map.has_key?(frame, :controller) or Map.has_key?(frame, :action))
  end

  # ============================================================================
  # Batching
  # ============================================================================

  @doc """
  Create a batched iterator over the dataset.

  ## Options
    - `:batch_size` - Batch size (default: 64)
    - `:shuffle` - Whether to shuffle (default: true)
    - `:drop_last` - Drop incomplete final batch (default: false)
    - `:frame_delay` - Fixed frames between state and action (default: 0)
    - `:frame_delay_augment` - Enable variable frame delay (default: false)
    - `:frame_delay_min` - Minimum delay when augmenting (default: 0)
    - `:frame_delay_max` - Maximum delay when augmenting (default: 18)
    - `:seed` - Random seed for shuffling

  ## Frame Delay Augmentation

  When `frame_delay_augment: true`, each sample randomly uses a delay
  between `frame_delay_min` and `frame_delay_max`. This trains models
  to be robust to both local play (0 delay) and online play (18+ delay).

  ## Returns
    Stream of batches, each containing:
    - `states`: Tensor [batch_size, embed_size]
    - `actions`: Map of action tensors
  """
  @spec batched(t(), keyword()) :: Enumerable.t()
  def batched(dataset, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 64)
    shuffle = Keyword.get(opts, :shuffle, true)
    drop_last = Keyword.get(opts, :drop_last, false)
    frame_delay = Keyword.get(opts, :frame_delay, 0)
    frame_delay_augment = Keyword.get(opts, :frame_delay_augment, false)
    frame_delay_min = Keyword.get(opts, :frame_delay_min, 0)
    frame_delay_max = Keyword.get(opts, :frame_delay_max, 18)
    seed = Keyword.get(opts, :seed, System.system_time())
    augment_fn = Keyword.get(opts, :augment_fn, nil)
    character_weights = Keyword.get(opts, :character_weights, nil)

    # Determine max delay for index bounds
    max_delay = if frame_delay_augment, do: frame_delay_max, else: frame_delay

    # Prepare indices (leave room for max possible delay)
    valid_indices = 0..(dataset.size - 1 - max_delay) |> Enum.to_list()

    # Seed random number generator
    :rand.seed(:exsss, {seed, seed, seed})

    indices = cond do
      # Character-balanced sampling (weighted by inverse frequency)
      character_weights != nil ->
        alias ExPhil.Training.CharacterBalance
        # Get weights only for valid indices
        frame_weights = valid_indices
        |> Enum.map(fn idx -> Enum.at(dataset.frames, idx) end)
        |> CharacterBalance.frame_weights(character_weights)

        # Weighted sampling with replacement to balance characters
        CharacterBalance.balanced_indices(frame_weights, length(valid_indices))

      # Standard shuffle
      shuffle ->
        Enum.shuffle(valid_indices)

      # No shuffle
      true ->
        valid_indices
    end

    # Build delay config
    delay_config = %{
      augment: frame_delay_augment,
      fixed: frame_delay,
      min: frame_delay_min,
      max: frame_delay_max
    }

    # Create batch stream
    indices
    |> Enum.chunk_every(batch_size)
    |> maybe_drop_last(drop_last, batch_size)
    |> Stream.map(fn batch_indices ->
      create_batch(dataset, batch_indices, delay_config, augment_fn)
    end)
  end

  defp maybe_drop_last(chunks, false, _batch_size), do: chunks
  defp maybe_drop_last(chunks, true, batch_size) do
    Enum.filter(chunks, &(length(&1) == batch_size))
  end

  defp create_batch(dataset, indices, delay_config, augment_fn) do
    # Check if we have precomputed embeddings (and no augmentation that modifies states)
    use_precomputed = dataset.embedded_frames != nil and augment_fn == nil

    if use_precomputed do
      create_batch_precomputed(dataset, indices, delay_config)
    else
      create_batch_standard(dataset, indices, delay_config, augment_fn)
    end
  end

  # Fast path: use precomputed embeddings
  defp create_batch_precomputed(dataset, indices, delay_config) do
    # Collect precomputed embeddings and actions
    {embeddings, actions} = indices
    |> Enum.map(fn idx ->
      frame_delay = get_frame_delay(delay_config)
      action_frame = Enum.at(dataset.frames, idx + frame_delay)

      # Get precomputed embedding for state frame
      embedding = :array.get(idx, dataset.embedded_frames)
      action = get_action(action_frame)

      {embedding, action}
    end)
    |> Enum.unzip()

    # Stack precomputed embeddings
    states = Nx.stack(embeddings)

    # Convert actions to tensors
    action_tensors = actions_to_tensors(actions)

    %{
      states: states,
      actions: action_tensors
    }
  end

  # Standard path: embed on the fly (used when augmentation is enabled)
  defp create_batch_standard(dataset, indices, delay_config, augment_fn) do
    # Collect frames with name_ids for style-conditional training
    frame_data = Enum.map(indices, fn idx ->
      # Determine frame delay for this sample
      frame_delay = get_frame_delay(delay_config)

      state_frame = Enum.at(dataset.frames, idx)
      action_frame = Enum.at(dataset.frames, idx + frame_delay)

      # Apply augmentation if provided
      {state_frame, action_frame} = if augment_fn do
        augmented = augment_fn.(%{
          game_state: state_frame.game_state,
          controller: action_frame[:controller] || action_frame[:action]
        })
        {%{state_frame | game_state: augmented.game_state},
         Map.put(action_frame, :controller, augmented[:controller])}
      else
        {state_frame, action_frame}
      end

      # Include name_id for player embedding (defaults to 0 if not available)
      name_id = state_frame[:name_id] || 0
      {state_frame.game_state, get_action(action_frame), name_id}
    end)

    # Embed states with name_ids
    {game_states, actions, name_ids} = unzip3(frame_data)
    states = embed_states(game_states, dataset.embed_config, name_ids)

    # Convert actions to tensors
    action_tensors = actions_to_tensors(actions)

    %{
      states: states,
      actions: action_tensors
    }
  end

  defp get_action(frame) do
    cond do
      Map.has_key?(frame, :action) -> frame.action
      Map.has_key?(frame, :controller) -> controller_to_action(frame.controller)
      true -> neutral_action()
    end
  end

  # Unzip a list of 3-tuples into three lists
  defp unzip3(list) do
    Enum.reduce(Enum.reverse(list), {[], [], []}, fn {a, b, c}, {as, bs, cs} ->
      {[a | as], [b | bs], [c | cs]}
    end)
  end

  # Get frame delay for this sample - either fixed or random based on config
  defp get_frame_delay(%{augment: true, min: min_delay, max: max_delay}) do
    # Random delay uniformly distributed between min and max
    min_delay + :rand.uniform(max_delay - min_delay + 1) - 1
  end
  defp get_frame_delay(%{augment: false, fixed: fixed_delay}) do
    fixed_delay
  end
  # Handle legacy integer delay (backward compatibility)
  defp get_frame_delay(delay) when is_integer(delay), do: delay

  # Embed states with optional name_ids for style-conditional training
  defp embed_states(game_states, embed_config, nil) do
    embeddings = Enum.map(game_states, fn gs ->
      Embeddings.embed(gs, nil, embed_config: embed_config)
    end)

    Nx.stack(embeddings)
  end

  defp embed_states(game_states, embed_config, name_ids) when is_list(name_ids) do
    embeddings =
      Enum.zip(game_states, name_ids)
      |> Enum.map(fn {gs, name_id} ->
        # Pass name_id for player embedding (style-conditional training)
        Embeddings.embed(gs, nil, embed_config: embed_config, name_id: name_id)
      end)

    Nx.stack(embeddings)
  end

  # ============================================================================
  # Action Conversion
  # ============================================================================

  @doc """
  Convert controller state to action dictionary.

  Actions are discretized for the policy network:
  - Buttons: boolean for each button
  - Sticks: bucket index (0 to axis_buckets-1)
  - Shoulder: bucket index (0 to shoulder_buckets-1)
  """
  @spec controller_to_action(ControllerState.t(), keyword()) :: map()
  def controller_to_action(controller, opts \\ []) do
    axis_buckets = Keyword.get(opts, :axis_buckets, 16)
    shoulder_buckets = Keyword.get(opts, :shoulder_buckets, 4)

    %{
      buttons: %{
        a: controller.button_a,
        b: controller.button_b,
        x: controller.button_x,
        y: controller.button_y,
        z: controller.button_z,
        l: controller.button_l,
        r: controller.button_r,
        d_up: controller.button_d_up
      },
      main_x: discretize_axis(controller.main_stick.x, axis_buckets),
      main_y: discretize_axis(controller.main_stick.y, axis_buckets),
      c_x: discretize_axis(controller.c_stick.x, axis_buckets),
      c_y: discretize_axis(controller.c_stick.y, axis_buckets),
      shoulder: discretize_shoulder(
        Kernel.max(controller.l_shoulder, controller.r_shoulder),
        shoulder_buckets
      )
    }
  end

  defp discretize_axis(value, buckets) do
    # value is in [0, 1], convert to bucket index
    bucket = floor(value * buckets)
    Kernel.min(bucket, buckets - 1)
  end

  defp discretize_shoulder(value, buckets) do
    bucket = floor(value * buckets)
    Kernel.min(bucket, buckets - 1)
  end

  defp neutral_action do
    %{
      buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false},
      main_x: 8,  # Center bucket for 16 buckets
      main_y: 8,
      c_x: 8,
      c_y: 8,
      shoulder: 0
    }
  end

  @doc """
  Convert list of actions to batched tensors.
  """
  @spec actions_to_tensors([map()]) :: map()
  def actions_to_tensors(actions) do
    # Button tensor [batch, 8]
    buttons = actions
    |> Enum.map(fn a ->
      b = a.buttons
      [b.a, b.b, b.x, b.y, b.z, b.l, b.r, b.d_up]
      |> Enum.map(&if(&1, do: 1, else: 0))
    end)
    |> Nx.tensor(type: :s64)

    # Axis tensors [batch]
    main_x = Nx.tensor(Enum.map(actions, & &1.main_x), type: :s64)
    main_y = Nx.tensor(Enum.map(actions, & &1.main_y), type: :s64)
    c_x = Nx.tensor(Enum.map(actions, & &1.c_x), type: :s64)
    c_y = Nx.tensor(Enum.map(actions, & &1.c_y), type: :s64)
    shoulder = Nx.tensor(Enum.map(actions, & &1.shoulder), type: :s64)

    %{
      buttons: buttons,
      main_x: main_x,
      main_y: main_y,
      c_x: c_x,
      c_y: c_y,
      shoulder: shoulder
    }
  end

  # ============================================================================
  # Dataset Operations
  # ============================================================================

  @doc """
  Split dataset into train/validation sets.

  ## Options
    - `:ratio` - Train ratio (default: 0.9)
    - `:shuffle` - Shuffle before split (default: true)
  """
  @spec split(t(), keyword()) :: {t(), t()}
  def split(dataset, opts \\ []) do
    ratio = Keyword.get(opts, :ratio, 0.9)
    shuffle = Keyword.get(opts, :shuffle, true)

    frames = if shuffle do
      Enum.shuffle(dataset.frames)
    else
      dataset.frames
    end

    split_idx = floor(length(frames) * ratio)
    {train_frames, val_frames} = Enum.split(frames, split_idx)

    train = %{dataset |
      frames: train_frames,
      size: length(train_frames)
    }

    val = %{dataset |
      frames: val_frames,
      size: length(val_frames)
    }

    {train, val}
  end

  @doc """
  Create an empty dataset with the same configuration as the given dataset.

  Useful when validation split is disabled (val_split = 0.0) but code
  still expects a validation dataset struct.
  """
  @spec empty(t()) :: t()
  def empty(dataset) do
    %{dataset |
      frames: [],
      size: 0,
      embedded_sequences: if(dataset.embedded_sequences, do: :array.new(), else: nil)
    }
  end

  @doc """
  Concatenate multiple datasets.
  """
  @spec concat([t()]) :: t()
  def concat(datasets) do
    [first | _rest] = datasets

    all_frames = Enum.flat_map(datasets, & &1.frames)

    %__MODULE__{
      frames: all_frames,
      metadata: %{sources: Enum.map(datasets, & &1.metadata)},
      embed_config: first.embed_config,
      size: length(all_frames)
    }
  end

  @doc """
  Sample a subset of the dataset.
  """
  @spec sample(t(), non_neg_integer()) :: t()
  def sample(dataset, n) do
    sampled_frames = dataset.frames
    |> Enum.shuffle()
    |> Enum.take(n)

    %{dataset |
      frames: sampled_frames,
      size: length(sampled_frames)
    }
  end

  # ============================================================================
  # Temporal/Sequence Data
  # ============================================================================

  @doc """
  Convert dataset to sequences for temporal training.

  Creates overlapping windows of consecutive frames. Each sequence contains
  `window_size` frames, and the action is from the last frame.

  ## Options
    - `:window_size` - Number of frames per sequence (default: 60)
    - `:stride` - Step between sequences (default: 1)

  ## Returns
    New dataset where each "frame" is actually a sequence.
  """
  @spec to_sequences(t(), keyword()) :: t()
  def to_sequences(dataset, opts \\ []) do
    window_size = Keyword.get(opts, :window_size, 60)
    stride = Keyword.get(opts, :stride, 1)

    # Need at least window_size frames
    if dataset.size < window_size do
      Logger.warning("Dataset size #{dataset.size} < window_size #{window_size}, returning empty")
      %{dataset | frames: [], size: 0}
    else
      # Create sequences
      sequences = create_sequences(dataset.frames, window_size, stride)

      %{dataset |
        frames: sequences,
        size: length(sequences),
        metadata: Map.merge(dataset.metadata, %{
          temporal: true,
          window_size: window_size,
          stride: stride,
          original_size: dataset.size
        })
      }
    end
  end

  defp create_sequences(frames, window_size, stride) do
    frames_array = :array.from_list(frames)
    max_start = :array.size(frames_array) - window_size

    0..max_start//stride
    |> Enum.map(fn start_idx ->
      # Get window of frames
      window_frames = for i <- start_idx..(start_idx + window_size - 1) do
        :array.get(i, frames_array)
      end

      # The action comes from the last frame
      last_frame = List.last(window_frames)

      %{
        sequence: window_frames,
        game_state: last_frame.game_state,  # Keep for compatibility
        controller: Map.get(last_frame, :controller),
        action: get_action(last_frame)
      }
    end)
  end

  @doc """
  Pre-compute embeddings for all sequences in the dataset.

  This significantly speeds up temporal training by embedding frames once
  instead of on every batch. Call this after `to_sequences/2` and before training.

  ## Example

      dataset
      |> Data.to_sequences(window_size: 30)
      |> Data.precompute_embeddings()

  ## Options
    - `:show_progress` - Show embedding progress (default: true)
  """
  @spec precompute_embeddings(t(), keyword()) :: t()
  def precompute_embeddings(dataset, opts \\ []) do
    show_progress = Keyword.get(opts, :show_progress, true)
    gc_every = Keyword.get(opts, :gc_every, 100)  # GC every N chunks

    total = dataset.size

    # Warn if dataset is very large - recommend streaming instead
    if total > 500_000 and show_progress do
      IO.puts(:stderr, "  ⚠ Large dataset (#{total} sequences) - consider using --stream-chunk-size")
      IO.puts(:stderr, "    Pre-computing will use ~#{Float.round(total * 60 * 400 * 4 / 1_000_000_000, 1)}GB RAM")
    end

    if show_progress do
      IO.puts(:stderr, "  Pre-computing embeddings for #{total} sequences...")
    end

    embed_config = dataset.embed_config

    # Process in chunks to show progress and manage memory
    chunk_size = min(500, max(1, div(total, 10)))

    embedded = dataset.frames
    |> Enum.chunk_every(chunk_size)
    |> Enum.with_index()
    |> Enum.flat_map(fn {chunk, chunk_idx} ->
      # Show progress
      if show_progress do
        processed = min((chunk_idx + 1) * chunk_size, total)
        pct = round(processed / total * 100)
        IO.write(:stderr, "\r  Embedding: #{pct}% (#{processed}/#{total})    ")
      end

      # Periodic garbage collection to prevent memory buildup
      if gc_every > 0 and rem(chunk_idx, gc_every) == 0 and chunk_idx > 0 do
        :erlang.garbage_collect()
      end

      # Batch embed each sequence in this chunk
      # For each sequence, we batch all frames together
      # CRITICAL: Copy to CPU after embedding to avoid GPU OOM
      # (175K sequences × 30 frames × 1204 dims × 4 bytes = 25GB, exceeds GPU memory)
      Enum.map(chunk, fn frame ->
        game_states = Enum.map(frame.sequence, & &1.game_state)
        # Use batch embedding: all frames at once, then copy to CPU
        Embeddings.Game.embed_states_fast(game_states, 1, config: embed_config)
        |> Nx.backend_copy(Nx.BinaryBackend)
      end)
    end)

    if show_progress do
      IO.puts(:stderr, "\r  Embedding: 100% (#{total}/#{total}) - done!    ")
    end

    %{dataset | embedded_sequences: embedded}
  end

  @doc """
  Pre-compute embeddings for all frames in the dataset (single-frame MLP training).

  This significantly speeds up MLP training by embedding frames once
  instead of on every batch. Call this before training.

  ## Example

      dataset
      |> Data.precompute_frame_embeddings()

  ## Options
    - `:show_progress` - Show embedding progress (default: true)
    - `:batch_size` - Embedding batch size for GPU efficiency (default: 1000)
  """
  @spec precompute_frame_embeddings(t(), keyword()) :: t()
  def precompute_frame_embeddings(dataset, opts \\ []) do
    show_progress = Keyword.get(opts, :show_progress, true)
    batch_size = Keyword.get(opts, :batch_size, 1000)
    gc_every = Keyword.get(opts, :gc_every, 100)  # GC every N batches

    total = dataset.size

    # Warn if dataset is very large - recommend streaming instead
    if total > 1_000_000 and show_progress do
      IO.puts(:stderr, "  ⚠ Large dataset (#{total} frames) - consider using --stream-chunk-size")
      IO.puts(:stderr, "    Pre-computing will use ~#{Float.round(total * 400 * 4 / 1_000_000_000, 1)}GB RAM")
    end

    if show_progress do
      IO.puts(:stderr, "  Pre-computing embeddings for #{total} frames...")
    end

    embed_config = dataset.embed_config

    # Process in batches for GPU efficiency and memory management
    embedded = dataset.frames
    |> Enum.chunk_every(batch_size)
    |> Enum.with_index()
    |> Enum.flat_map(fn {chunk, chunk_idx} ->
      # Show progress
      if show_progress do
        processed = min((chunk_idx + 1) * batch_size, total)
        pct = round(processed / total * 100)
        IO.write(:stderr, "\r  Embedding: #{pct}% (#{processed}/#{total})    ")
      end

      # Periodic garbage collection to prevent memory buildup
      if gc_every > 0 and rem(chunk_idx, gc_every) == 0 and chunk_idx > 0 do
        :erlang.garbage_collect()
      end

      # Batch embed all frames in this chunk with name_ids for style-conditional training
      game_states = Enum.map(chunk, & &1.game_state)
      name_ids = Enum.map(chunk, fn f -> f[:name_id] || 0 end)
      batch_embedded = Embeddings.Game.embed_states_fast(game_states, 1,
        config: embed_config, name_id: name_ids)

      # Convert to list of individual embeddings
      # CRITICAL: Copy to CPU after embedding to avoid GPU OOM when dataset is large
      batch_embedded
      |> Nx.to_batched(1)
      |> Enum.map(fn t ->
        t |> Nx.squeeze(axes: [0]) |> Nx.backend_copy(Nx.BinaryBackend)
      end)
    end)

    if show_progress do
      IO.puts(:stderr, "\r  Embedding: 100% (#{total}/#{total}) - done!    ")
    end

    # Convert to array for O(1) access during batching
    embedded_array = :array.from_list(embedded)

    %{dataset | embedded_frames: embedded_array}
  end

  @doc """
  Check if dataset has precomputed frame embeddings.
  """
  def has_precomputed_embeddings?(%__MODULE__{embedded_frames: nil}), do: false
  def has_precomputed_embeddings?(%__MODULE__{embedded_frames: _}), do: true

  @doc """
  Create batched iterator for sequence data (temporal training).

  Similar to `batched/2` but handles sequences properly.

  ## Options
    - `:batch_size` - Batch size (default: 64)
    - `:shuffle` - Whether to shuffle (default: true)
    - `:drop_last` - Drop incomplete final batch (default: false)

  ## Returns
    Stream of batches containing:
    - `states`: Tensor [batch_size, seq_len, embed_size]
    - `actions`: Map of action tensors
  """
  @spec batched_sequences(t(), keyword()) :: Enumerable.t()
  def batched_sequences(dataset, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 64)
    shuffle = Keyword.get(opts, :shuffle, true)
    drop_last = Keyword.get(opts, :drop_last, false)
    seed = Keyword.get(opts, :seed, System.system_time())
    character_weights = Keyword.get(opts, :character_weights, nil)

    # Convert lists to arrays for O(1) index access (vs O(n) for lists)
    frames_array = :array.from_list(dataset.frames)
    embeddings_array = if dataset.embedded_sequences do
      :array.from_list(dataset.embedded_sequences)
    else
      nil
    end

    # Prepare indices
    valid_indices = 0..(dataset.size - 1) |> Enum.to_list()

    # Seed random number generator
    :rand.seed(:exsss, {seed, seed, seed})

    indices = cond do
      # Character-balanced sampling (weighted by inverse frequency)
      character_weights != nil ->
        alias ExPhil.Training.CharacterBalance
        # Get weights for all sequences
        frame_weights = CharacterBalance.frame_weights(dataset.frames, character_weights)
        # Weighted sampling with replacement
        CharacterBalance.balanced_indices(frame_weights, length(valid_indices))

      # Standard shuffle
      shuffle ->
        Enum.shuffle(valid_indices)

      # No shuffle
      true ->
        valid_indices
    end

    # Create batch stream with array-backed lookup
    indices
    |> Enum.chunk_every(batch_size)
    |> maybe_drop_last(drop_last, batch_size)
    |> Stream.map(fn batch_indices ->
      create_sequence_batch_fast(frames_array, embeddings_array, batch_indices)
    end)
  end

  # Fast batch creation using arrays for O(1) lookup
  defp create_sequence_batch_fast(frames_array, embeddings_array, indices) when embeddings_array != nil do
    # Fast path: use pre-computed embeddings with O(1) array access
    batch_data = Enum.map(indices, fn idx ->
      frame = :array.get(idx, frames_array)
      embedding = :array.get(idx, embeddings_array)
      {embedding, frame.action}
    end)

    {embeddings, actions} = Enum.unzip(batch_data)

    # Stack embeddings: [batch, seq_len, embed_size]
    states = Nx.stack(embeddings)

    # Convert actions to tensors
    action_tensors = actions_to_tensors(actions)

    %{
      states: states,
      actions: action_tensors
    }
  end

  defp create_sequence_batch_fast(frames_array, nil, indices) do
    # Slow path: no pre-computed embeddings, embed on-the-fly
    sequence_data = Enum.map(indices, fn idx ->
      frame = :array.get(idx, frames_array)
      # Extract name_id from first frame of sequence for style-conditional training
      name_id = get_in(frame, [:sequence, Access.at(0), :name_id]) || 0
      {frame.sequence, frame.action, name_id}
    end)

    {sequences, actions, name_ids} = unzip3(sequence_data)

    # Embed sequences with name_ids
    embedded = Enum.zip(sequences, name_ids)
    |> Enum.map(fn {seq, name_id} ->
      game_states = Enum.map(seq, & &1.game_state)
      Embeddings.Game.embed_states_fast(game_states, 1, name_id: name_id)
    end)

    states = Nx.stack(embedded)
    action_tensors = actions_to_tensors(actions)

    %{
      states: states,
      actions: action_tensors
    }
  end

  # ============================================================================
  # Statistics
  # ============================================================================

  @doc """
  Compute dataset statistics.
  """
  @spec stats(t()) :: map()
  def stats(dataset) do
    actions = Enum.map(dataset.frames, &get_action/1)

    # Button press rates
    button_counts = Enum.reduce(actions, %{}, fn action, acc ->
      Enum.reduce(action.buttons, acc, fn {button, pressed}, inner_acc ->
        if pressed do
          Map.update(inner_acc, button, 1, &(&1 + 1))
        else
          inner_acc
        end
      end)
    end)

    button_rates = Map.new(button_counts, fn {button, count} ->
      {button, count / dataset.size}
    end)

    # Stick distributions
    main_x_dist = distribution(Enum.map(actions, & &1.main_x), 16)
    main_y_dist = distribution(Enum.map(actions, & &1.main_y), 16)

    %{
      size: dataset.size,
      button_rates: button_rates,
      main_x_distribution: main_x_dist,
      main_y_distribution: main_y_dist
    }
  end

  defp distribution(values, num_buckets) do
    counts = Enum.frequencies(values)

    0..(num_buckets - 1)
    |> Enum.map(fn bucket ->
      Map.get(counts, bucket, 0)
    end)
  end
end
