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
    # List of frame data
    :frames,
    # Dataset metadata
    :metadata,
    # Embedding configuration
    :embed_config,
    # Number of frames
    :size,
    # Pre-computed embeddings (optional, for temporal training)
    :embedded_sequences,
    # Pre-computed single-frame embeddings (optional, for MLP training)
    :embedded_frames,
    # Player tag -> ID mapping for style-conditional training
    :player_registry
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
    embed_config =
      Keyword.get_lazy(opts, :embed_config, fn ->
        Embeddings.config()
      end)

    try do
      frames =
        path
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
    embed_config =
      Keyword.get_lazy(opts, :embed_config, fn ->
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
    - `:mirror_prob` - Mirror augmentation probability for augmented caches (default: 0.5)
    - `:noise_prob` - Noise augmentation probability for augmented caches (default: 0.3)
    - `:num_noisy_variants` - Number of noisy variants in augmented cache (default: 2)

  ## Frame Delay Augmentation

  When `frame_delay_augment: true`, each sample randomly uses a delay
  between `frame_delay_min` and `frame_delay_max`. This trains models
  to be robust to both local play (0 delay) and online play (18+ delay).

  ## Augmented Embedding Support

  If the dataset has augmented embeddings (from `precompute_augmented_frame_embeddings`),
  and `augment_fn` is NOT provided, variant selection will be used automatically.
  This provides ~100x speedup over on-the-fly augmentation.

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

    # Augmentation options for augmented embedding caches
    mirror_prob = Keyword.get(opts, :mirror_prob, 0.5)
    noise_prob = Keyword.get(opts, :noise_prob, 0.3)
    num_noisy_variants = Keyword.get(opts, :num_noisy_variants, 2)

    # Determine max delay for index bounds
    max_delay = if frame_delay_augment, do: frame_delay_max, else: frame_delay

    # Prepare indices (leave room for max possible delay)
    valid_indices = 0..(dataset.size - 1 - max_delay) |> Enum.to_list()

    # Seed random number generator
    :rand.seed(:exsss, {seed, seed, seed})

    indices =
      cond do
        # Character-balanced sampling (weighted by inverse frequency)
        character_weights != nil ->
          alias ExPhil.Training.CharacterBalance
          # Get weights only for valid indices
          frame_weights =
            valid_indices
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

    # Build augmentation config for variant selection
    augment_config = %{
      mirror_prob: mirror_prob,
      noise_prob: noise_prob,
      num_noisy_variants: num_noisy_variants
    }

    # Create batch stream
    indices
    |> Enum.chunk_every(batch_size)
    |> maybe_drop_last(drop_last, batch_size)
    |> Stream.map(fn batch_indices ->
      create_batch(dataset, batch_indices, delay_config, augment_fn, augment_config)
    end)
  end

  defp maybe_drop_last(chunks, false, _batch_size), do: chunks

  defp maybe_drop_last(chunks, true, batch_size) do
    Enum.filter(chunks, &(length(&1) == batch_size))
  end

  defp create_batch(dataset, indices, delay_config, augment_fn, augment_config) do
    # Check if we have precomputed embeddings
    has_precomputed = dataset.embedded_frames != nil

    # Check if we have augmented embeddings (3D tensor)
    has_augmented = has_precomputed and has_augmented_embeddings?(dataset)

    cond do
      # Fast path: augmented embeddings with variant selection
      # Use this when we have augmented embeddings and no custom augment_fn
      has_augmented and augment_fn == nil ->
        create_batch_augmented(dataset, indices, delay_config, augment_config)

      # Fast path: regular precomputed embeddings (no augmentation)
      has_precomputed and augment_fn == nil ->
        create_batch_precomputed(dataset, indices, delay_config)

      # Slow path: on-the-fly embedding (with optional augmentation)
      true ->
        create_batch_standard(dataset, indices, delay_config, augment_fn)
    end
  end

  # Fast path: use precomputed embeddings (stacked tensor format)
  # Performance: O(1) slice vs O(batch_size) stack of individual tensors
  defp create_batch_precomputed(dataset, indices, delay_config) do
    # Collect actions for each frame (still need frame delay logic)
    actions =
      Enum.map(indices, fn idx ->
        frame_delay = get_frame_delay(delay_config)
        action_frame = Enum.at(dataset.frames, idx + frame_delay)
        get_action(action_frame)
      end)

    # Get embeddings using efficient tensor indexing
    # embedded_frames is either:
    #   - Stacked tensor {num_frames, embed_size} (new format, fast)
    #   - Erlang array of individual tensors (legacy format, slow fallback)
    states =
      case dataset.embedded_frames do
        tensor when is_struct(tensor, Nx.Tensor) ->
          # NEW: Fast path - gather by indices from stacked tensor
          # This is O(1) on GPU vs O(batch_size) for stacking individual tensors
          indices_tensor = Nx.tensor(indices, type: :s64)
          Nx.take(tensor, indices_tensor, axis: 0)
          |> Nx.backend_transfer(EXLA.Backend)

        array when is_tuple(array) and elem(array, 0) == :array ->
          # LEGACY: Slow path - stack individual tensors (for backwards compatibility)
          embeddings = Enum.map(indices, fn idx -> :array.get(idx, array) end)
          embeddings
          |> Nx.stack()
          |> Nx.backend_transfer(EXLA.Backend)
      end

    # Convert actions to tensors and transfer to GPU
    action_tensors =
      actions_to_tensors(actions)
      |> transfer_actions_to_gpu()

    %{
      states: states,
      actions: action_tensors
    }
  end

  # Fast path: use augmented precomputed embeddings with variant selection
  # embedded_frames shape: {num_frames, num_variants, embed_size}
  # Variant layout: 0=original, 1=mirrored, 2+=noisy
  defp create_batch_augmented(dataset, indices, delay_config, augment_config) do
    %{
      mirror_prob: mirror_prob,
      noise_prob: noise_prob,
      num_noisy_variants: num_noisy_variants
    } = augment_config

    # Collect actions for each frame (still need frame delay logic)
    actions =
      Enum.map(indices, fn idx ->
        frame_delay = get_frame_delay(delay_config)
        action_frame = Enum.at(dataset.frames, idx + frame_delay)
        get_action(action_frame)
      end)

    # Select variant for each sample in batch
    # This provides stochasticity similar to on-the-fly augmentation
    variant_indices =
      Enum.map(indices, fn _idx ->
        select_variant_index(
          mirror_prob: mirror_prob,
          noise_prob: noise_prob,
          num_noisy_variants: num_noisy_variants
        )
      end)

    # Get embeddings using efficient 2D tensor indexing
    # embedded_frames is {num_frames, num_variants, embed_size}
    tensor = dataset.embedded_frames

    # Create indices for gather
    frame_indices = Nx.tensor(indices, type: :s64)
    variant_tensor = Nx.tensor(variant_indices, type: :s64)

    # Index into augmented embeddings
    # Two-step indexing: first select frames, then select variants per sample
    # Step 1: Nx.take on axis 0 gives us {batch_size, num_variants, embed_size}
    frames_all_variants = Nx.take(tensor, frame_indices, axis: 0)

    # Step 2: For each sample, select its variant
    states =
      frames_all_variants
      |> select_variants_from_batch(variant_tensor)
      |> Nx.backend_transfer(EXLA.Backend)

    # Convert actions to tensors and transfer to GPU
    action_tensors =
      actions_to_tensors(actions)
      |> transfer_actions_to_gpu()

    %{
      states: states,
      actions: action_tensors
    }
  end

  # Select one variant per sample from a batch of all variants
  # input: {batch_size, num_variants, embed_size}
  # variant_indices: {batch_size} - which variant to select for each sample
  # output: {batch_size, embed_size}
  defp select_variants_from_batch(batch_all_variants, variant_indices) do
    {batch_size, _num_variants, embed_size} = Nx.shape(batch_all_variants)

    # Build indices for gather: for each sample i, select [i, variant_indices[i], :]
    batch_indices = Nx.iota({batch_size, 1})

    # Combine into {batch_size, 2} index array: [[0, v0], [1, v1], ...]
    variant_indices_2d = Nx.reshape(variant_indices, {batch_size, 1})
    gather_indices = Nx.concatenate([batch_indices, variant_indices_2d], axis: 1)

    # Use Nx.gather to select elements
    Nx.gather(
      batch_all_variants,
      gather_indices,
      axes: [0, 1]
    )
    |> Nx.reshape({batch_size, embed_size})
  end

  # Transfer action tensors to GPU
  defp transfer_actions_to_gpu(actions) when is_map(actions) do
    Map.new(actions, fn {k, v} ->
      {k, Nx.backend_transfer(v, EXLA.Backend)}
    end)
  end

  # Standard path: embed on the fly (used when augmentation is enabled)
  defp create_batch_standard(dataset, indices, delay_config, augment_fn) do
    # Collect frames with name_ids for style-conditional training
    frame_data =
      Enum.map(indices, fn idx ->
        # Determine frame delay for this sample
        frame_delay = get_frame_delay(delay_config)

        state_frame = Enum.at(dataset.frames, idx)
        action_frame = Enum.at(dataset.frames, idx + frame_delay)

        # Apply augmentation if provided
        {state_frame, action_frame} =
          if augment_fn do
            augmented =
              augment_fn.(%{
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
    states = embed_states_parallel(game_states, dataset.embed_config, name_ids)

    # Convert actions to tensors and transfer to GPU
    action_tensors =
      actions_to_tensors(actions)
      |> transfer_actions_to_gpu()

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

  # Batch embedding for non-precompute path - uses vectorized Nx operations
  # instead of per-frame loops for much better performance
  #
  # Performance comparison for 512 frames:
  #   - Old (Task.async_stream): ~16s/batch (512 × 25 tensor ops = 12,800 ops)
  #   - New (embed_states_fast): ~0.5s/batch (~25 batch tensor ops total)
  #
  # Memory efficiency (batch_size=512, embed_size=408):
  #   - Old: 512 process mailboxes + 512 × 25 intermediate tensors ≈ 5-10MB peak
  #   - New: Elixir lists (~50KB) + batch tensors (~1MB) ≈ 1-2MB peak
  #
  # For typical batch sizes (32-512), this is more memory-efficient than
  # the per-frame approach because it avoids intermediate tensor allocations.
  @embed_chunk_size 1024

  defp embed_states_parallel(game_states, embed_config, nil) do
    if length(game_states) > @embed_chunk_size do
      # Chunk very large batches to limit peak memory usage
      game_states
      |> Enum.chunk_every(@embed_chunk_size)
      |> Enum.map(fn chunk ->
        Embeddings.Game.embed_states_fast(chunk, 1, config: embed_config)
      end)
      |> Nx.concatenate()
      |> Nx.backend_transfer(EXLA.Backend)
    else
      # Direct batch embedding for typical sizes
      Embeddings.Game.embed_states_fast(game_states, 1, config: embed_config)
      |> Nx.backend_transfer(EXLA.Backend)
    end
  end

  defp embed_states_parallel(game_states, embed_config, name_ids) when is_list(name_ids) do
    if length(game_states) > @embed_chunk_size do
      # Chunk very large batches with corresponding name_ids
      Enum.zip(game_states, name_ids)
      |> Enum.chunk_every(@embed_chunk_size)
      |> Enum.map(fn chunk ->
        {chunk_states, chunk_ids} = Enum.unzip(chunk)

        Embeddings.Game.embed_states_fast(chunk_states, 1,
          config: embed_config,
          name_id: chunk_ids
        )
      end)
      |> Nx.concatenate()
      |> Nx.backend_transfer(EXLA.Backend)
    else
      # Direct batch embedding for typical sizes
      Embeddings.Game.embed_states_fast(game_states, 1,
        config: embed_config,
        name_id: name_ids
      )
      |> Nx.backend_transfer(EXLA.Backend)
    end
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
      shoulder:
        discretize_shoulder(
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
      buttons: %{
        a: false,
        b: false,
        x: false,
        y: false,
        z: false,
        l: false,
        r: false,
        d_up: false
      },
      # Center bucket for 16 buckets
      main_x: 8,
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
    buttons =
      actions
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

    frames =
      if shuffle do
        Enum.shuffle(dataset.frames)
      else
        dataset.frames
      end

    split_idx = floor(length(frames) * ratio)
    {train_frames, val_frames} = Enum.split(frames, split_idx)

    train = %{dataset | frames: train_frames, size: length(train_frames)}

    val = %{dataset | frames: val_frames, size: length(val_frames)}

    {train, val}
  end

  @doc """
  Create an empty dataset with the same configuration as the given dataset.

  Useful when validation split is disabled (val_split = 0.0) but code
  still expects a validation dataset struct.
  """
  @spec empty(t()) :: t()
  def empty(dataset) do
    %{
      dataset
      | frames: [],
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
    sampled_frames =
      dataset.frames
      |> Enum.shuffle()
      |> Enum.take(n)

    %{dataset | frames: sampled_frames, size: length(sampled_frames)}
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

      %{
        dataset
        | frames: sequences,
          size: length(sequences),
          metadata:
            Map.merge(dataset.metadata, %{
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
      window_frames =
        for i <- start_idx..(start_idx + window_size - 1) do
          :array.get(i, frames_array)
        end

      # The action comes from the last frame
      last_frame = List.last(window_frames)

      %{
        sequence: window_frames,
        # Keep for compatibility
        game_state: last_frame.game_state,
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
    # GC every N chunks
    gc_every = Keyword.get(opts, :gc_every, 100)

    total = dataset.size

    # Warn if dataset is very large - recommend streaming instead
    if total > 500_000 and show_progress do
      IO.puts(
        :stderr,
        "  ⚠ Large dataset (#{total} sequences) - consider using --stream-chunk-size"
      )

      IO.puts(
        :stderr,
        "    Pre-computing will use ~#{Float.round(total * 60 * 400 * 4 / 1_000_000_000, 1)}GB RAM"
      )
    end

    if show_progress do
      IO.puts(:stderr, "  Pre-computing embeddings for #{total} sequences...")
    end

    embed_config = dataset.embed_config

    # Process in chunks to show progress and manage memory
    chunk_size = min(500, max(1, div(total, 10)))

    embedded =
      dataset.frames
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

    # Convert to array for O(1) access during batching and cache compatibility
    embedded_array = :array.from_list(embedded)

    %{dataset | embedded_sequences: embedded_array}
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
    # GC every N batches
    gc_every = Keyword.get(opts, :gc_every, 100)

    total = dataset.size

    # Warn if dataset is very large - recommend streaming instead
    if total > 1_000_000 and show_progress do
      IO.puts(:stderr, "  ⚠ Large dataset (#{total} frames) - consider using --stream-chunk-size")

      IO.puts(
        :stderr,
        "    Pre-computing will use ~#{Float.round(total * 400 * 4 / 1_000_000_000, 1)}GB RAM"
      )
    end

    if show_progress do
      IO.puts(:stderr, "  Pre-computing embeddings for #{total} frames...")
    end

    embed_config = dataset.embed_config

    # Process in batches for GPU efficiency and memory management
    # Collect batch tensors (each is {batch_size, embed_size})
    batch_tensors =
      dataset.frames
      |> Enum.chunk_every(batch_size)
      |> Enum.with_index()
      |> Enum.map(fn {chunk, chunk_idx} ->
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

        # Embed batch and copy to CPU to avoid GPU OOM
        # Returns tensor of shape {chunk_size, embed_size}
        Embeddings.Game.embed_states_fast(game_states, 1,
          config: embed_config,
          name_id: name_ids
        )
        |> Nx.backend_copy(Nx.BinaryBackend)
      end)

    if show_progress do
      IO.puts(:stderr, "\r  Embedding: 100% (#{total}/#{total}) - done!    ")
    end

    # Concatenate all batch tensors into a single tensor {num_frames, embed_size}
    # This enables O(1) batch creation via Nx.take() instead of O(batch_size) stacking
    #
    # Performance impact:
    #   OLD: 512 individual tensors → Nx.stack() = ~50 seconds/batch
    #   NEW: Single tensor → Nx.take() = ~50 milliseconds/batch (1000x faster)
    stacked_embeddings = Nx.concatenate(batch_tensors, axis: 0)

    %{dataset | embedded_frames: stacked_embeddings}
  end

  @doc """
  Check if dataset has precomputed frame embeddings.
  """
  def has_precomputed_embeddings?(%__MODULE__{embedded_frames: nil}), do: false
  def has_precomputed_embeddings?(%__MODULE__{embedded_frames: _}), do: true

  # ============================================================================
  # Embedding Cache Integration
  # ============================================================================

  @doc """
  Pre-compute frame embeddings with optional disk caching.

  If caching is enabled and a cached version exists, loads from disk.
  Otherwise computes embeddings and optionally saves to cache.

  ## Options
    - `:cache` - Enable caching (default: false)
    - `:cache_dir` - Cache directory (default: "cache/embeddings")
    - `:force_recompute` - Ignore cache and recompute (default: false)
    - `:replay_files` - List of replay file paths (required for cache key)
    - All options from `precompute_frame_embeddings/2`

  ## Example

      # With caching enabled
      dataset = Data.precompute_frame_embeddings_cached(dataset,
        cache: true,
        replay_files: replay_files
      )
  """
  @spec precompute_frame_embeddings_cached(t(), keyword()) :: t()
  def precompute_frame_embeddings_cached(dataset, opts \\ []) do
    cache_enabled = Keyword.get(opts, :cache, false)
    force_recompute = Keyword.get(opts, :force_recompute, false)
    replay_files = Keyword.get(opts, :replay_files, [])
    cache_dir = Keyword.get(opts, :cache_dir, "cache/embeddings")

    alias ExPhil.Training.EmbeddingCache

    if cache_enabled and not force_recompute and length(replay_files) > 0 do
      cache_key = EmbeddingCache.cache_key(dataset.embed_config, replay_files, temporal: false)

      Logger.info("[EmbeddingCache] Looking for cache key: #{cache_key} (frame embeddings)")

      case EmbeddingCache.load(cache_key, cache_dir: cache_dir) do
        {:ok, embedded_array} ->
          Logger.info("[EmbeddingCache] Using cached frame embeddings")
          %{dataset | embedded_frames: embedded_array}

        {:error, :not_found} ->
          available =
            EmbeddingCache.list(cache_dir: cache_dir)
            |> Enum.map(& &1.key)
            |> Enum.join(", ")

          Logger.info("[EmbeddingCache] Cache miss for #{cache_key}")
          Logger.info("[EmbeddingCache] Available caches: #{available}")
          result = precompute_frame_embeddings(dataset, opts)

          # Save to cache
          if result.embedded_frames do
            Logger.info("[EmbeddingCache] Saving frame embeddings to #{cache_key}...")
            case EmbeddingCache.save(cache_key, result.embedded_frames, cache_dir: cache_dir) do
              :ok ->
                Logger.info("[EmbeddingCache] Successfully saved frame cache")
              {:error, reason} ->
                Logger.error("[EmbeddingCache] Failed to save frame cache: #{inspect(reason)}")
            end
          else
            Logger.warning("[EmbeddingCache] No embedded_frames to save (this is a bug)")
          end

          result

        {:error, reason} ->
          Logger.warning(
            "[EmbeddingCache] Failed to load cache: #{inspect(reason)}, recomputing..."
          )

          precompute_frame_embeddings(dataset, opts)
      end
    else
      precompute_frame_embeddings(dataset, opts)
    end
  end

  # ============================================================================
  # Augmented Embedding Cache
  # ============================================================================

  @doc """
  Pre-compute frame embeddings with augmented variants.

  Creates multiple augmented versions of each frame and embeds them all,
  resulting in a tensor of shape `{num_frames, num_variants, embed_size}`.

  This enables ~100x speedup when using `--augment` during training, as
  augmentation is precomputed rather than applied per-batch.

  ## Variant Layout
    - Index 0: Original (unaugmented)
    - Index 1: Mirrored (X positions/velocities flipped)
    - Index 2+: Noisy variants (Gaussian noise with deterministic seeds)

  ## Options
    - `:num_noisy_variants` - Number of noisy variants to generate (default: 2)
    - `:noise_scale` - Standard deviation for noise (default: 0.01)
    - `:show_progress` - Show progress bar (default: true)
    - `:batch_size` - Embedding batch size for GPU efficiency (default: 500)
    - `:gc_every` - Run GC every N batches (default: 50)

  ## Returns

  Updated dataset with `embedded_frames` of shape `{num_frames, num_variants, embed_size}`

  ## Example

      # Precompute with 2 noisy variants (total 4 variants: original, mirror, noisy1, noisy2)
      dataset = Data.precompute_augmented_frame_embeddings(dataset,
        num_noisy_variants: 2,
        noise_scale: 0.01
      )

      # During training, randomly select variant per sample
      variant_idx = select_variant(mirror_prob: 0.5, noise_prob: 0.3)
      embedding = Nx.take(dataset.embedded_frames, indices) |> Nx.take(..., axis: 1, indices: variant_idx)
  """
  @spec precompute_augmented_frame_embeddings(t(), keyword()) :: t()
  def precompute_augmented_frame_embeddings(dataset, opts \\ []) do
    alias ExPhil.Training.Augmentation

    num_noisy_variants = Keyword.get(opts, :num_noisy_variants, 2)
    noise_scale = Keyword.get(opts, :noise_scale, 0.01)
    show_progress = Keyword.get(opts, :show_progress, true)
    batch_size = Keyword.get(opts, :batch_size, 500)
    gc_every = Keyword.get(opts, :gc_every, 50)

    # Total variants: 1 (original) + 1 (mirrored) + num_noisy_variants
    num_variants = 2 + num_noisy_variants
    total_frames = dataset.size

    if show_progress do
      IO.puts(:stderr, "  Pre-computing augmented embeddings for #{total_frames} frames...")
      IO.puts(:stderr, "    Variants: original + mirrored + #{num_noisy_variants} noisy = #{num_variants} total")
      estimated_mb = Float.round(total_frames * num_variants * 300 * 4 / 1_000_000_000, 2)
      IO.puts(:stderr, "    Estimated size: ~#{estimated_mb} GB")
    end

    embed_config = dataset.embed_config

    # Process frames in batches, generating all variants for each frame
    batch_tensors =
      dataset.frames
      |> Enum.chunk_every(batch_size)
      |> Enum.with_index()
      |> Enum.map(fn {chunk, chunk_idx} ->
        if show_progress do
          processed = min((chunk_idx + 1) * batch_size, total_frames)
          pct = round(processed / total_frames * 100)
          IO.write(:stderr, "\r  Augmented embedding: #{pct}% (#{processed}/#{total_frames})    ")
        end

        # Periodic garbage collection
        if gc_every > 0 and rem(chunk_idx, gc_every) == 0 and chunk_idx > 0 do
          :erlang.garbage_collect()
        end

        # Generate all variants for this chunk
        # For each frame, create: [original, mirrored, noisy1, noisy2, ...]
        variants_per_frame =
          Enum.map(chunk, fn frame ->
            original = frame
            mirrored = Augmentation.mirror(frame)

            # Generate noisy variants with deterministic seeds based on frame index
            # This ensures reproducibility across runs
            noisy_variants =
              for noise_idx <- 0..(num_noisy_variants - 1) do
                # Use frame hash + noise_idx as seed for reproducibility
                frame_hash = :erlang.phash2(frame.game_state)
                seed = frame_hash + noise_idx * 12345
                :rand.seed(:exsss, {seed, seed + 1, seed + 2})

                Augmentation.add_noise(frame, scale: noise_scale)
              end

            [original, mirrored | noisy_variants]
          end)

        # Flatten to process all variants at once: chunk_size * num_variants frames
        all_frames = List.flatten(variants_per_frame)
        game_states = Enum.map(all_frames, & &1.game_state)

        # Extract name_ids (same for all variants of a frame)
        base_name_ids = Enum.map(chunk, fn f -> f[:name_id] || 0 end)
        name_ids = Enum.flat_map(base_name_ids, fn id -> List.duplicate(id, num_variants) end)

        # Embed all variants in one batch
        all_embeddings =
          Embeddings.Game.embed_states_fast(game_states, 1,
            config: embed_config,
            name_id: name_ids
          )
          |> Nx.backend_copy(Nx.BinaryBackend)

        # Reshape from {chunk_size * num_variants, embed_size} to {chunk_size, num_variants, embed_size}
        embed_size = Nx.axis_size(all_embeddings, 1)
        chunk_size = length(chunk)

        all_embeddings
        |> Nx.reshape({chunk_size, num_variants, embed_size})
      end)

    if show_progress do
      IO.puts(:stderr, "\r  Augmented embedding: 100% (#{total_frames}/#{total_frames}) - done!    ")
    end

    # Concatenate all batches: {total_frames, num_variants, embed_size}
    stacked_embeddings = Nx.concatenate(batch_tensors, axis: 0)

    %{dataset | embedded_frames: stacked_embeddings}
  end

  @doc """
  Pre-compute augmented frame embeddings with optional disk caching.

  ## Options
    - `:cache` - Enable caching (default: false)
    - `:cache_dir` - Cache directory (default: "cache/embeddings")
    - `:force_recompute` - Ignore cache and recompute (default: false)
    - `:replay_files` - List of replay file paths (required for cache key)
    - `:num_noisy_variants` - Number of noisy variants (default: 2)
    - `:noise_scale` - Noise scale (default: 0.01)
    - All options from `precompute_augmented_frame_embeddings/2`

  ## Example

      dataset = Data.precompute_augmented_frame_embeddings_cached(dataset,
        cache: true,
        replay_files: replay_files,
        num_noisy_variants: 2
      )
  """
  @spec precompute_augmented_frame_embeddings_cached(t(), keyword()) :: t()
  def precompute_augmented_frame_embeddings_cached(dataset, opts \\ []) do
    cache_enabled = Keyword.get(opts, :cache, false)
    force_recompute = Keyword.get(opts, :force_recompute, false)
    replay_files = Keyword.get(opts, :replay_files, [])
    cache_dir = Keyword.get(opts, :cache_dir, "cache/embeddings")
    num_noisy_variants = Keyword.get(opts, :num_noisy_variants, 2)
    noise_scale = Keyword.get(opts, :noise_scale, 0.01)

    alias ExPhil.Training.EmbeddingCache

    if cache_enabled and not force_recompute and length(replay_files) > 0 do
      cache_key =
        EmbeddingCache.cache_key(dataset.embed_config, replay_files,
          temporal: false,
          augmented: true,
          num_noisy_variants: num_noisy_variants,
          noise_scale: noise_scale
        )

      case EmbeddingCache.load(cache_key, cache_dir: cache_dir) do
        {:ok, embedded_array} ->
          Logger.info("[EmbeddingCache] Using cached augmented frame embeddings")
          %{dataset | embedded_frames: embedded_array}

        {:error, :not_found} ->
          Logger.info("[EmbeddingCache] Cache miss, computing augmented embeddings...")
          result = precompute_augmented_frame_embeddings(dataset, opts)

          if result.embedded_frames do
            EmbeddingCache.save(cache_key, result.embedded_frames, cache_dir: cache_dir)
          end

          result

        {:error, reason} ->
          Logger.warning(
            "[EmbeddingCache] Failed to load cache: #{inspect(reason)}, recomputing..."
          )

          precompute_augmented_frame_embeddings(dataset, opts)
      end
    else
      precompute_augmented_frame_embeddings(dataset, opts)
    end
  end

  @doc """
  Select variant index based on augmentation probabilities.

  Used when batching from augmented embeddings to randomly select
  which variant (original, mirrored, or noisy) to use for each sample.

  ## Options
    - `:mirror_prob` - Probability of selecting mirrored variant (default: 0.5)
    - `:noise_prob` - Probability of selecting a noisy variant (default: 0.3)
    - `:num_noisy_variants` - Number of noisy variants available (default: 2)

  ## Returns

  Integer index into the variant dimension:
    - 0: Original
    - 1: Mirrored
    - 2+: Noisy variant

  ## Selection Logic

  1. First, decide augmentation type: original (1-mirror-noise), mirror, or noise
  2. If noise selected, randomly pick which noisy variant
  """
  @spec select_variant_index(keyword()) :: non_neg_integer()
  def select_variant_index(opts \\ []) do
    mirror_prob = Keyword.get(opts, :mirror_prob, 0.5)
    noise_prob = Keyword.get(opts, :noise_prob, 0.3)
    num_noisy_variants = Keyword.get(opts, :num_noisy_variants, 2)

    # Roll for augmentation type
    roll = :rand.uniform()

    cond do
      # Noise takes priority (applied after mirror decision in original code)
      roll < noise_prob ->
        # Randomly select which noisy variant (indices 2, 3, ...)
        2 + :rand.uniform(num_noisy_variants) - 1

      roll < noise_prob + mirror_prob * (1 - noise_prob) ->
        # Mirrored (index 1)
        1

      true ->
        # Original (index 0)
        0
    end
  end

  @doc """
  Check if dataset has augmented (multi-variant) frame embeddings.
  """
  @spec has_augmented_embeddings?(t()) :: boolean()
  def has_augmented_embeddings?(%__MODULE__{embedded_frames: nil}), do: false

  def has_augmented_embeddings?(%__MODULE__{embedded_frames: embeddings}) do
    # Augmented embeddings have 3 dimensions: {frames, variants, embed_size}
    # Regular embeddings have 2 dimensions: {frames, embed_size}
    tuple_size(Nx.shape(embeddings)) == 3
  end

  @doc """
  Pre-compute sequence embeddings with optional disk caching.

  If caching is enabled and a cached version exists, loads from disk.
  Otherwise computes embeddings and optionally saves to cache.

  ## Options
    - `:cache` - Enable caching (default: false)
    - `:cache_dir` - Cache directory (default: "cache/embeddings")
    - `:force_recompute` - Ignore cache and recompute (default: false)
    - `:replay_files` - List of replay file paths (required for cache key)
    - `:window_size` - Window size (used in cache key)
    - `:stride` - Stride (used in cache key)
    - All options from `precompute_embeddings/2`
  """
  @spec precompute_embeddings_cached(t(), keyword()) :: t()
  def precompute_embeddings_cached(dataset, opts \\ []) do
    cache_enabled = Keyword.get(opts, :cache, false)
    force_recompute = Keyword.get(opts, :force_recompute, false)
    replay_files = Keyword.get(opts, :replay_files, [])
    cache_dir = Keyword.get(opts, :cache_dir, "cache/embeddings")
    window_size = Keyword.get(opts, :window_size, 30)
    stride = Keyword.get(opts, :stride, 1)

    alias ExPhil.Training.EmbeddingCache

    if cache_enabled and not force_recompute and length(replay_files) > 0 do
      cache_key =
        EmbeddingCache.cache_key(dataset.embed_config, replay_files,
          temporal: true,
          window_size: window_size,
          stride: stride
        )

      Logger.info(
        "[EmbeddingCache] Looking for cache key: #{cache_key} (temporal, window=#{window_size}, stride=#{stride})"
      )

      case EmbeddingCache.load(cache_key, cache_dir: cache_dir) do
        {:ok, cached_dataset} when is_struct(cached_dataset, __MODULE__) ->
          # Full dataset was cached
          Logger.info("[EmbeddingCache] Using cached sequence embeddings")
          cached_dataset

        {:ok, embedded_seqs} ->
          # Just the embeddings were cached
          Logger.info("[EmbeddingCache] Using cached sequence embeddings")
          %{dataset | embedded_sequences: embedded_seqs}

        {:error, :not_found} ->
          # Show available caches for debugging
          available =
            EmbeddingCache.list(cache_dir: cache_dir)
            |> Enum.map(& &1.key)
            |> Enum.join(", ")

          Logger.info("[EmbeddingCache] Cache miss for #{cache_key}")
          Logger.info("[EmbeddingCache] Available caches: #{available}")
          result = precompute_embeddings(dataset, opts)

          # Save to cache
          if result.embedded_sequences do
            Logger.info("[EmbeddingCache] Saving sequence embeddings to #{cache_key}...")
            case EmbeddingCache.save(cache_key, result, cache_dir: cache_dir) do
              :ok ->
                Logger.info("[EmbeddingCache] Successfully saved sequence cache")
              {:error, reason} ->
                Logger.error("[EmbeddingCache] Failed to save sequence cache: #{inspect(reason)}")
            end
          else
            Logger.warning("[EmbeddingCache] No embedded_sequences to save (this is a bug)")
          end

          result

        {:error, reason} ->
          Logger.warning(
            "[EmbeddingCache] Failed to load cache: #{inspect(reason)}, recomputing..."
          )

          precompute_embeddings(dataset, opts)
      end
    else
      precompute_embeddings(dataset, opts)
    end
  end

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

    embeddings_array =
      cond do
        is_nil(dataset.embedded_sequences) ->
          nil

        # Already an array (from precompute_embeddings or cache load)
        is_tuple(dataset.embedded_sequences) and elem(dataset.embedded_sequences, 0) == :array ->
          dataset.embedded_sequences

        # List that needs conversion
        is_list(dataset.embedded_sequences) ->
          :array.from_list(dataset.embedded_sequences)

        true ->
          nil
      end

    # Prepare indices
    valid_indices = 0..(dataset.size - 1) |> Enum.to_list()

    # Seed random number generator
    :rand.seed(:exsss, {seed, seed, seed})

    indices =
      cond do
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
  defp create_sequence_batch_fast(frames_array, embeddings_array, indices)
       when embeddings_array != nil do
    # Fast path: use pre-computed embeddings with O(1) array access
    batch_data =
      Enum.map(indices, fn idx ->
        frame = :array.get(idx, frames_array)
        embedding = :array.get(idx, embeddings_array)
        {embedding, frame.action}
      end)

    {embeddings, actions} = Enum.unzip(batch_data)

    # Stack embeddings and transfer to GPU: [batch, seq_len, embed_size]
    # Embeddings are stored on CPU to avoid GPU OOM, transfer here for training
    states =
      embeddings
      |> Nx.stack()
      |> Nx.backend_transfer(EXLA.Backend)

    # Convert actions to tensors and transfer to GPU
    action_tensors =
      actions_to_tensors(actions)
      |> transfer_actions_to_gpu()

    %{
      states: states,
      actions: action_tensors
    }
  end

  defp create_sequence_batch_fast(frames_array, nil, indices) do
    # Slow path: no pre-computed embeddings, embed on-the-fly
    sequence_data =
      Enum.map(indices, fn idx ->
        frame = :array.get(idx, frames_array)
        # Extract name_id from first frame of sequence for style-conditional training
        name_id = get_in(frame, [:sequence, Access.at(0), :name_id]) || 0
        {frame.sequence, frame.action, name_id}
      end)

    {sequences, actions, name_ids} = unzip3(sequence_data)

    # Embed sequences with name_ids
    embedded =
      Enum.zip(sequences, name_ids)
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
    button_counts =
      Enum.reduce(actions, %{}, fn action, acc ->
        Enum.reduce(action.buttons, acc, fn {button, pressed}, inner_acc ->
          if pressed do
            Map.update(inner_acc, button, 1, &(&1 + 1))
          else
            inner_acc
          end
        end)
      end)

    button_rates =
      Map.new(button_counts, fn {button, count} ->
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
