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

  require Logger

  defstruct [
    :frames,              # List of frame data
    :metadata,            # Dataset metadata
    :embed_config,        # Embedding configuration
    :size,                # Number of frames
    :embedded_sequences   # Pre-computed embeddings (optional, for temporal training)
  ]

  @type frame :: %{
    game_state: GameState.t(),
    controller: ControllerState.t() | nil,
    action: map() | nil
  }

  @type t :: %__MODULE__{
    frames: [frame()],
    metadata: map(),
    embed_config: map(),
    size: non_neg_integer()
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
  """
  @spec from_frames([frame()], keyword()) :: t()
  def from_frames(frames, opts \\ []) do
    embed_config = Keyword.get_lazy(opts, :embed_config, fn ->
      Embeddings.config()
    end)

    %__MODULE__{
      frames: frames,
      metadata: Keyword.get(opts, :metadata, %{}),
      embed_config: embed_config,
      size: length(frames)
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
    - `:frame_delay` - Frames between state and action (default: 0)
    - `:seed` - Random seed for shuffling

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
    seed = Keyword.get(opts, :seed, System.system_time())

    # Prepare indices
    indices = 0..(dataset.size - 1 - frame_delay) |> Enum.to_list()

    indices = if shuffle do
      :rand.seed(:exsss, {seed, seed, seed})
      Enum.shuffle(indices)
    else
      indices
    end

    # Create batch stream
    indices
    |> Enum.chunk_every(batch_size)
    |> maybe_drop_last(drop_last, batch_size)
    |> Stream.map(fn batch_indices ->
      create_batch(dataset, batch_indices, frame_delay)
    end)
  end

  defp maybe_drop_last(chunks, false, _batch_size), do: chunks
  defp maybe_drop_last(chunks, true, batch_size) do
    Enum.filter(chunks, &(length(&1) == batch_size))
  end

  defp create_batch(dataset, indices, frame_delay) do
    # Collect frames
    frame_data = Enum.map(indices, fn idx ->
      state_frame = Enum.at(dataset.frames, idx)
      action_frame = Enum.at(dataset.frames, idx + frame_delay)

      {state_frame.game_state, get_action(action_frame)}
    end)

    # Embed states
    {game_states, actions} = Enum.unzip(frame_data)
    states = embed_states(game_states, dataset.embed_config)

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

  defp embed_states(game_states, embed_config) do
    embeddings = Enum.map(game_states, fn gs ->
      # Embed using the standard interface
      # Passes embed_config as opts
      Embeddings.embed(gs, nil, embed_config: embed_config)
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

    if show_progress do
      IO.puts("  Pre-computing embeddings for #{dataset.size} sequences...")
    end

    embed_config = dataset.embed_config
    total = dataset.size

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
        IO.puts("  Embedding: #{pct}% (#{processed}/#{total})")
      end

      # Batch embed each sequence in this chunk
      # For each sequence, we batch all frames together
      Enum.map(chunk, fn frame ->
        game_states = Enum.map(frame.sequence, & &1.game_state)
        # Use batch embedding: all frames at once
        Embeddings.Game.embed_states_fast(game_states, 1, config: embed_config)
      end)
    end)

    if show_progress do
      IO.puts("  Embedding: 100% (#{total}/#{total}) - done!")
    end

    %{dataset | embedded_sequences: embedded}
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

    # Convert lists to arrays for O(1) index access (vs O(n) for lists)
    frames_array = :array.from_list(dataset.frames)
    embeddings_array = if dataset.embedded_sequences do
      :array.from_list(dataset.embedded_sequences)
    else
      nil
    end

    # Prepare indices
    indices = 0..(dataset.size - 1) |> Enum.to_list()

    indices = if shuffle do
      :rand.seed(:exsss, {seed, seed, seed})
      Enum.shuffle(indices)
    else
      indices
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
      {frame.sequence, frame.action}
    end)

    {sequences, actions} = Enum.unzip(sequence_data)

    # Embed sequences
    embedded = Enum.map(sequences, fn seq ->
      game_states = Enum.map(seq, & &1.game_state)
      Embeddings.Game.embed_states_fast(game_states, 1)
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
