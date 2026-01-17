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
    :frames,        # List of frame data
    :metadata,      # Dataset metadata
    :embed_config,  # Embedding configuration
    :size           # Number of frames
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
