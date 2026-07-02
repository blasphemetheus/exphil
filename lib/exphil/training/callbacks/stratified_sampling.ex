defmodule ExPhil.Training.Callbacks.StratifiedSampling do
  @moduledoc """
  Ensure each batch contains a minimum percentage of action-rich frames.

  Without stratification, batches can be 95%+ neutral frames (no buttons pressed,
  sticks centered). This gives uniform "predict neutral" gradient signal.

  Stratified sampling guarantees each batch has at least `min_action_pct`% of
  frames with button presses or non-center sticks, by oversampling from an
  action-rich index set.

  ## Usage

      # In Pipeline or Data module:
      Data.batched_sequences(dataset,
        stratified: true,
        min_action_pct: 0.3  # 30% of each batch has actions
      )

  ## Note

  This is a data-level solution complementary to entropy regularization (loss-level).
  Both address mode collapse from different angles.
  """

  # This module provides the stratified index generation logic.
  # It's used by Data.batched_sequences when stratified: true is passed.

  @doc """
  Generate stratified indices for batching.

  Given a list of frames, returns indices that ensure each batch of `batch_size`
  has at least `min_action_pct` proportion of action-containing frames.

  ## Parameters
  - `frames` — list of frame maps with controller/action data
  - `num_indices` — total number of indices to generate
  - `batch_size` — size of each batch
  - `opts` — options:
    - `:min_action_pct` — minimum fraction of action frames per batch (default: 0.3)
    - `:shuffle` — shuffle within strata (default: true)
  """
  def generate_indices(frames, num_indices, batch_size, opts \\ []) do
    min_action_pct = Keyword.get(opts, :min_action_pct, 0.3)
    shuffle = Keyword.get(opts, :shuffle, true)

    # Classify frames as "action" or "neutral"
    {action_indices, neutral_indices} =
      frames
      |> Enum.with_index()
      |> Enum.split_with(fn {frame, _idx} -> has_action?(frame) end)

    action_indices = Enum.map(action_indices, &elem(&1, 1))
    neutral_indices = Enum.map(neutral_indices, &elem(&1, 1))

    action_indices = if shuffle, do: Enum.shuffle(action_indices), else: action_indices
    neutral_indices = if shuffle, do: Enum.shuffle(neutral_indices), else: neutral_indices

    # For each batch, pick min_action_pct from action set, rest from neutral
    action_per_batch = max(round(batch_size * min_action_pct), 1)
    neutral_per_batch = batch_size - action_per_batch

    num_batches = div(num_indices, batch_size)

    # Build interleaved indices: cycle through action and neutral pools
    action_cycle = if action_indices == [], do: neutral_indices, else: action_indices
    neutral_cycle = if neutral_indices == [], do: action_indices, else: neutral_indices

    indices =
      for batch_num <- 0..(num_batches - 1) do
        action_start = rem(batch_num * action_per_batch, length(action_cycle))
        neutral_start = rem(batch_num * neutral_per_batch, length(neutral_cycle))

        action_batch = Enum.slice(Stream.cycle(action_cycle) |> Enum.take(length(action_cycle) + action_per_batch), action_start, action_per_batch)
        neutral_batch = Enum.slice(Stream.cycle(neutral_cycle) |> Enum.take(length(neutral_cycle) + neutral_per_batch), neutral_start, neutral_per_batch)

        batch = action_batch ++ neutral_batch
        if shuffle, do: Enum.shuffle(batch), else: batch
      end
      |> List.flatten()

    indices
  end

  defp has_action?(frame) do
    cond do
      Map.has_key?(frame, :action) ->
        action = frame.action
        buttons = action[:buttons] || %{}
        any_button = Enum.any?(buttons, fn {_k, v} -> v == true end)
        stick_moved = (action[:main_x] || 8) != 8 or (action[:main_y] || 8) != 8
        any_button or stick_moved

      Map.has_key?(frame, :controller) ->
        c = frame.controller
        any_button = c.button_a or c.button_b or c.button_x or c.button_y or
                     c.button_z or c.button_l or c.button_r
        stick_moved = abs(c.main_stick.x - 0.5) > 0.2 or abs(c.main_stick.y - 0.5) > 0.2
        any_button or stick_moved

      true -> false
    end
  end

end
