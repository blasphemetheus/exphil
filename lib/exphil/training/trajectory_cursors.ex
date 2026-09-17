defmodule ExPhil.Training.TrajectoryCursors do
  @moduledoc """
  Contiguous-BPTT batch loader (slippi-ai TrajectoryManager design,
  see docs/planning/BPTT_LOADER_DESIGN.md).

  Instead of independent random windows, each of the `batch_size` rows
  is a CURSOR walking one replay segment in order. Consecutive batches
  from this stream are temporally consecutive for every row, so the
  trainer can carry the GRU hidden state from batch k to batch k+1
  (resetting only the rows whose `is_resetting` flag is set — rows that
  just started a new segment).

  ## Batch shape (per element of the stream)

      %{
        states:       {batch, unroll, embed_dim}  (f32, EXLA)
        actions:      %{buttons: {batch, unroll, 8}, main_x: {batch, unroll}, ...}
        frame_weights:{batch, unroll} f32 (neutral/transition weighting,
                       same rules as the windowed batcher)
        is_resetting: {batch} u8 — 1 where this row starts a new segment
      }

  Supervision is PER-TIMESTEP (every frame in the unroll), unlike the
  windowed batcher's last-frame-only labels.

  ## Segment law

  A segment is a contiguous portion of a replay. A frame-counter reset,
  repeat, or forward gap starts a new segment (`cur != prev + 1`). Hidden
  state resets at these boundaries, never at ordinary chunk edges.

  ## Coverage

  Every frame is emitted once. Short segments and final partial rows are padded
  by repeating their last frame with zero loss weight. Inactive rows are also
  masked and reset. A row never crosses a segment boundary within an unroll;
  padded carry is discarded before the next segment starts.
  """

  alias ExPhil.Training.Data

  @doc """
  Recover segment boundaries from a flat frame list.

  Returns `[{start_index, length}]` in corpus order. Boundary rule:
  Slippi frame counter is not the immediate successor (`cur != prev + 1`).
  """
  @spec segments([map()]) :: [{non_neg_integer(), pos_integer()}]
  def segments(frames) do
    frames
    |> Enum.with_index()
    |> Enum.reduce({[], nil, nil}, fn {frame, idx}, {done, seg_start, prev} ->
      cur = frame.game_state.frame

      cond do
        seg_start == nil ->
          {done, idx, cur}

        cur != prev + 1 ->
          {[{seg_start, idx - seg_start} | done], idx, cur}

        true ->
          {done, seg_start, cur}
      end
    end)
    |> then(fn
      {done, nil, _} -> done
      {done, seg_start, _} -> [{seg_start, length(frames) - seg_start} | done]
    end)
    |> Enum.reverse()
  end

  @doc """
  Build the contiguous batch stream over a prepared dataset
  (`dataset.frames` + `dataset.embedded_frames` — the lazy/streaming
  layout, same precondition as `Data.batched_sequences(lazy: true)`).

  ## Options
    - `:batch_size` (required)
    - `:unroll` - timesteps per chunk (default 80)
    - `:overlap` - frames shared between consecutive chunks of the same
      segment (must be 0: carry already includes the entire previous chunk)
    - `:seed` - segment shuffle seed (default 42)
    - `:neutral_weight` / `:transition_weight` - per-frame weighting,
      as in the windowed batcher
    - `:gpu` - transfer states to EXLA.Backend (default true)
  """
  @spec batch_stream(Data.t(), keyword()) :: Enumerable.t()
  def batch_stream(%Data{frames: []}, _opts), do: []

  def batch_stream(dataset, opts) do
    batch_size = Keyword.fetch!(opts, :batch_size)
    unroll = Keyword.get(opts, :unroll, 80)
    overlap = Keyword.get(opts, :overlap, 0)
    seed = Keyword.get(opts, :seed, 42)
    neutral_weight = Keyword.get(opts, :neutral_weight, 0.25)
    transition_weight = Keyword.get(opts, :transition_weight)
    offstage_weight = Keyword.get(opts, :offstage_weight)
    gpu = Keyword.get(opts, :gpu, true)

    if dataset.embedded_frames == nil do
      raise ArgumentError,
            "TrajectoryCursors needs dataset.embedded_frames (precomputed lazy layout)"
    end

    unless overlap == 0,
      do:
        raise(
          ArgumentError,
          "BPTT overlap must be 0; carried state already consumed the previous unroll"
        )

    unless is_integer(batch_size) and batch_size > 0 and is_integer(unroll) and unroll > 0,
      do: raise(ArgumentError, "batch_size and unroll must be positive integers")

    frames_array = :array.from_list(dataset.frames)
    queue = seeded_shuffle(segments(dataset.frames), seed)

    # Cursor: %{start: seg_start, len: seg_len, off: offset_into_segment}
    # nil = needs a segment.
    init = %{cursors: List.duplicate(nil, batch_size), queue: queue}

    Stream.resource(
      fn -> init end,
      fn state ->
        next_batch(
          state,
          frames_array,
          dataset.embedded_frames,
          batch_size,
          unroll,
          overlap,
          {neutral_weight, transition_weight, offstage_weight},
          gpu
        )
      end,
      fn _ -> :ok end
    )
  end

  # -- internals -------------------------------------------------------------

  defp next_batch(
         state,
         frames_array,
         embedded,
         batch_size,
         unroll,
         _overlap,
         {neutral_w, transition_w, offstage_w},
         gpu
       ) do
    case assign_cursors(state.cursors, state.queue, unroll) do
      :exhausted ->
        {:halt, state}

      {cursors, queue, resets} ->
        rows =
          Enum.map(cursors, fn
            nil ->
              {List.duplicate(0, unroll), List.duplicate(0.0, unroll)}

            c ->
              indices = for t <- 0..(unroll - 1), do: c.start + min(c.off + t, c.len - 1)
              mask = for t <- 0..(unroll - 1), do: if(c.off + t < c.len, do: 1.0, else: 0.0)
              {indices, mask}
          end)

        indices = Enum.flat_map(rows, &elem(&1, 0))

        valid_mask =
          Enum.flat_map(rows, &elem(&1, 1)) |> Nx.tensor() |> Nx.reshape({batch_size, unroll})

        # One gather instead of batch_size slices + stack: eager EXLA
        # bakes slice START offsets into the compiled executable, so
        # per-row slicing recompiled XLA programs every batch (measured
        # 1.9s/batch = 98.5% of step time, 2026-09-04 bptt-prof).
        # Gather indices are runtime DATA — one cached executable.
        states =
          prof(:states_gather, fn ->
            embed_dim = Nx.axis_size(embedded, 1)

            idx = Nx.tensor(indices, type: :s32)

            embedded
            |> Nx.take(idx)
            |> Nx.reshape({batch_size, unroll, embed_dim})
          end)

        states =
          prof(:states_transfer, fn ->
            if gpu, do: Nx.backend_transfer(states, EXLA.Backend), else: states
          end)

        flat_actions = Enum.map(indices, &Data.frame_action(:array.get(&1, frames_array)))

        flat_prev =
          Enum.zip(rows, cursors)
          |> Enum.flat_map(fn {{row_indices, _mask}, c} ->
            Enum.map(row_indices, fn idx ->
              if c != nil and idx > c.start,
                do: Data.frame_action(:array.get(idx - 1, frames_array)),
                else: nil
            end)
          end)

        flat_offstage =
          if offstage_w,
            do: Enum.map(indices, &Data.frame_offstage?(:array.get(&1, frames_array)))

        weights =
          prof(:frame_weights, fn ->
            flat_actions
            |> Data.compute_frame_weights(
              neutral_weight: neutral_w,
              transition_weight: transition_w,
              prev_actions: if(transition_w, do: flat_prev),
              offstage_weight: offstage_w,
              offstage: flat_offstage
            )
            |> Nx.reshape({batch_size, unroll})
            |> Nx.multiply(valid_mask)
          end)

        targets =
          prof(:targets_tensorize, fn ->
            flat_actions
            |> Data.actions_to_tensors()
            |> Map.new(fn {head, t} ->
              new_shape =
                case Nx.shape(t) do
                  {_n} -> {batch_size, unroll}
                  {_n, k} -> {batch_size, unroll, k}
                end

              {head, Nx.reshape(t, new_shape)}
            end)
          end)

        batch = %{
          states: states,
          actions: targets,
          frame_weights: weights,
          valid_mask: valid_mask,
          is_resetting: Nx.tensor(resets, type: :u8)
        }

        new_cursors =
          Enum.map(cursors, fn
            nil -> nil
            c -> %{c | off: c.off + unroll}
          end)

        {[batch], %{state | cursors: new_cursors, queue: queue}}
    end
  end

  # Refill completed rows; stop only when ALL rows and the queue are exhausted.
  defp assign_cursors(cursors, queue, _unroll) do
    {rows, {queue, resets}} =
      Enum.map_reduce(cursors, {queue, []}, fn c, {q, resets} ->
        cond do
          c != nil and c.off < c.len ->
            {c, {q, [0 | resets]}}

          q == [] ->
            {nil, {q, [1 | resets]}}

          true ->
            [{start, len} | rest] = q
            {%{start: start, len: len, off: 0}, {rest, [1 | resets]}}
        end
      end)

    if Enum.all?(rows, &is_nil/1),
      do: :exhausted,
      else: {rows, queue, Enum.reverse(resets)}
  end

  # -- lightweight stage profiling (EXPHIL_BPTT_PROFILE=1) -------------------
  # Accumulates per-stage wall time in the consumer's process dictionary
  # (Stream.resource runs in the consuming process, same as the train
  # loop, so ProfReport.report/1 called there sees these totals).

  defp prof(key, fun) do
    if ExPhil.Training.BpttProf.enabled?() do
      ExPhil.Training.BpttProf.time(key, fun)
    else
      fun.()
    end
  end

  defp seeded_shuffle(list, seed) do
    :rand.seed(:exsss, {seed, seed * 7919, seed + 104_729})
    Enum.shuffle(list)
  end
end
