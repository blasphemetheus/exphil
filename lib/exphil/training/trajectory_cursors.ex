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

  A segment = one contiguous replay (game). Boundaries are recovered the
  same way `AdvantageWeighting.split_by_replay/1` does: the Slippi frame
  counter going backward (`cur <= prev`) marks a new game. Hidden state
  must reset ONLY at these boundaries — never at chunk edges.

  ## v0 limitations (deliberate)

  - Segments shorter than `unroll` are skipped.
  - When the segment queue is exhausted, the stream halts; rows still
    mid-segment lose their tail (< unroll frames each). With ~100-file
    chunks this loses < batch_size * unroll frames per chunk.
  - `batch_size` rows want >= batch_size segments per chunk; with fewer,
    rows beyond the segment count are never filled. The stream raises
    if fewer than `batch_size` usable segments exist (use bigger
    stream chunks or a smaller batch — see design doc "open decisions").
  """

  alias ExPhil.Training.Data

  @doc """
  Recover segment boundaries from a flat frame list.

  Returns `[{start_index, length}]` in corpus order. Boundary rule:
  Slippi frame counter decreases or repeats (`cur <= prev`).
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

        cur <= prev ->
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
      segment (default 1; use frame_delay + 1)
    - `:seed` - segment shuffle seed (default 42)
    - `:neutral_weight` / `:transition_weight` - per-frame weighting,
      as in the windowed batcher
    - `:gpu` - transfer states to EXLA.Backend (default true)
  """
  @spec batch_stream(Data.t(), keyword()) :: Enumerable.t()
  def batch_stream(dataset, opts) do
    batch_size = Keyword.fetch!(opts, :batch_size)
    unroll = Keyword.get(opts, :unroll, 80)
    overlap = Keyword.get(opts, :overlap, 1)
    seed = Keyword.get(opts, :seed, 42)
    neutral_weight = Keyword.get(opts, :neutral_weight, 0.25)
    transition_weight = Keyword.get(opts, :transition_weight)
    gpu = Keyword.get(opts, :gpu, true)

    if dataset.embedded_frames == nil do
      raise ArgumentError,
            "TrajectoryCursors needs dataset.embedded_frames (precomputed lazy layout)"
    end

    if overlap >= unroll do
      raise ArgumentError, "overlap (#{overlap}) must be < unroll (#{unroll})"
    end

    frames_array = :array.from_list(dataset.frames)

    segs =
      dataset.frames
      |> segments()
      |> Enum.filter(fn {_start, len} -> len >= unroll end)

    if length(segs) < batch_size do
      raise ArgumentError,
            "only #{length(segs)} segments >= #{unroll} frames for batch_size " <>
              "#{batch_size} — use larger stream chunks or a smaller batch " <>
              "(see BPTT_LOADER_DESIGN.md)"
    end

    queue = seeded_shuffle(segs, seed)

    # Cursor: %{start: seg_start, len: seg_len, off: offset_into_segment}
    # nil = needs a segment.
    init = %{cursors: List.duplicate(nil, batch_size), queue: queue}

    Stream.resource(
      fn -> init end,
      fn state -> next_batch(state, frames_array, dataset.embedded_frames, batch_size, unroll, overlap, neutral_weight, transition_weight, gpu) end,
      fn _ -> :ok end
    )
  end

  # -- internals -------------------------------------------------------------

  defp next_batch(state, frames_array, embedded, batch_size, unroll, overlap, neutral_w, transition_w, gpu) do
    case assign_cursors(state.cursors, state.queue, unroll) do
      :exhausted ->
        {:halt, state}

      {cursors, queue, resets} ->
        starts = Enum.map(cursors, fn c -> c.start + c.off end)

        # One gather instead of batch_size slices + stack: eager EXLA
        # bakes slice START offsets into the compiled executable, so
        # per-row slicing recompiled XLA programs every batch (measured
        # 1.9s/batch = 98.5% of step time, 2026-09-04 bptt-prof).
        # Gather indices are runtime DATA — one cached executable.
        states =
          prof(:states_gather, fn ->
            embed_dim = Nx.axis_size(embedded, 1)

            idx =
              starts
              |> Nx.tensor(type: :s32)
              |> Nx.reshape({batch_size, 1})
              |> Nx.add(Nx.iota({1, unroll}, type: :s32))
              |> Nx.reshape({batch_size * unroll})

            embedded
            |> Nx.take(idx)
            |> Nx.reshape({batch_size, unroll, embed_dim})
          end)

        states =
          prof(:states_transfer, fn ->
            if gpu, do: Nx.backend_transfer(states, EXLA.Backend), else: states
          end)

        # Per-timestep actions, row-major [b0t0, b0t1, ..., b1t0, ...]
        flat_actions =
          prof(:actions_extract, fn ->
            Enum.flat_map(starts, fn s ->
              Enum.map(s..(s + unroll - 1), fn idx ->
                Data.frame_action(:array.get(idx, frames_array))
              end)
            end)
          end)

        # Previous action for each position: within a row, index - 1
        # (for t=0 of a chunk mid-segment this reaches the true previous
        # frame; at a segment start there is none -> nil).
        flat_prev =
          prof(:prev_extract, fn ->
            Enum.flat_map(Enum.zip(starts, cursors), fn {s, c} ->
              Enum.map(s..(s + unroll - 1), fn idx ->
                if idx > c.start do
                  Data.frame_action(:array.get(idx - 1, frames_array))
                else
                  nil
                end
              end)
            end)
          end)

        weights =
          prof(:frame_weights, fn ->
            flat_actions
            |> Data.compute_frame_weights(
              neutral_weight: neutral_w,
              transition_weight: transition_w,
              prev_actions: if(transition_w, do: flat_prev)
            )
            |> Nx.reshape({batch_size, unroll})
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
          is_resetting: Nx.tensor(resets, type: :u8)
        }

        new_cursors = Enum.map(cursors, &%{&1 | off: &1.off + unroll - overlap})
        {[batch], %{state | cursors: new_cursors, queue: queue}}
    end
  end

  # Give every row a valid cursor (drawing from the queue where needed).
  # Returns {cursors, queue, resets} or :exhausted when any row needs a
  # segment and none remain.
  defp assign_cursors(cursors, queue, unroll) do
    {rev_cursors, queue, rev_resets, exhausted?} =
      Enum.reduce(cursors, {[], queue, [], false}, fn c, {acc, q, resets, ex} ->
        cond do
          ex ->
            {acc, q, resets, ex}

          c != nil and c.off + unroll <= c.len ->
            {[c | acc], q, [0 | resets], ex}

          q == [] ->
            {acc, q, resets, true}

          true ->
            [{start, len} | rest] = q
            {[%{start: start, len: len, off: 0} | acc], rest, [1 | resets], ex}
        end
      end)

    if exhausted? do
      :exhausted
    else
      {Enum.reverse(rev_cursors), queue, Enum.reverse(rev_resets)}
    end
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
