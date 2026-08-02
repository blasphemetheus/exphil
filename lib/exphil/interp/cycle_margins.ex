defmodule ExPhil.Interp.CycleMargins do
  @moduledoc """
  Cycle-margin probing at expert button EDGES — the library form of
  `scripts/probe_cycle_margins.exs`, extracted so training loops can
  measure margins per epoch on the teacher fixture (INIT_FORENSICS
  option 6 / early-reject; task #2 of the 2026-08-02 backlog).

  Events (single-frame edges, not holds — v1's phase labeling produced
  flip≈1.0 for the champion):

    * `:jc_event` — X edge while in grounded reflector (the jump-cancel)
    * `:aerial_shine_event` — B edge in jumpsquat/aerial jump (THE input)
    * `:ground_shine_event` — B edge grounded, non-reflector (cycle re-entry)

  The margin is the signed logit of the event's button at the event frame
  (positive = agrees with the label). Finding (2026-07-28): sustainers hold
  fat positive X plateaus through the JC window; breakers emit razor spikes
  — the jc-margin SIGN separated sustainers from breakers 10/10.

  On a training fixture the labels are aligned to the window's final index,
  so no bridge offset applies (`offset: 0` semantics; the live-replay
  script's `-1` exists for bridge delay). When frames have been through
  `Data.shift_actions/2`, compute events on the SHIFTED stream — margins
  are then label-aligned by construction; family names can blur at large
  delays but the sign metric stays exact.
  """

  alias ExPhil.Eval.ShineChain

  @critical [:jc_event, :aerial_shine_event]
  @event_btn %{jc_event: :x, aerial_shine_event: :b, ground_shine_event: :b}

  # Buttons-head logit columns (see Policy head layout).
  @col_b 1
  @col_x 2

  @doc "Critical event types (the chain-breaking joints)."
  def critical, do: @critical

  @doc """
  Classify the event at a frame from `{player, ctrl, prev_ctrl}`.
  Returns the event atom or nil.
  """
  def event_of(player, ctrl, prev_ctrl) do
    x_edge = ctrl.button_x and not (prev_ctrl && prev_ctrl.button_x)
    b_edge = ctrl.button_b and not (prev_ctrl && prev_ctrl.button_b)

    case ShineChain.family(player.action) do
      :ground_reflect when x_edge -> :jc_event
      :aerial_jump when b_edge -> :aerial_shine_event
      :jumpsquat when b_edge -> :aerial_shine_event
      fam when fam not in [:ground_reflect, :air_reflect] and b_edge -> :ground_shine_event
      _ -> nil
    end
  end

  @doc """
  Event index list over a training-frame list: `[{t, event}]`.
  Epoch-invariant — compute once before the training loop.

  The event's FAMILY is read from the frame BEFORE the edge (`t - 1`):
  training frames pair state_t with the applied controller whose effect
  state_t already shows (the "fixture is a one-frame-shifted dataset"
  fact, LATENCY_ARCHITECTURE), so at the X-edge index the state is
  already jumpsquat and at the B-edge index the reflector is already
  out. The decision state — the one the expert rule keys on — is the
  previous frame. Measured on fox_multishine_closed.slp: current-frame
  classification finds 0 events; prev-frame finds all 373.

  Edges are only counted across CONSECUTIVE game frames (synthesis blocks
  splice non-adjacent frame numbers; an apparent edge across a splice is
  an artifact, the same boundary rule `precompute_frame_embeddings` uses
  for prev-action threading).

  Options: `:port` (default 1).
  """
  def events(frames, opts \\ []) do
    port = Keyword.get(opts, :port, 1)
    arr = List.to_tuple(frames)

    for t <- 1..(tuple_size(arr) - 1),
        f = elem(arr, t),
        p = elem(arr, t - 1),
        f.game_state.frame == p.game_state.frame + 1,
        ev = event_of(p.game_state.players[port], f.controller, p.controller),
        ev != nil,
        do: {t, ev}
  end

  @doc """
  Gather the per-event window tensors from a precomputed embedding matrix
  (`{total, embed}`) once, before the epoch loop. Only events with a full
  window (`t >= window - 1`) are kept; at most `:max_events` (default 512,
  evenly subsampled) to bound per-epoch cost.

  Returns `{stacked_windows, kept_events}` where `stacked_windows` is
  `{n, window, embed}` and `kept_events` is `[{t, event}]` — or nil when
  no events survive.
  """
  def prepare(embedded_frames, events, window, opts \\ []) do
    max_events = Keyword.get(opts, :max_events, 512)
    {total, _} = Nx.shape(embedded_frames)

    kept = Enum.filter(events, fn {t, _} -> t >= window - 1 and t < total end)

    kept =
      if length(kept) > max_events do
        every = div(length(kept), max_events) + 1
        Enum.take_every(kept, every)
      else
        kept
      end

    case kept do
      [] ->
        nil

      kept ->
        wins =
          Enum.map(kept, fn {t, _} ->
            Nx.slice_along_axis(embedded_frames, t - window + 1, window, axis: 0)
          end)

        {Nx.stack(wins), kept}
    end
  end

  @doc """
  One probe pass: run the policy on the prepared windows, read each
  event's button logit, reduce to per-phase stats.

  `predict_fn.(params, {n, window, embed})` must return the head tuple
  with buttons logits at element 0 (`{n, num_buttons}`) — the same
  contract as `BasinRollout.rollout/4`.

  Returns a flat map ready for JSONL merging:
  `%{jc_n:, jc_p10:, jc_min:, jc_flip:, aerial_n:, aerial_p10:, ...,
  crit_p10_min:}` (absent phases omitted; crit_p10_min nil if neither
  critical phase has events).

  Options: `:batch_size` (default 256).
  """
  def margins(predict_fn, params, {stacked, kept}, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 256)
    {n, window, embed} = Nx.shape(stacked)

    logits =
      0..(n - 1)
      |> Enum.chunk_every(batch_size)
      |> Enum.flat_map(fn idxs ->
        batch = Nx.slice_along_axis(stacked, hd(idxs), length(idxs), axis: 0)
        batch = Nx.reshape(batch, {length(idxs), window, embed})
        buttons = predict_fn.(params, batch) |> elem(0)
        b = buttons[[.., @col_b]] |> Nx.to_flat_list()
        x = buttons[[.., @col_x]] |> Nx.to_flat_list()
        Enum.zip(b, x)
      end)

    by_phase =
      Enum.zip(kept, logits)
      |> Enum.reduce(%{}, fn {{_t, ev}, {b, x}}, acc ->
        logit = if @event_btn[ev] == :b, do: b, else: x
        Map.update(acc, ev, [logit], &[logit | &1])
      end)

    stats(by_phase)
  end

  @doc """
  Per-phase margin stats, flattened for JSONL:
  n / mean / p10 / min / flip (fraction below zero), prefixed
  `jc_` / `aerial_` / `ground_`, plus `crit_p10_min` (min p10 over the
  critical phases — the sustain predictor).
  """
  def stats(by_phase) do
    prefix = %{jc_event: "jc", aerial_shine_event: "aerial", ground_shine_event: "ground"}

    flat =
      Enum.reduce(by_phase, %{}, fn {phase, margins}, acc ->
        sorted = Enum.sort(margins)
        n = length(margins)
        pre = prefix[phase]

        acc
        |> Map.put("#{pre}_n", n)
        |> Map.put("#{pre}_mean", Float.round(Enum.sum(margins) / n, 3))
        |> Map.put("#{pre}_p10", Float.round(percentile(sorted, 0.10) * 1.0, 3))
        |> Map.put("#{pre}_min", Float.round(hd(sorted) * 1.0, 3))
        |> Map.put("#{pre}_flip", Float.round(Enum.count(margins, &(&1 < 0)) / n, 3))
      end)

    crit_p10s =
      @critical
      |> Enum.map(&flat["#{prefix[&1]}_p10"])
      |> Enum.reject(&is_nil/1)

    crit = if crit_p10s == [], do: nil, else: Float.round(Enum.min(crit_p10s) * 1.0, 3)
    Map.put(flat, "crit_p10_min", crit)
  end

  defp percentile(sorted, p) do
    n = length(sorted)
    Enum.at(sorted, min(trunc(p * n), n - 1))
  end
end
