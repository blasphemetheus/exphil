defmodule ExPhil.Training.ScheduledSampling do
  @moduledoc """
  Scheduled sampling (Bengio et al. 2015) for the prev-action channel and
  its queue-as-input generalization.

  Attacks exposure bias at TRAINING time: with probability `p` per sample,
  the prev-action slice(s) of the LAST window position are replaced by the
  model's OWN decoded predictions, so the policy learns to act on the
  conditioning it will actually see live (its own bucket-decoded outputs)
  rather than the teacher's ground-truth controllers.

  Queue-as-input (`ss_queue_depth` K > 1): slot k of the committed-action
  queue holds the action committed k decision-steps ago, which in shifted
  training data is exactly the model's TARGET at window position t-k
  (`Data.shift_actions/2` relabels `:controller` in place, and queue slots
  are built from the already-shifted frames — so no shift bookkeeping is
  needed here, under any `--pipeline-offset`/`--shift-jitter`/multi-delay
  mix). Slot k is therefore filled with the model's decoded prediction on
  the window truncated by k frames. All K slots swap together under one
  per-sample mask — live, every slot is self-generated, so a mixed queue
  is a training-only artifact we avoid.

  Scope: last-position-only. The game-state dims cannot be self-sampled
  without a simulator — that is what DAgger / perturbation recordings are
  for. This covers the input channels that are self-referential at
  inference time, and it composes with both.

  Decode parity is pinned to the live path (`Policy.to_controller_state/2`
  feeding `Controller.embed_continuous/1`):
  - buttons:  logit > 0 (sigmoid > 0.5), embedded as 0.0 / 1.0
  - sticks:   argmax bucket / axis_buckets, then (v - 0.5) * 2
  - shoulder: argmax bucket / shoulder_buckets (raw value, no rescale)

  Cost: one extra forward pass per slot per step (~+30% each; K=4 roughly
  doubles step time). Training loss under scheduled sampling is a strictly
  harder objective than teacher-forced loss — do not compare loss curves
  across the flag.
  """

  alias ExPhil.Training.Utils

  @doc """
  Build the jitted splice function: `fn params, states, mask -> states'`.

  `states` is `{batch, window, embed}`; `mask` is `{batch, 1}` f32 of
  0.0/1.0 (1.0 = use the model's own predictions for that sample).
  For each slot k in 1..`ss_queue_depth` (default 1), generation runs the
  model on the window truncated by k frames, so its prediction for frame
  t-k becomes slot k's conditioning at the window's final position t.

  Requires `config[:ss_prev_dims]` = `[offset, 13]` — the queue block's
  first slot; locate it with
  `ExPhil.Interp.Attribution.prev_action_dim_range/1` (empirical discovery;
  a hand-written offset table would drift when the embedding layout moves).
  Slots are contiguous: slot k lives at `offset + (k-1) * 13`.

  Requires `window > ss_queue_depth` (each slot needs a non-empty
  truncated window).
  """
  def build(predict_fn, config) do
    [offset, width] = ss_prev_dims!(config)
    depth = config[:ss_queue_depth] || 1
    axis_buckets = config[:axis_buckets] || 16
    shoulder_buckets = config[:shoulder_buckets] || 4

    if config[:kmeans_centers] do
      raise ArgumentError,
            "scheduled_sampling supports uniform bucket decode only (kmeans_centers is set)"
    end

    unless is_integer(depth) and depth >= 1 do
      raise ArgumentError, "ss_queue_depth must be a positive integer (got #{inspect(depth)})"
    end

    fun = fn params, states, mask ->
      batch = Nx.axis_size(states, 0)
      window = Nx.axis_size(states, 1)

      if window <= depth do
        raise ArgumentError,
              "ss_queue_depth #{depth} needs window > #{depth} (got window #{window})"
      end

      model_state = Utils.ensure_model_state(params)

      decode_stick = fn logits ->
        idx = logits |> Nx.argmax(axis: -1) |> Nx.as_type(:f32)
        # undiscretize_axis (idx / buckets) then embed_stick_continuous ((v-0.5)*2)
        idx
        |> Nx.divide(axis_buckets)
        |> Nx.subtract(0.5)
        |> Nx.multiply(2.0)
        |> Nx.new_axis(-1)
      end

      Enum.reduce(1..depth, states, fn k, acc ->
        truncated = Nx.slice_along_axis(states, 0, window - k, axis: 1)

        {btn, mx, my, cx, cy, sh} = predict_fn.(model_state, truncated)

        buttons = btn |> Nx.greater(0.0) |> Nx.as_type(:f32)

        shoulder =
          sh
          |> Nx.argmax(axis: -1)
          |> Nx.as_type(:f32)
          |> Nx.divide(shoulder_buckets)
          |> Nx.new_axis(-1)

        ctrl =
          Nx.concatenate(
            [buttons, decode_stick.(mx), decode_stick.(my), decode_stick.(cx), decode_stick.(cy), shoulder],
            axis: 1
          )

        slot_offset = offset + (k - 1) * width

        old =
          states
          |> Nx.slice([0, window - 1, slot_offset], [batch, 1, width])
          |> Nx.squeeze(axes: [1])

        mixed =
          Nx.multiply(mask, ctrl)
          |> Nx.add(Nx.multiply(Nx.subtract(1.0, mask), old))
          |> Nx.as_type(Nx.type(states))

        Nx.put_slice(acc, [0, window - 1, slot_offset], Nx.new_axis(mixed, 1))
      end)
    end

    Nx.Defn.jit(fun, compiler: EXLA, on_conflict: :reuse)
  end

  defp ss_prev_dims!(config) do
    case config[:ss_prev_dims] do
      [offset, width] when is_integer(offset) and width == 13 -> [offset, width]
      {offset, width} when is_integer(offset) and width == 13 -> [offset, width]
      other ->
        raise ArgumentError,
              "scheduled_sampling needs config[:ss_prev_dims] = [offset, 13] " <>
                "(got #{inspect(other)}); use Attribution.prev_action_dim_range/1"
    end
  end
end
