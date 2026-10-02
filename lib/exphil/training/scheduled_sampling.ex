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

  @doc """
  Autoregressive-head variant (2026-10-02): `fn params, states, mask -> states'`.

  Closer to the original recipe (Bengio et al. 2015, "Scheduled Sampling for
  Sequence Prediction with Recurrent Neural Networks") than the
  independent-head splice above in two ways:

  - SAMPLED feedback, not argmax: the trunk runs on a truncated window and
    the head is sampled sequentially with the live sampler
    (`Policy.Sampling.sample_autoregressive_from_features/3`, temperature
    1.0) — the previous input the model will actually see in play.
  - MULTI-STEP feedback: `config[:ss_steps]` K (default 1) trailing window
    positions are regenerated in order, oldest first, each sample drawn from
    a window that already contains the earlier self-generated inputs. The
    paper unrolls the model over the whole sequence; K is how far back that
    unroll starts here (K = window - 1 would be the full thing).

  The mixing probability is supplied per step by the train loop (a ramp from
  0, see `ss_ramp_start` / `ss_ramp_steps`): the paper's "schedule" — early
  samples are noise, and a flat rate from step 0 teaches the model to ignore
  the channel. One per-sample mask covers all K positions.

  Slot depth 1 only. Costs K trunk forwards + K head samples per step.
  """
  def build_autoregressive(trunk_predict_fn, config) do
    [offset, width] = ss_prev_dims!(config)
    axis_buckets = config[:axis_buckets] || 16
    shoulder_buckets = config[:shoulder_buckets] || 4
    steps = config[:ss_steps] || 1

    if config[:kmeans_centers], do: raise(ArgumentError, "scheduled_sampling supports uniform bucket decode only")
    if (config[:ss_queue_depth] || 1) != 1, do: raise(ArgumentError, "AR scheduled sampling supports queue depth 1 only")
    unless is_integer(steps) and steps >= 1, do: raise(ArgumentError, "ss_steps must be a positive integer")

    # one jitted splice per position-from-the-end k (static slice starts)
    splices =
      Map.new(1..steps, fn k ->
        {k,
         Nx.Defn.jit(
           fn states, mask, buttons, mx, my, cx, cy, sh ->
             batch = Nx.axis_size(states, 0)
             pos = Nx.axis_size(states, 1) - k
             stick = fn idx -> idx |> Nx.as_type(:f32) |> Nx.divide(axis_buckets) |> Nx.subtract(0.5) |> Nx.multiply(2.0) |> Nx.reshape({batch, 1}) end

             ctrl =
               Nx.concatenate(
                 [Nx.as_type(buttons, :f32), stick.(mx), stick.(my), stick.(cx), stick.(cy),
                  sh |> Nx.as_type(:f32) |> Nx.divide(shoulder_buckets) |> Nx.reshape({batch, 1})],
                 axis: 1
               )

             old = states |> Nx.slice([0, pos, offset], [batch, 1, width]) |> Nx.squeeze(axes: [1])

             mixed =
               Nx.multiply(mask, ctrl)
               |> Nx.add(Nx.multiply(Nx.subtract(1.0, mask), old))
               |> Nx.as_type(Nx.type(states))

             Nx.put_slice(states, [0, pos, offset], Nx.new_axis(mixed, 1))
           end,
           compiler: EXLA,
           on_conflict: :reuse
         )}
      end)

    fn params, states, mask ->
      window = Nx.axis_size(states, 1)
      if window <= steps, do: raise(ArgumentError, "ss_steps #{steps} needs window > #{steps}")
      model_state = Utils.ensure_model_state(params)

      # oldest regenerated position first: position (window - k) holds the
      # action of frame (window - k - 1), sampled from frames [0, window - k)
      Enum.reduce(steps..1//-1, states, fn k, acc ->
        truncated = Nx.slice_along_axis(acc, 0, window - k, axis: 1)
        features = trunk_predict_fn.(model_state, truncated)
        s = ExPhil.Networks.Policy.Sampling.sample_autoregressive_from_features(params, features, temperature: 1.0)
        splices[k].(acc, mask, Nx.reshape(s.buttons, {:auto, 8}), s.main_x, s.main_y, s.c_x, s.c_y, s.shoulder)
      end)
    end
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
