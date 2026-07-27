defmodule ExPhil.Training.ScheduledSampling do
  @moduledoc """
  Scheduled sampling (Bengio et al. 2015) for the prev-action channel.

  Attacks exposure bias at TRAINING time: with probability `p` per sample,
  the prev-action slice of the LAST window position is replaced by the
  model's OWN decoded prediction for the previous frame, so the policy
  learns to act on the conditioning it will actually see live (its own
  bucket-decoded output) rather than the teacher's ground-truth controller.

  Scope: depth-1, last-position-only. The game-state dims cannot be
  self-sampled without a simulator — that is what DAgger / perturbation
  recordings are for. This covers the one input channel that is
  self-referential at inference time, and it composes with both.

  Decode parity is pinned to the live path (`Policy.to_controller_state/2`
  feeding `Controller.embed_continuous/1`):
  - buttons:  logit > 0 (sigmoid > 0.5), embedded as 0.0 / 1.0
  - sticks:   argmax bucket / axis_buckets, then (v - 0.5) * 2
  - shoulder: argmax bucket / shoulder_buckets (raw value, no rescale)

  Training loss under scheduled sampling is a strictly harder objective
  than teacher-forced loss — do not compare loss curves across the flag.
  """

  alias ExPhil.Training.Utils

  @doc """
  Build the jitted splice function: `fn params, states, mask -> states'`.

  `states` is `{batch, window, embed}`; `mask` is `{batch, 1}` f32 of
  0.0/1.0 (1.0 = use the model's own prediction for that sample).
  Generation runs the model on the window truncated by one frame, so its
  prediction for frame t-1 becomes the prev-action conditioning at the
  window's final position t. One extra forward pass per step (~+30%).

  Requires `config[:ss_prev_dims]` = `[offset, 13]` — locate it with
  `ExPhil.Interp.Attribution.prev_action_dim_range/1` (empirical discovery;
  a hand-written offset table would drift when the embedding layout moves).
  """
  def build(predict_fn, config) do
    [offset, width] = ss_prev_dims!(config)
    axis_buckets = config[:axis_buckets] || 16
    shoulder_buckets = config[:shoulder_buckets] || 4

    if config[:kmeans_centers] do
      raise ArgumentError,
            "scheduled_sampling supports uniform bucket decode only (kmeans_centers is set)"
    end

    fun = fn params, states, mask ->
      batch = Nx.axis_size(states, 0)
      window = Nx.axis_size(states, 1)

      truncated = Nx.slice_along_axis(states, 0, window - 1, axis: 1)

      {btn, mx, my, cx, cy, sh} =
        predict_fn.(Utils.ensure_model_state(params), truncated)

      buttons = btn |> Nx.greater(0.0) |> Nx.as_type(:f32)

      decode_stick = fn logits ->
        idx = logits |> Nx.argmax(axis: -1) |> Nx.as_type(:f32)
        # undiscretize_axis (idx / buckets) then embed_stick_continuous ((v-0.5)*2)
        idx
        |> Nx.divide(axis_buckets)
        |> Nx.subtract(0.5)
        |> Nx.multiply(2.0)
        |> Nx.new_axis(-1)
      end

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

      old =
        states
        |> Nx.slice([0, window - 1, offset], [batch, 1, width])
        |> Nx.squeeze(axes: [1])

      mixed =
        Nx.multiply(mask, ctrl)
        |> Nx.add(Nx.multiply(Nx.subtract(1.0, mask), old))
        |> Nx.as_type(Nx.type(states))

      Nx.put_slice(states, [0, window - 1, offset], Nx.new_axis(mixed, 1))
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
