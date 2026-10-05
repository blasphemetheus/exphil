defmodule ExPhil.Training.SilentFallWeighting do
  @moduledoc """
  Per-frame LOSS weights for the rare offstage regimes (2026-10-05, the
  silent fall — INPUT_COHERENCE_2026-10-01.md "10-05 12:55").

  The bot's decided-trip deaths are input-free falls with the double jump in
  hand; teacher-forced it is calibrated through 24 frames of silence and
  under-fires in the ≥ 25-frame tail, which the expert visits ~4 frames per
  game. These weights multiply the imitation loss (the `:loss_weights`
  channel of `Data.batched_sequences/2`, keyed by each window's supervised
  frame) on:

    * `offstage_weight` — every frame where the subject is airborne beyond
      the ledge (the `--offstage-weight` knob, previously bptt-only);
    * `silent_fall_weight` — offstage frames where the subject's controller
      has been NEUTRAL for at least `silent_fall_min` previous frames.

  Both outcomes of such a frame (the expert keeps waiting / the expert acts)
  are weighted alike, so the conditional P(input | silent k) is sharpened,
  not biased. Weights are `max(1.0, w)` per frame; frames outside the
  regimes stay 1.0. The silence counter resets at replay boundaries
  (frame-number jumps) and whenever the controller is active.
  """

  alias ExPhil.Training.Data

  # |stick - 0.5| below this is "centred" — the probe's neutral zone
  # (|c| < 0.33 in -1..1 units).
  @deadzone 0.165

  @doc """
  Weights aligned with `frames` (a chunk's flat frame list). Returns `nil`
  when neither weight is set, so callers can skip the channel entirely.
  """
  @spec frame_weights([map()], keyword()) :: [float()] | nil
  def frame_weights(frames, opts) do
    off_w = Keyword.get(opts, :offstage_weight)
    sf_w = Keyword.get(opts, :silent_fall_weight)
    k_min = Keyword.get(opts, :silent_fall_min, 13)

    if off_w == nil and sf_w == nil do
      nil
    else
      {weights, _silence, _prev_frame} =
        Enum.reduce(frames, {[], 0, nil}, fn frame, {acc, silence, prev_num} ->
          num = frame_number(frame)
          # the silence seen BEFORE this frame: previous frames' controllers
          silence = if continuous?(prev_num, num), do: silence, else: 0
          off? = Data.frame_offstage?(frame)

          w = 1.0
          w = if off? and off_w != nil, do: max(w, off_w * 1.0), else: w

          w =
            if off? and sf_w != nil and silence >= k_min and falling?(frame),
              do: max(w, sf_w * 1.0),
              else: w

          silence = if neutral?(frame[:controller] || frame.controller), do: silence + 1, else: 0
          {[w | acc], silence, num}
        end)

      Enum.reverse(weights)
    end
  end

  @doc "True when the controller is centred with no button held."
  @spec neutral?(map() | nil) :: boolean()
  def neutral?(nil), do: true

  def neutral?(c) do
    ms = Map.get(c, :main_stick) || %{x: 0.5, y: 0.5}

    abs((ms[:x] || 0.5) - 0.5) < @deadzone and abs((ms[:y] || 0.5) - 0.5) < @deadzone and
      not Enum.any?([:button_a, :button_b, :button_x, :button_y, :button_z, :button_l, :button_r], &Map.get(c, &1))
  end

  # A silent fall is a silence the player could end: not hanging on the
  # ledge (CLIFF_* 252..263 — airborne, offstage and input-free for dozens of
  # frames, the confound that inflated the first silence measurements), not
  # dead/respawning (<= 13), not helpless (35), not in hitstun.
  @doc false
  def falling?(frame) do
    gs = frame[:game_state] || frame.game_state
    p = gs && gs.players && gs.players[1]

    case p do
      %{action: a} when is_integer(a) ->
        a > 13 and a != 35 and a not in 252..263 and (Map.get(p, :hitstun_frames_left) || 0) == 0

      _ ->
        false
    end
  end

  defp frame_number(frame) do
    gs = frame[:game_state] || frame.game_state
    gs && gs.frame
  end

  defp continuous?(prev, num) when is_integer(prev) and is_integer(num), do: num == prev + 1
  defp continuous?(_, _), do: false
end
