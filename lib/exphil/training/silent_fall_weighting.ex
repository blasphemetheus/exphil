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
      has been NEUTRAL for at least `silent_fall_min` previous frames;
    * `onset_weight` (2026-10-08, INPUT_COHERENCE "10-08 06:10") — offstage
      falling frames whose label is a recovery DECISION ONSET: a jump button
      (X/Y) or B going from released on the previous frame to pressed on
      this one. The two weights above sharpen P(input | state) over the
      whole offstage slice — which is ~97 % holds, so every lever that fits
      that slice harder learns the hold harder (the danger-context head made
      the silent fall worse). This one weights only the frames where the
      expert decides, leaving the holds at 1.0, so the decision rows of
      `ExPhil.Eval.DecisionMap` (jump by height, Firefox once the jump is
      spent) are what gets more gradient.

    * `veto_weight` (2026-10-10, INPUT_COHERENCE "10-10 09:10") — on a
      relabelled DAgger frame that still carries the bot's own input
      (`:actual`, kept by `sim_recovery_dagger_expert.exs --keep-actual`),
      a frame where the BOT pressed a decision button (X/Y/B edge against
      its own previous input) and the expert label holds it released. The
      onset weight is asymmetric: it weights the expert's presses, never the
      expert's hold at a state where the learner pressed — and the one
      death mode that replicated on every seed (an on-stage Illusion fired
      at the edge, 100 % fatal) is a 0.04 %-per-frame excess the unweighted
      loss cannot see although the labels already say "don't". This is
      DAgger's standard emphasis on learner-error states, at the press
      level; it applies on and off stage.

    * `onset_edge_window` (2026-10-10, INPUT_COHERENCE "10-10 13:30") —
      with `onset_weight`, extend it to AIRBORNE frames within this many
      units inside the stage edge (the state the carried side-B deaths
      start from: airborne at the edge, facing out, the stick still held
      outward, no input). There the onset is a jump edge or a B edge WITH
      a stick-zone change on the same frame — the expert, pressing B from
      an outward stick in that state, turns the stick on the same frame
      95 % of the time (up → Firefox, centre → inward Illusion); the bot
      20 %. A B press that keeps the held stick (the fatal joint, which
      the expert also fires at 0.3 %/frame) is not lifted. nil = off.

  The first two weight both outcomes of a frame alike, so the conditional
  P(input | silent k) is sharpened, not biased; `onset_weight` deliberately
  biases toward acting in danger. Weights are `max(1.0, w)` per frame;
  frames outside the regimes stay 1.0. The silence counter and the previous
  controller reset at replay boundaries (frame-number jumps).
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
    on_w = Keyword.get(opts, :onset_weight)
    buttons_only = Keyword.get(opts, :onset_buttons_only, false)
    veto_w = Keyword.get(opts, :veto_weight)
    edge_win = Keyword.get(opts, :onset_edge_window)
    k_min = Keyword.get(opts, :silent_fall_min, 13)

    if off_w == nil and sf_w == nil and on_w == nil and veto_w == nil do
      nil
    else
      {weights, _silence, _prev_frame, _prev_controller, _prev_actual} =
        Enum.reduce(frames, {[], 0, nil, nil, nil}, fn frame, {acc, silence, prev_num, prev_c, prev_a} ->
          num = frame_number(frame)
          cont? = continuous?(prev_num, num)
          # the silence / controller seen BEFORE this frame: previous frames'
          silence = if cont?, do: silence, else: 0
          prev_c = if cont?, do: prev_c, else: nil
          prev_a = if cont?, do: prev_a, else: nil
          c = frame[:controller] || frame.controller
          actual = frame[:actual]
          off? = Data.frame_offstage?(frame)

          w = 1.0
          w = if off? and off_w != nil, do: max(w, off_w * 1.0), else: w

          w =
            if off? and sf_w != nil and silence >= k_min and falling?(frame),
              do: max(w, sf_w * 1.0),
              else: w

          w =
            if off? and on_w != nil and prev_c != nil and falling?(frame) and
                 onset?(prev_c, c, if(buttons_only, do: nil, else: jumps_left(frame))),
              do: max(w, on_w * 1.0),
              else: w

          w =
            if not off? and on_w != nil and edge_win != nil and prev_c != nil and falling?(frame) and
                 near_edge_airborne?(frame, edge_win) and edge_onset?(prev_c, c),
              do: max(w, on_w * 1.0),
              else: w

          w =
            if veto_w != nil and veto?(prev_a, actual, c),
              do: max(w, veto_w * 1.0),
              else: w

          silence = if neutral?(c), do: silence + 1, else: 0
          {[w | acc], silence, num, c, actual}
        end)

      Enum.reverse(weights)
    end
  end

  @veto_buttons [:button_x, :button_y, :button_b]

  @doc """
  True when the bot's own input (`actual`) presses a decision button (X, Y
  or B) that it did not hold on its previous own input (`prev_actual`) and
  the label `c` holds that button released — the learner pressed where the
  expert would not. False without a previous own input (trip start after a
  boundary) or on frames with no `:actual` (plain replay frames).
  """
  @spec veto?(map() | nil, map() | nil, map() | nil) :: boolean()
  def veto?(nil, _, _), do: false
  def veto?(_, nil, _), do: false
  def veto?(_, _, nil), do: false

  def veto?(prev_actual, actual, c) do
    Enum.any?(@veto_buttons, fn b ->
      Map.get(actual, b) && !Map.get(prev_actual, b) && !Map.get(c, b)
    end)
  end

  # main stick y (0..1) at or above this is "up" — a Firefox aim
  @stick_up 0.75

  @doc """
  True when this frame's controller starts a recovery decision the previous
  frame had not:

    * a jump button (X or Y) pressed now with no jump button held before;
    * a Firefox onset: B pressed now (released before) WITH the stick up, or
      the stick entering UP from not-up once the double jump is spent
      (`jumps_left` 0) — the expert aims a few frames before the press and
      `--stick-duration` supervises the stick only at its change events, so
      the aim is its own onset (INPUT_COHERENCE "10-08 08:00"). Callers pass
      `jumps_left` nil to leave this term out (`--onset-buttons-only`, 10-09:
      the term is blind to x and bent the Firefox fire angle).

  A B press with the stick elsewhere (side-B, neutral-B) is not an onset.
  """
  @spec onset?(map() | nil, map() | nil, integer() | nil) :: boolean()
  def onset?(prev, c, jumps_left \\ nil)
  def onset?(nil, _, _), do: false
  def onset?(_, nil, _), do: false

  def onset?(prev, c, jumps_left) do
    jump_now = Map.get(c, :button_x) || Map.get(c, :button_y)
    jump_prev = Map.get(prev, :button_x) || Map.get(prev, :button_y)
    b_edge = Map.get(c, :button_b) && !Map.get(prev, :button_b)
    up_edge = stick_up?(c) and not stick_up?(prev)

    !!((jump_now && !jump_prev) || (b_edge && stick_up?(c)) || (up_edge && jumps_left == 0))
  end

  @doc """
  The near-edge onset (`onset_edge_window`): a jump button edge, or a B edge
  with the main stick in a different zone (neutral / up / down / left /
  right) than on the previous frame — the press comes with a turn.
  """
  @spec edge_onset?(map() | nil, map() | nil) :: boolean()
  def edge_onset?(nil, _), do: false
  def edge_onset?(_, nil), do: false

  def edge_onset?(prev, c) do
    jump_now = Map.get(c, :button_x) || Map.get(c, :button_y)
    jump_prev = Map.get(prev, :button_x) || Map.get(prev, :button_y)
    b_edge = Map.get(c, :button_b) && !Map.get(prev, :button_b)

    !!((jump_now && !jump_prev) || (b_edge && stick_zone(c) != stick_zone(prev)))
  end

  @doc false
  def stick_zone(c) do
    ms = Map.get(c, :main_stick) || %{x: 0.5, y: 0.5}
    x = (ms[:x] || 0.5) - 0.5
    y = (ms[:y] || 0.5) - 0.5

    cond do
      abs(x) < @deadzone and abs(y) < @deadzone -> :neutral
      abs(y) >= abs(x) and y > 0 -> :up
      abs(y) >= abs(x) -> :down
      x > 0 -> :right
      true -> :left
    end
  end

  @doc """
  Airborne, alive and within `window` units inside the stage edge (or past
  it): the state the carried side-B deaths start from. Offstage frames are
  handled by the offstage terms, so callers test `not frame_offstage?` first.
  """
  @spec near_edge_airborne?(map(), number()) :: boolean()
  def near_edge_airborne?(frame, window) do
    gs = frame[:game_state] || frame.game_state
    p = gs && gs.players && gs.players[1]

    case p do
      %{on_ground: false, x: x} when is_number(x) ->
        edge = Melee.Stages.edge_ground_position(gs.stage) || 85.57
        abs(x) > edge - window

      _ ->
        false
    end
  end

  @doc false
  def stick_up?(nil), do: false

  def stick_up?(c) do
    ms = Map.get(c, :main_stick) || %{x: 0.5, y: 0.5}
    (ms[:y] || 0.5) >= @stick_up
  end

  defp jumps_left(frame) do
    gs = frame[:game_state] || frame.game_state
    p = gs && gs.players && gs.players[1]
    p && Map.get(p, :jumps_left)
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
