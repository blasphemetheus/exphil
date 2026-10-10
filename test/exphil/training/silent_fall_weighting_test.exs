defmodule ExPhil.Training.SilentFallWeightingTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.SilentFallWeighting, as: SFW

  @neutral %{main_stick: %{x: 0.5, y: 0.5}, button_a: false, button_b: false, button_x: false, button_y: false,
             button_z: false, button_l: false, button_r: false}
  @up %{@neutral | main_stick: %{x: 0.5, y: 0.95}}

  defp frame(num, x, on_ground, ctrl, action \\ nil, jumps_left \\ 1) do
    action = action || if(on_ground, do: 14, else: 29)

    %{
      game_state: %{
        frame: num,
        stage: 32,
        players: %{1 => %{x: x, y: if(on_ground, do: 0.0, else: -20.0), on_ground: on_ground, action: action, hitstun_frames_left: 0, jumps_left: jumps_left}}
      },
      controller: ctrl
    }
  end

  test "ledge hangs and helpless frames are not silent falls (but still offstage)" do
    frames = [frame(0, 100.0, false, @neutral, 253), frame(1, 100.0, false, @neutral, 253), frame(2, 100.0, false, @neutral, 35)]
    assert SFW.frame_weights(frames, silent_fall_weight: 3.0, silent_fall_min: 1) == [1.0, 1.0, 1.0]
    assert SFW.frame_weights(frames, offstage_weight: 2.0) == [2.0, 2.0, 2.0]
  end

  test "nil when neither knob is set" do
    assert SFW.frame_weights([frame(0, 100.0, false, @neutral)], []) == nil
  end

  test "offstage_weight lifts every offstage frame; onstage frames stay 1.0" do
    frames = [frame(0, 10.0, true, @neutral), frame(1, 100.0, false, @neutral), frame(2, 100.0, false, @up)]
    assert SFW.frame_weights(frames, offstage_weight: 4.0) == [1.0, 4.0, 4.0]
  end

  test "silent_fall_weight needs k_min previous neutral frames offstage, both outcomes weighted" do
    # 3 neutral offstage frames, then the expert acts on the 4th, then neutral again
    frames =
      for {ctrl, i} <- Enum.with_index([@neutral, @neutral, @neutral, @up, @neutral]),
          do: frame(i, 100.0, false, ctrl)

    # silence before frame i: 0,1,2,3,0 -> with k_min 2 frames 2 and 3 (the "act" frame) are weighted
    assert SFW.frame_weights(frames, silent_fall_weight: 6.0, silent_fall_min: 2) == [1.0, 1.0, 6.0, 6.0, 1.0]
  end

  test "silence counter resets at replay boundaries and only counts offstage frames as targets" do
    neutral_onstage = for i <- 0..4, do: frame(i, 10.0, true, @neutral)
    # new replay (frame number jumps back): the onstage silence must not carry over
    offstage = for i <- 0..2, do: frame(i, 100.0, false, @neutral)
    ws = SFW.frame_weights(neutral_onstage ++ offstage, silent_fall_weight: 3.0, silent_fall_min: 2)
    # onstage frames never weighted; offstage frames reach k_min only on the 3rd
    assert ws == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]
  end

  test "onset_weight lifts only the offstage frame where a jump button or B is first pressed" do
    jump = %{@neutral | button_x: true}
    firefox = %{@up | button_b: true}
    # hold, jump edge, jump held, release, up-B edge, up-B held
    ctrls = [@neutral, jump, jump, @neutral, firefox, firefox]
    frames = for {c, i} <- Enum.with_index(ctrls), do: frame(i, 100.0, false, c)
    assert SFW.frame_weights(frames, onset_weight: 5.0) == [1.0, 5.0, 1.0, 1.0, 5.0, 1.0]

    # the same edges onstage, on the ledge, or helpless are not weighted
    onstage = for {c, i} <- Enum.with_index(ctrls), do: frame(i, 10.0, true, c)
    assert SFW.frame_weights(onstage, onset_weight: 5.0) == List.duplicate(1.0, 6)
    ledge = for {c, i} <- Enum.with_index(ctrls), do: frame(i, 100.0, false, c, 253)
    assert SFW.frame_weights(ledge, onset_weight: 5.0) == List.duplicate(1.0, 6)

    # a replay boundary has no previous controller: the first frame is never an onset
    assert SFW.frame_weights([frame(0, 100.0, false, jump)], onset_weight: 5.0) == [1.0]
    assert SFW.frame_weights([frame(7, 100.0, false, @neutral), frame(0, 100.0, false, jump)], onset_weight: 5.0) == [1.0, 1.0]

    # composes with offstage_weight by max
    assert SFW.frame_weights(Enum.take(frames, 3), offstage_weight: 3.0, onset_weight: 5.0) == [3.0, 5.0, 3.0]
  end

  test "veto_weight lifts the relabelled frame where the bot pressed X/Y/B and the label holds — on or off stage" do
    side_b = %{@neutral | main_stick: %{x: 0.95, y: 0.5}, button_b: true}
    side = %{@neutral | main_stick: %{x: 0.95, y: 0.5}}
    with_actual = fn f, a -> Map.put(f, :actual, a) end
    # bot: hold, side-B edge, side-B held, release; label: holds the stick sideways, never B
    bot = [@neutral, side_b, side_b, @neutral]
    frames = for {a, i} <- Enum.with_index(bot), do: with_actual.(frame(i, 80.0, true, side), a)
    assert SFW.frame_weights(frames, veto_weight: 8.0) == [1.0, 8.0, 1.0, 1.0]
    # the label agreeing (B pressed too) is not a veto; a bot HOLD is never one
    agree = List.update_at(frames, 1, &%{&1 | controller: side_b})
    assert SFW.frame_weights(agree, veto_weight: 8.0) == List.duplicate(1.0, 4)
    # frames without the bot's own input (plain replay frames) are untouched; nil when only this knob is unset
    assert SFW.frame_weights(Enum.map(frames, &Map.delete(&1, :actual)), veto_weight: 8.0) == List.duplicate(1.0, 4)
    assert SFW.frame_weights(frames, []) == nil
    # the first frame after a boundary has no previous own input
    assert SFW.frame_weights([with_actual.(frame(0, 80.0, true, side), side_b)], veto_weight: 8.0) == [1.0]
    # composes with the other knobs by max, offstage too
    off = for {a, i} <- Enum.with_index(bot), do: with_actual.(frame(i, 100.0, false, side), a)
    assert SFW.frame_weights(off, offstage_weight: 3.0, veto_weight: 8.0) == [3.0, 8.0, 3.0, 3.0]
    assert SFW.veto?(@neutral, %{@neutral | button_x: true}, @neutral)
    refute SFW.veto?(@neutral, %{@neutral | button_a: true}, @neutral)
  end

  test "a Firefox is two onsets (stick up once the jump is spent, then B with the stick up); side-B is none" do
    side = %{@neutral | main_stick: %{x: 0.95, y: 0.5}}
    side_b = %{side | button_b: true}
    up_b = %{@up | button_b: true}
    # jump spent: hold side, side-B edge, release, stick up (aim), up-B edge, held
    ctrls = [side, side_b, @neutral, @up, up_b, up_b]
    spent = for {c, i} <- Enum.with_index(ctrls), do: frame(i, 100.0, false, c, nil, 0)
    assert SFW.frame_weights(spent, onset_weight: 5.0) == [1.0, 1.0, 1.0, 5.0, 5.0, 1.0]

    # with a jump in hand the aim alone is not an onset; B with the stick up still is
    in_hand = for {c, i} <- Enum.with_index(ctrls), do: frame(i, 100.0, false, c, nil, 1)
    assert SFW.frame_weights(in_hand, onset_weight: 5.0) == [1.0, 1.0, 1.0, 1.0, 5.0, 1.0]
  end

  test "neutral? respects the deadzone and buttons" do
    assert SFW.neutral?(@neutral)
    assert SFW.neutral?(%{@neutral | main_stick: %{x: 0.6, y: 0.4}})
    refute SFW.neutral?(@up)
    refute SFW.neutral?(%{@neutral | button_b: true})
  end
end
