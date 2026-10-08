defmodule ExPhil.Training.SilentFallWeightingTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.SilentFallWeighting, as: SFW

  @neutral %{main_stick: %{x: 0.5, y: 0.5}, button_a: false, button_b: false, button_x: false, button_y: false,
             button_z: false, button_l: false, button_r: false}
  @up %{@neutral | main_stick: %{x: 0.5, y: 0.95}}

  defp frame(num, x, on_ground, ctrl, action \\ nil) do
    action = action || if(on_ground, do: 14, else: 29)

    %{
      game_state: %{
        frame: num,
        stage: 32,
        players: %{1 => %{x: x, y: if(on_ground, do: 0.0, else: -20.0), on_ground: on_ground, action: action, hitstun_frames_left: 0}}
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

  test "neutral? respects the deadzone and buttons" do
    assert SFW.neutral?(@neutral)
    assert SFW.neutral?(%{@neutral | main_stick: %{x: 0.6, y: 0.4}})
    refute SFW.neutral?(@up)
    refute SFW.neutral?(%{@neutral | button_b: true})
  end
end
