defmodule ExPhil.Interp.StyleTimingTest do
  use ExUnit.Case, async: true
  alias ExPhil.Interp.StyleTiming

  # Minimal per-frame player + controller maps. Only the fields the
  # detectors read are set; everything else is absent on purpose.
  defp pl(action, extra \\ %{}),
    do: Map.merge(%{action: action, x: 0.0, y: 0.0, on_ground: action < 25}, extra)

  defp ct(extra \\ %{}),
    do: Map.merge(%{main_stick_x: 0.5, main_stick_y: 0.5, l_trigger: 0.0, r_trigger: 0.0, button_l: false, button_r: false, button_z: false}, extra)

  defp run(pairs) do
    {players, controllers} = Enum.unzip(pairs)
    StyleTiming.features(players, controllers)
  end

  test "empty situations give zeros for every key" do
    f = run(for _ <- 1..10, do: {pl(14), ct()})
    assert Map.keys(f) |> Enum.sort() == Enum.sort(StyleTiming.keys())
    assert Enum.all?(f, fn {_, v} -> v == 0.0 end)
  end

  test "L-cancel press offset counts frames before the aerial landing" do
    # aerial (65) for 10 frames; L pressed 3 frames before landing (70)
    frames =
      for i <- 0..9, do: {pl(65), ct(%{button_l: i >= 7})}

    frames = frames ++ [{pl(70), ct(%{button_l: true})}, {pl(70), ct(%{button_l: true})}, {pl(14), ct()}]
    f = run(frames)
    assert f.lcancel_attempt_frac == 1.0
    assert f.lcancel_press_offset_mean == 3.0
    assert f.lcancel_press_offset_cv == 0.0
  end

  test "analog shoulder press counts as an L-cancel attempt; no press is a miss" do
    a = for i <- 0..9, do: {pl(65), ct(%{r_trigger: if(i >= 8, do: 0.6, else: 0.0)})}
    a = a ++ [{pl(71), ct(%{r_trigger: 0.6})}, {pl(14), ct()}]
    b = for _ <- 0..9, do: {pl(65), ct()}
    b = b ++ [{pl(72), ct()}, {pl(14), ct()}]
    f = run(a ++ b)
    assert f.lcancel_attempt_frac == 0.5
    assert f.lcancel_press_offset_mean == 2.0
  end

  test "wavedash angle and jumpsquat frame come from the airdodge frame's stick" do
    # stand, 3 frames of jumpsquat, airdodge (stick down-forward at 45°) for 2 frames, special landing
    stick = ct(%{main_stick_x: 0.5 + 0.35, main_stick_y: 0.5 - 0.35})

    frames =
      [{pl(14), ct()}] ++
        for(_ <- 1..3, do: {pl(24), ct()}) ++
        [{pl(25), ct()}, {pl(236), stick}, {pl(236), stick}] ++
        for(_ <- 1..5, do: {pl(43), ct()}) ++ [{pl(14), ct()}]

    f = run(frames)
    assert_in_delta f.wavedash_angle_mean, 45.0, 0.01
    assert f.wavedash_angle_cv == 0.0
    assert f.wavedash_jumpsquat_frame_mean == 4.0
  end

  test "an airdodge that does not land specially is not a wavedash" do
    frames = [{pl(14), ct()}, {pl(24), ct()}, {pl(25), ct()}, {pl(236), ct(%{main_stick_y: 0.1})}] ++ for(_ <- 1..10, do: {pl(29), ct()})
    assert run(frames).wavedash_angle_mean == 0.0
  end

  test "DI perpendicularity is 1 for a stick at right angles to knockback" do
    frames = [
      {pl(14), ct()},
      {pl(75, %{x: 0.0}), ct(%{main_stick_y: 0.95})},
      {pl(75, %{x: 3.0}), ct()},
      {pl(75, %{x: 6.0}), ct()},
      {pl(14), ct()},
      {pl(78, %{x: 6.0}), ct()},
      {pl(78, %{x: 9.0}), ct()},
      {pl(78, %{x: 12.0}), ct()}
    ]
    f = run(frames)
    assert f.di_active_frac == 0.5
    assert_in_delta f.di_perp_mean, 1.0, 1.0e-9
  end

  test "OOS latency is frames from leaving shieldstun to the first non-shield action" do
    frames =
      [{pl(179), ct()}, {pl(181), ct()}, {pl(181), ct()}] ++
        for(_ <- 1..4, do: {pl(179), ct()}) ++ [{pl(24), ct()}, {pl(25), ct()}]

    f = run(frames)
    assert f.oos_latency_mean == 4.0
    assert f.oos_jump_frac == 1.0
  end

  test "short hops are jumps whose takeoff speed is well below the game's max" do
    jump = fn apex ->
      [{pl(14), ct()}, {pl(24), ct()}] ++
        Enum.map([0.3, 0.7, 1.0, 0.7, 0.3], &{pl(25, %{y: &1 * apex}), ct()}) ++ [{pl(42), ct()}]
    end

    f = run(jump.(40.0) ++ jump.(12.0) ++ jump.(13.0) ++ jump.(39.0))
    assert f.short_hop_frac == 0.5
  end

  test "fingerprint vector includes the timing keys and stays finite" do
    alias ExPhil.Interp.StyleFingerprint, as: FP
    assert Enum.all?(StyleTiming.keys(), &(&1 in FP.keys()))
    assert Enum.all?(StyleTiming.invariant_keys(), &(&1 in FP.invariant_keys()))
  end
end
