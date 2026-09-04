defmodule ExPhil.Interp.StyleFingerprintTest do
  use ExUnit.Case, async: true

  alias ExPhil.Interp.StyleFingerprint, as: FP
  alias ExPhil.Bridge.{GameState, Player, ControllerState}

  @wait 14

  defp controller(opts \\ []) do
    %ControllerState{
      main_stick: Keyword.get(opts, :main, %{x: 0.5, y: 0.5}),
      c_stick: Keyword.get(opts, :c, %{x: 0.5, y: 0.5}),
      l_shoulder: Keyword.get(opts, :l_shoulder, 0.0),
      r_shoulder: 0.0,
      button_a: Keyword.get(opts, :a, false),
      button_b: false,
      button_x: Keyword.get(opts, :x, false),
      button_y: Keyword.get(opts, :y, false),
      button_z: false,
      button_l: Keyword.get(opts, :bl, false),
      button_r: false,
      button_d_up: Keyword.get(opts, :d_up, false)
    }
  end

  defp player(action, ctrl) do
    %Player{
      character: 2, x: 0.0, y: 0.0, percent: 0.0, stock: 4, facing: 1,
      action: action, action_frame: 0, invulnerable: false, jumps_left: 2,
      on_ground: true, shield_strength: 60.0, hitstun_frames_left: 0,
      speed_air_x_self: 0.0, speed_ground_x_self: 0.0, speed_y_self: 0.0,
      speed_x_attack: 0.0, speed_y_attack: 0.0, nana: nil,
      controller_state: ctrl
    }
  end

  defp gs(action, ctrl_opts \\ []) do
    %GameState{
      frame: 0, stage: 32, menu_state: 2,
      players: %{1 => player(action, controller(ctrl_opts)), 2 => player(@wait, controller())},
      projectiles: [], items: [], distance: 10.0
    }
  end

  defp pad(n), do: List.duplicate(gs(@wait), n)

  test "vector/keys are aligned and stable" do
    fp = FP.fingerprint(pad(120), 1)
    assert length(FP.vector(fp)) == length(FP.keys())
    assert Enum.all?(FP.vector(fp), &is_number/1)
  end

  test "jump_x_ratio from X vs Y press edges" do
    # 3 X presses, 1 Y press (each press = one held frame between waits)
    states =
      Enum.flat_map(1..3, fn _ -> [gs(@wait), gs(@wait, x: true)] end) ++
        [gs(@wait), gs(@wait, y: true), gs(@wait)]

    fp = FP.fingerprint(states, 1)
    assert_in_delta fp.jump_x_ratio, 0.75, 1.0e-9
  end

  test "d_up press rate is per minute" do
    # Two d_up rising edges across exactly one minute of frames (edges are
    # detected from frame PAIRS, so a frame-0 press has no prior frame and
    # doesn't count — start with a pad)
    states = pad(500) ++ [gs(@wait, d_up: true)] ++ pad(1000) ++ [gs(@wait, d_up: true)] ++ pad(2098)
    fp = FP.fingerprint(states, 1)
    assert_in_delta fp.press_d_up_per_min, 2.0, 0.01
  end

  test "throw direction mix" do
    # up, up, forward throw entries (219 = forward, 221 = up)
    states =
      pad(5) ++ [gs(221)] ++ pad(5) ++ [gs(221)] ++ pad(5) ++ [gs(219)] ++ pad(5)

    fp = FP.fingerprint(states, 1)
    assert_in_delta fp.throw_up_mix, 2 / 3, 1.0e-9
    assert_in_delta fp.throw_forward_mix, 1 / 3, 1.0e-9
    assert fp.throw_back_mix == 0.0
  end

  test "tech option mix" do
    # 199 = tech in place, 200 = tech roll
    states = pad(5) ++ [gs(199)] ++ pad(5) ++ [gs(200)] ++ pad(5) ++ [gs(199)] ++ pad(5)
    fp = FP.fingerprint(states, 1)
    assert_in_delta fp.tech_in_place_mix, 2 / 3, 1.0e-9
    assert_in_delta fp.tech_roll_mix, 1 / 3, 1.0e-9
  end

  test "aerial mix and c-stick attribution" do
    # fair (66) entered with c-stick deflected; nair (65) with a-press
    states =
      pad(3) ++
        [gs(66, c: %{x: 0.9, y: 0.5})] ++
        pad(3) ++
        [gs(65, a: true)] ++ pad(3)

    fp = FP.fingerprint(states, 1)
    assert_in_delta fp.fair_mix, 0.5, 1.0e-9
    assert_in_delta fp.nair_mix, 0.5, 1.0e-9
    assert_in_delta fp.cstick_aerial_frac, 0.5, 1.0e-9
  end

  test "stick occupancy concentrates where the stick sits" do
    center = FP.fingerprint(pad(60), 1)
    assert_in_delta center.stick_cell_4, 1.0, 1.0e-9

    left = FP.fingerprint(List.duplicate(gs(@wait, main: %{x: 0.05, y: 0.5}), 60), 1)
    assert_in_delta left.stick_cell_3, 1.0, 1.0e-9
  end

  test "light shield vs full shield" do
    light = FP.fingerprint(List.duplicate(gs(@wait, l_shoulder: 0.4), 60), 1)
    assert_in_delta light.lightshield_frac, 1.0, 1.0e-9

    full = FP.fingerprint(List.duplicate(gs(@wait, l_shoulder: 1.0, bl: true), 60), 1)
    assert_in_delta full.lightshield_frac, 0.0, 1.0e-9
  end

  test "distance: identity is zero, difference is positive, symmetric" do
    a = FP.fingerprint(pad(120), 1)
    b = FP.fingerprint([gs(@wait, d_up: true)] ++ pad(119) ++ [gs(221)] ++ pad(20), 1)

    assert FP.distance(a, a) == 0.0
    assert FP.distance(a, b) > 0.0
    assert_in_delta FP.distance(a, b), FP.distance(b, a), 1.0e-9
  end

  test "missing controller data degrades to zeros, not crashes" do
    no_ctrl = %GameState{
      frame: 0, stage: 32, menu_state: 2,
      players: %{1 => player(@wait, nil), 2 => player(@wait, nil)},
      projectiles: [], items: [], distance: 10.0
    }

    fp = FP.fingerprint(List.duplicate(no_ctrl, 60), 1)
    assert fp.press_a_per_min == 0.0
    assert fp.jump_x_ratio == 0.0
  end
end
